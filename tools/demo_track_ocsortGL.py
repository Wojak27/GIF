import argparse
import copy
import json
import os
import os.path as osp
import time
import cv2
import torch

from loguru import logger

from ap3d_reid.reid_eval import AP3DCLIPReID, AP3DReID
from clip_reid.reid_eval import CLIPExtraProjectionLayer, CLIPReID
from trackers.ocsort_tracker.ocsort_ReID import OCSortReID
from yolox.data.data_augment import preproc
from yolox.evaluators.mot_evaluator_nsvatrackv3_ReID import group_gallery_like_text, load_gallery_features, load_text_features
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import get_color, plot_tracking
from trackers.tracking_utils.timer import Timer
import numpy as np


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

from utils.args import make_parser

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def plot_tracking_with_class(image, tlwhs, obj_ids, labels, frame_id=0, fps=0.):
    im = np.ascontiguousarray(np.copy(image))
    text_scale = 1.0
    text_thickness = 2
    line_thickness = 3

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i][0])
        obj_cls = int(obj_ids[i][1])
        label = labels.get(obj_cls, "unknown")
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, f"{label}", (intbox[0], intbox[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), thickness=text_thickness)
    return im


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
        return outputs, img_info

def make_player_labels(info_file_path):
    file = json.load(open(info_file_path))
    labels = {}
    
    for label in file:
        labels[int(label.split("_")[-1])] = file[label]["player_name"]
        if file[label]["long_video_appearances"] != []:
            labels[int(label.split("_")[-1])] = labels[int(label.split("_")[-1])] + " " + file[label]["long_video_appearances"][0]["jersey_number"]
        else:
            labels[int(label.split("_")[-1])] = labels[int(label.split("_")[-1])] +" " + file[label]["short_video_appearances"][0]["jersey_number"]
            
    return labels

def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo_type == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    assert args.id_level in ["global", "game"], "id_level must be in ['global','game']"
    assert args.use_age_with_gl in [None,"youngest", "oldest"], "use_age_with_gl must be in [None,'youngest','oldest']"
    assert args.use_text == False if args.id_level == "global" else True, "We currently only support gallery for the global mode"
    assert args.use_text == False if args.feature_extractor == "ap3d" else True, "We do not support text with AP3D"
    assert args.players_info_file != None, "Need players info file"
    
    player_labels = make_player_labels(args.players_info_file)
    id_level = args.id_level
    use_gallery = args.use_gallery
    assert args.gallery_path != "", "Provide path to gallery features"
    all_gallery_features = load_gallery_features(args.gallery_path)
    use_text = args.use_text
    if use_text or id_level == "game":
        assert args.text_path != "" if args.id_level == "game" else True, "Provide path to text features"
        all_text_features = load_text_features(args.text_path)
        all_gallery_features = group_gallery_like_text(all_gallery_features, all_text_features)
    if args.demo_type == "video":
        save_path = args.out_path
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = OCSortReID(det_thresh = args.track_thresh, iou_threshold=args.iou_thresh,
                                asso_func=args.asso, delta_t=args.deltat, inertia=args.inertia, use_byte=args.use_byte)
    if args.feature_extractor == "ap3d":
        encoder = AP3DReID(ckpt_path="models/best_model_ap3d.pth.tar")
    elif args.feature_extractor == "both":
        encoder = AP3DCLIPReID(ckpt_path="models/best_model_ap3d.pth.tar")
    else:
        encoder = CLIPReID(model_name="openai/clip-vit-large-patch14")
    
    if args.use_clip_projection:
        assert args.clip_projection_weights != "", "You need to provide weights for the projection"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        extra_clip_projection = CLIPExtraProjectionLayer().to(device)
        extra_clip_projection.load_state_dict(torch.load(args.clip_projection_weights, weights_only=True))
        extra_clip_projection.eval()
    assert args.video_name != None, "You need to provide the name of the video"
    video_name = args.video_name
    if id_level in ["game"]:
        if use_text and use_gallery and not args.feature_extractor == "both":
            player_ids = []
            combined_embeddings = []
            # Assume video_name is the game id key in all_text_features (a dict: {gameid: {playerid: text_emb}})
            game_name = video_name.split("-")[0]
            if game_name in all_text_features:
                for pid, text_emb in all_text_features[game_name].items():
                    if pid in all_gallery_features[game_name]:
                        gallery_emb = all_gallery_features[game_name][pid]
                        combined = (gallery_emb + text_emb) / 2.0
                        combined_embeddings.append(combined)
                        player_ids.append(pid)
            gallery_feature = torch.stack(combined_embeddings).squeeze() if combined_embeddings else None
            if args.use_clip_projection:
                gallery_feature = extra_clip_projection(gallery_feature.to(device)).cpu()
        elif use_gallery: #just use gallery embeddings
            player_ids_gallery = []
            embeddings = []
            # Assume video_name is the game id key in all_text_features (a dict: {gameid: {playerid: text_emb}})
            game_name = video_name.split("-")[0] # some videos are subsequences of frames of a different video
            if game_name in all_gallery_features:
                for pid, emb in all_gallery_features[game_name].items():
                    combined = all_gallery_features[game_name][pid]
                    embeddings.append(combined)
                    player_ids_gallery.append(pid)
            gallery_feature = torch.stack(embeddings).squeeze() if embeddings else None
            player_ids = player_ids_gallery.copy()
            if args.use_clip_projection and not args.feature_extractor == "both":
                gallery_feature = extra_clip_projection(gallery_feature.to(device)).cpu()
        else: # use text
            player_ids = []
            embeddings = []
            # Assume video_name is the game id key in all_text_features (a dict: {gameid: {playerid: text_emb}})
            game_name = video_name.split("-")[0] # some videos are subsequences of frames of a different video
            if game_name in all_text_features:
                for pid, emb in all_text_features[game_name].items():
                    combined = all_text_features[game_name][pid]
                    embeddings.append(torch.tensor(combined))
                    player_ids.append(pid)
            gallery_feature = torch.stack(embeddings).squeeze() if embeddings else None
            if args.use_clip_projection:
                gallery_feature = extra_clip_projection(gallery_feature.to(device)).cpu()
        if args.feature_extractor == "both":
            player_ids = []
            embeddings = []
            # Assume video_name is the game id key in all_text_features (a dict: {gameid: {playerid: text_emb}})
            game_name = video_name.split("-")[0] # some videos are subsequences of frames of a different video
            if game_name in all_text_features:
                for pid, emb in all_text_features[game_name].items():
                    if pid not in player_ids_gallery:
                        continue
                    combined = all_text_features[game_name][pid]
                    embeddings.append(torch.tensor(combined))
                    player_ids.append(pid)
            tmp_gallery_feature = torch.stack(embeddings).squeeze() if embeddings else None
            if args.use_clip_projection:
                tmp_gallery_feature = extra_clip_projection(tmp_gallery_feature.to(device)).cpu()
            gallery_feature = {
                "clip": tmp_gallery_feature,
                "ap3d":gallery_feature
            }

    else:
        embeddings = []
        player_ids = []
        for pid, emb in all_gallery_features.items():
            embeddings.append(torch.tensor(emb))
            player_ids.append(pid)
        gallery_feature = torch.stack(embeddings).squeeze() if embeddings else None
        if args.use_clip_projection:
                gallery_feature = extra_clip_projection(gallery_feature.to(device)).cpu()
    timer = Timer()
    frame_id = 0
    results = []
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            if outputs[0] is not None:
                raw_img = frame
                raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                raw_img=torch.tensor(raw_img).permute(-1,0,1).unsqueeze(0)
                bbox_xyxy = copy.deepcopy(outputs[0][:, :4])
                query_feature = encoder.inference(raw_img, bbox_xyxy.cpu().detach().numpy())
                if args.use_clip_projection:
                    if query_feature != None:
                        f = torch.tensor(query_feature).squeeze().to(device)
                        if len(f.shape) != 2:
                            f = f.unsqueeze(0)
                        query_feature = extra_clip_projection(f).cpu()
                if len(query_feature.shape) != 2:
                    query_feature = query_feature.unsqueeze(0)
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']],  exp.test_size, query_feature, gallery_feature, player_ids, args.reid_thresh, args.average_query_embeds, args)
                online_tlwhs = []
                online_ids = []
                for t in online_targets:
                    tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                    tid = t[4]
                    cid = int(t[-1])
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append((tid, cid))
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1.0,-1,-1,-1\n"
                        )
                timer.toc()
                online_im = plot_tracking_with_class(
                    img_info['raw_img'], online_tlwhs, online_ids, player_labels , frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']
            if args.save_result:
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def main(exp, args):
    if not args.expn:
        args.expn = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.expn)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    assert args.demo_type == "video", "We support only videos"
    imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)

