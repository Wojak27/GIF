from collections import defaultdict
import copy
import pickle
import cv2
from loguru import logger
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

import torch

from ap3d_reid.reid_eval import AP3DReID
from clip_reid.reid_eval import CLIPExtraProjectionLayer, CLIPReID
from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)
from trackers.ocsort_tracker.ocsort_ReID import OCSortReID
import numpy as np

import contextlib
import io
import os
import itertools
import json
import tempfile
import time
from utils.utils import write_results, write_results_and_category_no_score, write_results_no_score

def load_gallery_features(gallery_path):
    temp_features = defaultdict(list)
    for file in os.listdir(gallery_path):
        if file.endswith(".pt"):
            player_id = int(file[:4])
            emb = torch.load(os.path.join(gallery_path, file))
            temp_features[player_id].append(emb)
    all_gallery_features = {}
    for pid, feats in temp_features.items():
        all_gallery_features[pid] = torch.mean(torch.stack(feats), dim=0)
    return all_gallery_features

def load_text_features(text_path):
    all_text_features = defaultdict(dict)
    for file in os.listdir(text_path):
        if file.endswith(".pt"):
            base = file[:-3]
            gameid_str, _, playerid_str = base.partition("_player_")
            gameid = gameid_str
            playerid = int(playerid_str)
            emb = torch.load(os.path.join(text_path, file))
            all_text_features[gameid][playerid] = emb
    return all_text_features

def group_gallery_like_text(all_gallery_features, all_text_features):
    grouped_gallery = defaultdict(dict)
    for gameid, players in all_text_features.items():
        for pid in players:
            if pid in all_gallery_features:
                grouped_gallery[gameid][pid] = all_gallery_features[pid]
    return grouped_gallery

def get_players_for_frame(game_ranges, game_id, frame_id):
    for start_frame, end_frame, player_ids in game_ranges.get(game_id, []):
        if start_frame <= frame_id <= end_frame:
            return player_ids
    return []


class MOTEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, args, dataloader, img_size, confthre, nmsthre, num_classes):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        assert args.yolo_model in ["yolox_s","yolox_m","yolox_l","yolox_x"], "Invalid model name"
        assert args.yolo_model in args.ckpt, "Possibly invalid model path, model name not in path"
        self.args = args
        

    def evaluate_bytetrack(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.
        NOTE: This function will change training mode to False, please save states if needed.
        Args:
            model : model to evaluate.
        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1


        from trackers.byte_tracker.byte_tracker import BYTETracker
        tracker = BYTETracker(self.args)
        video_id = 0
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[-3]
                img_name = img_file_name[0].split("/")[-1]

                if frame_id == 1:
                    tracker = BYTETracker(self.args)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results(result_filename, results)
                        results = []
                if video_name not in video_names.values():
                    video_names[video_id] = video_name
                    video_id += 1

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start
    
            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                if tlwh[2] * tlwh[3] > self.args.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            # save results
            results.append((frame_id, online_tlwhs, online_ids, online_scores))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id-1]))
                write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results
    

    def evaluate_ocsort(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
        per_frame_ids=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.
        NOTE: This function will change training mode to False, please save states if needed.
        Args:
            model : model to evaluate.
        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        assert self.args is not None, "args should not be None"
        args = self.args
        if args.dataset == "mupnit":
            DATASET_ROOT = os.environ.get('MUPNIT_DATASET_ROOT', '/4TBSSD_Permanent/datasets/MuPNIT_30fps_global')
        cache_dir = f"{args.dataset}_cache_{args.yolo_model}_{args.feature_extractor}"
        reid_thresh = args.reid_thresh
        
        assert args.id_level in ["global", "game"], "id_level must be in ['global','game']"
        assert args.use_age_with_gl in [None,"youngest", "oldest"], "use_age_with_gl must be in [None,'youngest','oldest']"
        id_level = args.id_level
        use_gallery = args.use_gallery
        use_text = args.use_text

        evaluate_identification = use_text or use_gallery
        all_gallery_features = {}
        all_text_features = {}

        if evaluate_identification:
            assert args.use_text == False if args.id_level == "global" else True, "We currently only support gallery for the global mode"
            assert args.use_text == False if args.feature_extractor == "ap3d" else True, "We do not support text with AP3D"
            assert args.gallery_path != "", "Provide path to gallery features"
            all_gallery_features = load_gallery_features(args.gallery_path)
            if use_text or id_level == "game":
                assert args.text_path != "" if args.id_level == "game" else True, "Provide path to text features"
                all_text_features = load_text_features(args.text_path)
                all_gallery_features = group_gallery_like_text(all_gallery_features, all_text_features)
            
        os.makedirs(cache_dir, exist_ok=True)
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        if evaluate_identification:
            results_identification = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        
        detections = dict()

        video_id = 0
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[-3]
                img_name = img_file_name[0].split("/")[-1]
                
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()


                if frame_id == 1:
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        if evaluate_identification:
                            write_results_and_category_no_score(result_filename, results)
                        else:
                            write_results_no_score(result_filename, results)
                        results = []
                            
                    tracker = OCSortReID(det_thresh = self.args.track_thresh, iou_threshold=self.args.iou_thresh,
                                asso_func=self.args.asso, delta_t=self.args.deltat, inertia=self.args.inertia, use_byte=self.args.use_byte)
                    if args.feature_extractor == "ap3d":
                        encoder = AP3DReID(ckpt_path="models/best_model_ap3d.pth.tar")
                    else:
                        encoder = CLIPReID(model_name="openai/clip-vit-large-patch14")
                    
                    if args.use_clip_projection:
                        assert args.clip_projection_weights != "", "You need to provide weights for the projection"
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        extra_clip_projection = CLIPExtraProjectionLayer().to(device)
                        extra_clip_projection.load_state_dict(torch.load(args.clip_projection_weights, weights_only=True))
                        extra_clip_projection.eval()
                    
                    ckt_file =  f"mupnit_detections_{args.yolo_model}/{video_name}_detection.pkl"
                    all_dets = []
                    # Not even faster...
                    if os.path.exists(ckt_file):
                        # outputs = [torch.load(ckt_file)]
                        if not video_name in detections:
                            dets = torch.load(ckt_file)
                            detections[video_name] = dets 
                    
                        all_dets = detections[video_name]
                    if video_id > 0:
                        prev_video_name = video_names[video_id - 1]
                        if prev_video_name in detections and not os.path.exists(f"mupnit_detections_{args.yolo_model}/{prev_video_name}_detection.pkl"):
                            os.makedirs(f"mupnit_detections_{args.yolo_model}", exist_ok=True)
                            torch.save(detections[prev_video_name], f"mupnit_detections_{args.yolo_model}/{prev_video_name}_detection.pkl")
                    player_ids = None
                    gallery_feature = None
                    if evaluate_identification:
                        if id_level in ["game"]:
                            if use_text and use_gallery:
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
                                player_ids = []
                                embeddings = []
                                # Assume video_name is the game id key in all_text_features (a dict: {gameid: {playerid: text_emb}})
                                game_name = video_name.split("-")[0] # some videos are subsequences of frames of a different video
                                if game_name in all_gallery_features:
                                    for pid, emb in all_gallery_features[game_name].items():
                                        combined = all_gallery_features[game_name][pid]
                                        embeddings.append(combined)
                                        player_ids.append(pid)
                                gallery_feature = torch.stack(embeddings).squeeze() if embeddings else None
                                if args.use_clip_projection:
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

                        else:
                            embeddings = []
                            player_ids = []
                            for pid, emb in all_gallery_features.items():
                                embeddings.append(torch.tensor(emb))
                                player_ids.append(pid)
                            gallery_feature = torch.stack(embeddings).squeeze() if embeddings else None
                            if args.use_clip_projection:
                                    gallery_feature = extra_clip_projection(gallery_feature.to(device)).cpu()
                        # game_gallery_feature = gallery_feature
                        # game_player_ids = player_ids.copy()
                if video_name not in video_names.values():
                    video_names[video_id] = video_name
                    video_id += 1
                
                # if evaluate_identification and id_level == "court":
                #     court_players = get_players_for_frame(per_frame_ids, video_name, frame_id)
                #     if set(court_players) != set(player_ids):
                        
                #         court_gallery_feature = []
                #         for i,x in enumerate(game_player_ids):
                #             if x in court_players:
                #                 court_gallery_feature.append(game_gallery_feature[i])
                #         player_ids = [x for x in game_player_ids if x in court_players]
                #         gallery_feature = torch.stack(court_gallery_feature)

                detection_file = os.path.join(cache_dir, f"{video_name}_frame_{frame_id}_detection.pkl")
                embedding_file = os.path.join(cache_dir, f"{video_name}_frame_{frame_id}_embedding.pkl")
                
                if os.path.exists(detection_file) and os.path.exists(embedding_file):
                    with open(detection_file, "rb") as f:
                        outputs = pickle.load(f)
                    with open(embedding_file, "rb") as f:
                        query_feature = pickle.load(f)
                else:
                    if len(all_dets) > 0:
                        outputs = all_dets[frame_id]
                    else:
                        imgs = imgs.type(tensor_type)
                        outputs = model(imgs)
                        if decoder is not None:
                            outputs = decoder(outputs, dtype=outputs.type())
                        outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
                    
                    if outputs[0] is not None:
                        # raw_img = imgs.cpu().float()
                        raw_img = os.path.join(DATASET_ROOT,img_file_name[0])
                        raw_img = Image.open(raw_img).convert('RGB')
                        if args.dataset == "sportsmot":
                            raw_img = raw_img.resize(( 1440,800))
                        raw_img = torch.tensor(np.array(raw_img)).permute(-1,0,1).unsqueeze(0).float()
                        bbox_xyxy = copy.deepcopy(outputs[0][:, :4])
                        query_feature = encoder.inference(raw_img, bbox_xyxy.cpu().detach().numpy())
                        if len(query_feature.shape) != 2:
                            query_feature = query_feature.unsqueeze(0)
                    else:
                        query_feature = None
                    with open(detection_file, "wb") as f:
                        pickle.dump(outputs, f)
                    with open(embedding_file, "wb") as f:
                        pickle.dump(query_feature, f)
                if args.use_clip_projection:
                    if query_feature != None:
                        f = torch.tensor(query_feature).squeeze().to(device)
                        if len(f.shape) != 2:
                            f = f.unsqueeze(0)
                        query_feature = extra_clip_projection(f).cpu()
                        
                
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size, query_feature, gallery_feature, player_ids, reid_thresh, args.average_query_embeds, args)
            online_tlwhs = []
            online_ids = []
            if evaluate_identification:
                identified_players = []
            for t in online_targets:
                """
                    Here is minor issue that DanceTrack uses the same annotation
                    format as MOT17/MOT20, namely xywh to annotate the object bounding
                    boxes. But DanceTrack annotation is cropped at the image boundary, 
                    which is different from MOT17/MOT20. So, cropping the output
                    bounding boxes at the boundary may slightly fix this issue. But the 
                    influence is minor. For example, with my results on the interpolated
                    OC-SORT:
                    * without cropping: HOTA=55.731
                    * with cropping: HOTA=55.737
                """
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                if tlwh[2] * tlwh[3] > self.args.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    if evaluate_identification:
                        identified_players.append(t[-1])
            # save results
            
            if evaluate_identification:
                results.append((frame_id, online_tlwhs, online_ids,identified_players))
            else:
                results.append((frame_id, online_tlwhs, online_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                if evaluate_identification:
                    write_results_and_category_no_score(result_filename, results)
                else:
                    write_results_no_score(result_filename, results)
                

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results


    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list



    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        track_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_track_time = 1000 * track_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "track", "inference"],
                    [a_infer_time, a_track_time, (a_infer_time + a_track_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
            from yolox.layers import COCOeval_opt as COCOeval
            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info