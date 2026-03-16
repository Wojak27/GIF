import argparse
import copy
import json
import os
import os.path as osp
import time
import cv2
import torch

from loguru import logger
import numpy as np

from clip_reid.reid_eval import CLIPReID
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import get_color, plot_tracking
from trackers.ocsort_tracker.ocsort_reid import OCSORT_ReIDv1 as OCSORT_ReIDv1
from trackers.tracking_utils.timer import Timer


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
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
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


def image_demo(predictor, vis_folder, current_time, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    tracker = OCSORT_ReIDv1(args, det_thresh = args.track_thresh, iou_threshold=args.iou_thresh, asso_func=args.asso, delta_t=args.deltat, inertia=args.inertia)
    timer = Timer()
    results = []

    for frame_id, img_path in enumerate(files, 1):
        outputs, img_info = predictor.inference(img_path, timer)
        if outputs[0] is not None:
            online_targets, track_embeddings = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size, get_embeddings=True)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1.0,-1,-1,-1\n"
                    )
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if args.save_result:
            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_folder = osp.join(vis_folder, timestamp)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")
        
        
# First, add these imports at the top of the file
import clip
import torch.nn.functional as F

attributes_dict = {
    "jersey_numbers": [20,12,23,7,9,33,30,51,13,3],
    "jersey_colors": ["blue", "white"],
    "ethnicities": ["white", "black", "hispanic"],
}

placeholder_ = "<placeholder>"

texts = {
    # "jersey_numbers": f"jersey number {placeholder_}, text number {placeholder_}",
    # "jersey_numbers": f"Picture of a jersey number {placeholder_} with text number {placeholder_}",
    "jersey_numbers": f"a clear view of jersey number {placeholder_} on a basketball uniform",
    # "jersey_numbers": f"a picture of number {placeholder_}",
    # "jersey_numbers": f"a number {placeholder_}",
    "jersey_colors": f"a {placeholder_} jersey, {placeholder_} color",
    "ethnicities": f"a {placeholder_} basketball player",
}

# Create a helper class to manage CLIP embeddings and similarity calculations
class AttributeClassifier:
    def __init__(self, device, attribute_dict=None):
        if attribute_dict is None:
            attribute_dict = attributes_dict
        self.device = device
        self.model, self.preprocess = clip.load("ViT-L/14", device=device)
        
        # Generate text embeddings for all attributes
        self.attribute_embeddings = {}
        for attr_name, values in attribute_dict.items():
            attr_texts = [texts[attr_name].replace(placeholder_, str(val)) for val in values]
            text_tokens = clip.tokenize(attr_texts).to(device)
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            self.attribute_embeddings[attr_name] = {
                'values': values,
                'embeddings': text_features
            }
    
    def get_attributes(self, image_embedding, mem_dict=None):
        """Calculate attributes for a given image embedding using cosine similarity"""
        results = {}
        
        # Convert numpy array to torch tensor if needed
        if isinstance(image_embedding, np.ndarray):
            image_embedding = torch.from_numpy(image_embedding).to(self.device).to(torch.float16)
        
        # Ensure it's a 2D tensor
        if image_embedding.dim() == 1:
            image_embedding = image_embedding.unsqueeze(0)
            
        # Normalize the embedding
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        
        for attr_name, attr_data in self.attribute_embeddings.items():
            similarities = (100.0 * image_embedding @ attr_data['embeddings'].T).softmax(dim=-1)
            if mem_dict is not None:
                # Use memory to track jersey numbers
                if attr_name not in mem_dict:
                    mem_dict[attr_name] = {}
                weights= [0.6,0.4]
                for idx, value in enumerate(attr_data['values']):
                    if mem_dict[attr_name].get(value) is None:
                        mem_dict[attr_name][value] = 0.0
                    # mem_dict[attr_name][value] = (similarities[0][idx].item()*weights[0] + mem_dict[attr_name][value]*weights[1])/2 if similarities[0][idx].item() > 0.8 else mem_dict[attr_name][value]
                    mem_dict[attr_name][value] = 1 + mem_dict[attr_name][value] if similarities[0][idx].item() > 0.7 else mem_dict[attr_name][value]
                    similarities[0][idx] = mem_dict[attr_name][value]
                
            max_idx = similarities.argmax().item()
            confidence = similarities[0][max_idx].item()
            if confidence == 0:
                value = "unknown"
            else:
                value = attr_data['values'][max_idx]
            results[attr_name] = (value, confidence)
        
        if mem_dict is not None:
            # Update memory with the latest jersey number
            if 'jersey_numbers' in results:
                mem_dict['jersey_numbers'][results['jersey_numbers'][0]] = results['jersey_numbers'][1]
            return results, mem_dict
        return results

# Modified plot_tracking function to include attribute display
def plot_tracking_with_attributes(image, tlwhs, obj_ids, attributes, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]
    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255
    
    # Fixed values as per reference
    text_scale = 2
    text_thickness = 2
    line_thickness = 3
    radius = max(5, int(im_w/140.))
    
    # Draw frame info
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
    
    # Draw boxes and attributes
    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        
        color = get_color(abs(obj_id))
        
        # Draw bounding box
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        
        # Draw ID
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
        
        # Draw attributes if available
        if obj_id in attributes:
            attr = attributes[obj_id]
            attr_text = f"#{attr['jersey_numbers'][0]},{int(attr['jersey_numbers'][1])}x c:{attr['jersey_colors'][0]} e:{attr['ethnicities'][0]}"
            cv2.putText(im, attr_text, 
                       (intbox[0], intbox[1] + int(30 * text_scale)), 
                       cv2.FONT_HERSHEY_PLAIN, text_scale, color,
                       thickness=text_thickness)
    
    return im
        


def imageflow_demo(predictor, vis_folder, current_time, args, attribute_dict=None, player_mappings=None):
    cap = cv2.VideoCapture(args.path if args.demo_type == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    #CLAUDE: import clip for text embedding generation
    os.makedirs(save_folder, exist_ok=True)
    if args.demo_type == "video":
        save_path = args.out_path
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    attribute_classifier = AttributeClassifier(args.device, attribute_dict)
    track_attributes = {}  # Store attributes for each track ID
    
    tracker = OCSORT_ReIDv1(args, det_thresh = args.track_thresh, iou_threshold=args.iou_thresh, asso_func=args.asso, delta_t=args.deltat, inertia=args.inertia)
    encoder = CLIPReID(model_name="openai/clip-vit-large-patch14")
    timer = Timer()
    frame_id = 0
    results = []
    num_mem_dict= {}
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            if outputs[0] is not None:
                raw_img = frame
                # raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                raw_img=torch.tensor(raw_img).permute(-1,0,1).unsqueeze(0)
                bbox_xyxy = copy.deepcopy(outputs[0][:, :4])
                id_feature = encoder.inference(raw_img, bbox_xyxy.cpu().detach().numpy())
                online_targets, embeddings = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size, id_feature=id_feature, get_embeddings=True)
                online_tlwhs = []
                online_ids = []
                
                for t in online_targets:
                    tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                    tid = t[4]
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1.0,-1,-1,-1\n"
                        )
                if embeddings is not None:
                    for i, tid in enumerate(online_ids):
                        if embeddings[tid] is not None:
                            if num_mem_dict is not None:
                                if num_mem_dict.get(tid) is None:
                                    num_mem_dict[tid] = {}
                                attrs, mem_dict = attribute_classifier.get_attributes(embeddings[tid], mem_dict=num_mem_dict[tid])
                                num_mem_dict[tid] = mem_dict
                            else:
                                attrs = attribute_classifier.get_attributes(embeddings[tid])
                            track_attributes[tid] = attrs
                timer.toc()
                # Modify the plot_tracking call to include attributes:
                online_im = plot_tracking_with_attributes(
                    img_info['raw_img'], online_tlwhs, online_ids, track_attributes,
                    frame_id=frame_id + 1, fps=1. / timer.average_time
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

# First, let's extract the unique values and create the attribute dictionary
def create_attribute_dict(data):
    jersey_numbers = []
    ethnicities = set()
    jersey_colors = [data["colorA"], data["colorB"]]
    
    for player_info in data["players"].values():
        if "jersey_number" in player_info:
            jersey_numbers.append(player_info["jersey_number"])
        if "ethnicity" in player_info:
            ethnicities.add(player_info["ethnicity"])
    
    attribute_dict = {
        "jersey_numbers": sorted(list(set(jersey_numbers))),
        "jersey_colors": jersey_colors,
        "ethnicities": sorted(list(ethnicities))
    }
    
    return attribute_dict

# Create reverse mappings from attributes to player names
def create_player_mappings(data):
    player_mappings = {
        "by_jersey_number": {},
        "by_team_color": {
            data["colorA"]: [],
            data["colorB"]: []
        },
        "by_ethnicity": {}
    }
    
    # Map jersey numbers to players
    for player_name, info in data["players"].items():
        if "jersey_number" in info:
            if info["jersey_number"] not in player_mappings["by_jersey_number"]:
                player_mappings["by_jersey_number"][info["jersey_number"]] = []
            player_mappings["by_jersey_number"][info["jersey_number"]].append(player_name)
        
        # Map team colors to players
        if info["team_name"] == data["teamA"]:
            player_mappings["by_team_color"][data["colorA"]].append(player_name)
        else:
            player_mappings["by_team_color"][data["colorB"]].append(player_name)
        
        # Map ethnicities to players
        if "ethnicity" in info:
            if info["ethnicity"] not in player_mappings["by_ethnicity"]:
                player_mappings["by_ethnicity"][info["ethnicity"]] = []
            player_mappings["by_ethnicity"][info["ethnicity"]].append(player_name)
    
    return player_mappings

def main(exp, args):
    if not args.expn:
        args.expn = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.expn)
    os.makedirs(output_dir, exist_ok=True)
    
    if args.rosters_file is not None:
        rosters_file = json.load(open(args.rosters_file))
        gameID = args.path.split("/")[-1].split(".")[0]
        attribute_dict = create_attribute_dict(rosters_file[gameID])
        player_mappings = create_player_mappings(rosters_file[gameID])

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
    if args.demo_type == "image":
        image_demo(predictor, vis_folder, current_time, args)
    elif args.demo_type == "video" or args.demo_type == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args, attribute_dict, player_mappings)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
