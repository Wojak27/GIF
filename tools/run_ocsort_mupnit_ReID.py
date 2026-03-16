from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger
from yolox.evaluators import MOTEvaluatorMuPNIT_ReID as MOTEvaluator
import json

from utils.args import make_parser
import os
import random
import warnings
import glob
import motmetrics as mm
from collections import OrderedDict
from pathlib import Path

def get_gt_files(split, root_dir):
    """
    Get ground truth file paths for all sequences in a split.

    Args:
        split (str): Split name (e.g., 'train', 'val', 'test').
        root_dir (str): Root directory of the MuPNIT dataset.

    Returns:
        list: List of paths to ground truth files.
    """
    gt_files = []
    split_dir = os.path.join(root_dir, split)
    for seq_name in sorted(os.listdir(split_dir)):
        gt_path = os.path.join(split_dir, seq_name, 'gt', 'gt.txt')
        if os.path.isfile(gt_path):
            gt_files.append(gt_path)
    return gt_files


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:            
            logger.info('Comparing {}...'.format(k))
            # if "-71" not in k:
            #     continue
            # if "-547" not in k:
            #     continue
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names

def get_per_frame_ids(gt):
    results = {}
    for game_key, df in gt.items():
        frames = sorted(df.index.get_level_values('FrameId').unique())
        segments = []
        start_frame = frames[0]
        seg_ids = set(df.xs(start_frame, level='FrameId').index.get_level_values('Id'))
        seg_cat = "normal" if len(seg_ids) <= 10 else "overflow"
        prev_frame = start_frame

        for f in frames[1:]:
            frame_ids = set(df.xs(f, level='FrameId').index.get_level_values('Id'))
            frame_cat = "normal" if len(frame_ids) <= 10 else "overflow"
            # For normal segments, try union accumulation if it stays normal.
            if seg_cat == "normal":
                candidate = seg_ids.union(frame_ids)
                if len(candidate) <= 10 and frame_cat == "normal":
                    seg_ids = candidate
                else:
                    segments.append((start_frame, prev_frame, sorted(seg_ids)))
                    start_frame = f
                    seg_ids = frame_ids
                    seg_cat = "overflow" if len(frame_ids) > 10 else "normal"
            else:
                # For overflow segments, continue accumulating if still overflow.
                if frame_cat == "overflow":
                    seg_ids = seg_ids.union(frame_ids)
                else:
                    segments.append((start_frame, prev_frame, sorted(seg_ids)))
                    start_frame = f
                    seg_ids = frame_ids
                    seg_cat = "normal"
            prev_frame = f
        segments.append((start_frame, frames[-1], sorted(seg_ids)))
        
        # Merge any single-frame segments with neighbors
        merged = []
        i = 0
        while i < len(segments):
            start, end, ids_list = segments[i]
            if start == end:
                # Try merging with previous segment if exists.
                if merged:
                    prev_start, prev_end, prev_ids = merged.pop()
                    new_ids = set(prev_ids).union(ids_list)
                    merged.append((prev_start, end, sorted(new_ids)))
                # Else, try merging with next segment.
                elif i < len(segments) - 1:
                    next_start, next_end, next_ids = segments[i+1]
                    new_ids = set(ids_list).union(next_ids)
                    merged.append((start, next_end, sorted(new_ids)))
                    i += 1
                else:
                    merged.append((start, end, ids_list))
            else:
                merged.append((start, end, ids_list))
            i += 1

        results[game_key] = merged
    return results


@logger.catch
def main(exp, args, num_gpu):
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    cudnn.benchmark = True
    assert args.yolo_model in ["yolox_s","yolox_m","yolox_l","yolox_x"], "Invalid model name"
    assert args.yolo_model in args.ckpt, "Possibly invalid model path, model name not in path"
    rank = args.local_rank
    file_name = os.path.join(exp.output_dir, args.expn)
    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    result_dir = "{}_test".format(args.expn) if args.test else "{}_val".format(args.expn)
    results_folder = os.path.join(file_name, result_dir)
    os.makedirs(results_folder, exist_ok=True)
    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    val_loader = exp.get_eval_loader(args.batch_size, is_distributed, args.test)
    evaluator = MOTEvaluator(
        args=args,
        dataloader=val_loader,
        img_size=exp.test_size,
        confthre=exp.test_conf,
        nmsthre=exp.nmsthre,
        num_classes=exp.num_classes,
        )

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    if not args.speed and not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_file, map_location=loc)
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert (
            not args.fuse and not is_distributed and args.batch_size == 1
        ), "TensorRT model is not support model fusing and distributed inferencing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None
        
    root_dir = os.environ.get('MUPNIT_DATASET_ROOT', '/4TBSSD_Permanent/datasets/MuPNIT_30fps_global')
    split_type = 'test' if args.test else 'val'

    gtfiles = get_gt_files(split_type, root_dir)
    gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])

        
    per_frame_ids = None
    if args.id_level == "court":
        per_frame_ids = get_per_frame_ids(gt)

    # start tracking
    if args.bytetrack:
        *_, summary = evaluator.evaluate_bytetrack(
            model, is_distributed, args.fp16, trt_file, decoder, exp.test_size, results_folder
        )
    else:
        *_, summary = evaluator.evaluate_ocsort(
                model, is_distributed, args.fp16, trt_file, decoder, exp.test_size, results_folder, per_frame_ids=per_frame_ids
        )
    logger.info("\n" + summary)
    
    if args.test:
        # we skip evaluation for inference on test set
        return 

    # if we evaluate on validation set, 

    # evaluate on the validation set
    mm.lap.default_solver = 'lap'




    print('gt_files', gtfiles)
    tsfiles = [f for f in glob.glob(os.path.join(results_folder, '*.txt')) if not os.path.basename(f).startswith('eval')]

    logger.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
    logger.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
    logger.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
    logger.info('Loading files.')
    
    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1)) for f in tsfiles])    
    
    mh = mm.metrics.create()    
    accs, names = compare_dataframes(gt, ts)
    
    logger.info('Running metrics')
    metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked',
               'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
               'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    div_dict = {
        'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
        'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']}
    for divisor in div_dict:
        for divided in div_dict[divisor]:
            summary[divided] = (summary[divided] / summary[divisor])
    fmt = mh.formatters
    change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mostly_tracked',
                       'partially_tracked', 'mostly_lost']
    for k in change_fmt_list:
        fmt[k] = fmt['mota']
    print(mm.io.render_summary(summary, formatters=fmt, namemap=mm.io.motchallenge_metric_names))

    metrics = mm.metrics.motchallenge_metrics + ['num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    logger.info('Completed')


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.expn:
        args.expn = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args, num_gpu),
    )