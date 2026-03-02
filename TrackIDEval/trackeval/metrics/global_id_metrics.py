import numpy as np
from ._base_metric import _BaseMetric
from .. import _timing

class GlobalIDMetrics(_BaseMetric):
    """
    Extra global identity metrics:
      - FTFI: frames-to-first-identification per track (number of frames from track start until the track’s predicted class equals its majority vote)
      - FTFIc: FTFI_correct frames-to-first-correct-identification (number of frames from track start until the predicted class equals the ground-truth majority, ignoring invalid assignments)
      - AvgClsSw: average number of times a track’s predicted class changes
      - MCorrRat: mean correct ratio (average fraction of frames where the predicted class equals the track’s predicted majority)
      - mAP: overall ratio of correct predicted frames across all tracklets
      - NDTPC: NumDifferentTrackletsPerClass summary of how many tracklets ended up with each ground-truth majority class (via majority vote on GT values)
    """
    def __init__(self, config=None):
        super().__init__()
        self.fields = ['FTFI', 'FTFIc', 'AvgClsSw', 'MCorrRat', 'mAP', 'NDTPC']
        self.summary_fields = self.fields

    @_timing.time
    def eval_sequence(self, data):
        # Build per-tracklet data from per-frame predictions.
        # Expected keys in data:
        #   'tracker_ids': list (over frames) of np.array of predicted track IDs
        #   'tracker_classes': list (over frames) of np.array of predicted classes
        #   'gt_classes': list (over frames) of np.array of ground-truth classes
        #   'similarity_scores': list (over frames) of 2D arrays linking GT and tracker detections.
        track_data = {}
        num_frames = data['num_timesteps']
        for t in range(num_frames):
            t_ids = data['tracker_ids'][t]
            t_preds = data['tracker_classes'][t]
            # Use gt_classes from data; assume they are valid.
            gt_cls = data['gt_classes'][t]
            sim = data['similarity_scores'][t] if 'similarity_scores' in data else None
            for i, tid in enumerate(t_ids):
                if tid not in track_data:
                    track_data[tid] = {'frames': [], 'predicted': [], 'gt': []}
                track_data[tid]['frames'].append(t)
                track_data[tid]['predicted'].append(t_preds[i])
                # Association: if similarity is provided and its max exceeds threshold, use that associated GT.
                # Otherwise, simply take the i-th GT value (if available).
                if sim is not None and sim.shape[0] > 0 and sim.shape[1] > i and np.max(sim[:, i]) > 0.5:
                    assigned_gt = gt_cls[np.argmax(sim[:, i])]
                else:
                    assigned_gt = gt_cls[i] if i < len(gt_cls) else -1
                track_data[tid]['gt'].append(assigned_gt)
        
        ftfi_list = []          # FTFI: based on majority vote of predicted classes.
        ftfi_correct_list = []  # FTFIc: based on ground-truth majority (ignoring invalids).
        switch_list = []
        correct_ratios = []
        gt_valid_frames_total = 0  # For mAP computed relative to GT.
        gt_correct_frames_total = 0
        class_to_tracks = {}
        for tid, d in track_data.items():
            if len(d['predicted']) == 0:
                continue
            frames = d['frames']
            preds = d['predicted']
            # Compute majority of predicted classes (for FTFI and MCorrRat).
            majority_pred = np.bincount(np.array(preds, dtype=int)).argmax()
            first_match = next((f for f, p in zip(frames, preds) if p == majority_pred), None)
            ftfi = first_match - frames[0] if first_match is not None else np.nan
            ftfi_list.append(ftfi)
            
            # Compute FTFIc: use majority vote on GT values (only valid GT, i.e. >= 0).
            gt_arr = np.array(d['gt'], dtype=int)
            valid = gt_arr >= 0
            if np.any(valid):
                gt_majority = np.bincount(gt_arr[valid]).argmax()
                first_correct = next((f for f, p in zip(frames, preds) if p == gt_majority), None)
                ftfi_correct = first_correct - frames[0] if first_correct is not None else np.nan
            else:
                ftfi_correct = np.nan
            ftfi_correct_list.append(ftfi_correct)
            
            # Count class switches.
            switches = np.sum(np.diff(preds) != 0)
            switch_list.append(switches)
            
            # Correct ratio: fraction of frames where predicted equals the predicted majority.
            cr = np.mean(np.array(preds) == majority_pred)
            correct_ratios.append(cr)
            
            # For mAP relative to GT, count valid frames and frames where prediction equals GT majority.
            if np.any(valid):
                gt_majority = np.bincount(gt_arr[valid]).argmax()
                # Only consider frames with valid GT.
                valid_preds = np.array(d['predicted'])[valid]
                gt_correct = np.sum(valid_preds == gt_majority)
                gt_valid_frames = np.sum(valid)
            else:
                gt_correct = 0
                gt_valid_frames = 0
            gt_correct_frames_total += gt_correct
            gt_valid_frames_total += gt_valid_frames
            
            # Tally tracklets per ground-truth majority.
            if np.any(valid):
                gt_maj = np.bincount(gt_arr[valid]).argmax()
            else:
                gt_maj = -1
            class_to_tracks[gt_maj] = class_to_tracks.get(gt_maj, 0) + 1

        FTFI = np.nanmean(ftfi_list) if ftfi_list else np.nan
        FTFIc = np.nanmean(ftfi_correct_list) if ftfi_correct_list else np.nan
        AvgClsSw = np.nanmean(switch_list) if switch_list else np.nan
        MCorrRat = np.nanmean(correct_ratios) if correct_ratios else np.nan
        mAP = gt_correct_frames_total / gt_valid_frames_total if gt_valid_frames_total > 0 else np.nan
        NDTPC = "; ".join([f"{cls}: {cnt}" for cls, cnt in class_to_tracks.items()])
        return {
            'FTFI': FTFI,
            'FTFIc': FTFIc,
            'AvgClsSw': AvgClsSw,
            'MCorrRat': MCorrRat,
            'mAP': mAP,
            'NDTPC': NDTPC
        }

    def combine_sequences(self, all_res):
        combined = {}
        for field in ['FTFI', 'FTFIc', 'AvgClsSw', 'MCorrRat', 'mAP']:
            vals = [res[field] for res in all_res.values() if not np.isnan(res[field])]
            combined[field] = np.nanmean(vals) if vals else np.nan
        class_strs = [res['NDTPC'] for res in all_res.values() if res['NDTPC']]
        combined['NDTPC'] = "; ".join(class_strs)
        return combined

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=False):
        return self.combine_sequences(all_res)

    def combine_classes_det_averaged(self, all_res):
        return self.combine_sequences(all_res)

    def _summary_row(self, results_):
        # Format each numeric value with three digits after the decimal.
        row = []
        for h in self.summary_fields:
            val = results_[h]
            if isinstance(val, str):
                row.append(val)
            elif isinstance(val, (float, int, np.number)):
                row.append("{0:.3f}".format(float(val)))
            elif isinstance(val, (list, np.ndarray)):
                row.append("{0:.3f}".format(np.mean(val)))
            else:
                row.append(str(val))
        return row