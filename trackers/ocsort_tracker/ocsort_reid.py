"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
from __future__ import print_function

import numpy as np
import copy
from .association_reid import *
from collections import deque       # [hgx0418] deque for reid feature
np.random.seed(0)
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler

def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age-dt]
    max_age = max(observations.keys())
    return observations[max_age]


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h  # scale is just area
    r = w / float(h+1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score == None):
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0]+bbox1[2]) / 2.0, (bbox1[1]+bbox1[3])/2.0
    cx2, cy2 = (bbox2[0]+bbox2[2]) / 2.0, (bbox2[1]+bbox2[3])/2.0
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm

def speed_direction_lt(bbox1, bbox2):
    cx1, cy1 = bbox1[0], bbox1[1]
    cx2, cy2 = bbox2[0], bbox2[1]
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm

def speed_direction_rt(bbox1, bbox2):
    cx1, cy1 = bbox1[0], bbox1[3]
    cx2, cy2 = bbox2[0], bbox2[3]
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm

def speed_direction_lb(bbox1, bbox2):
    cx1, cy1 = bbox1[2], bbox1[1]
    cx2, cy2 = bbox2[2], bbox2[1]
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm

def speed_direction_rb(bbox1, bbox2):
    cx1, cy1 = bbox1[2], bbox1[3]
    cx2, cy2 = bbox2[2], bbox2[3]
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def combine_features_svd(features):
    """
    Combines multiple features into a single representative vector using SVD.
    """
    # Center the data
    features_centered = features - np.mean(features, axis=0)
    
    # Compute SVD directly
    _, _, vh = np.linalg.svd(features_centered, full_matrices=False)
    
    # Take the first right singular vector as our combined feature
    combined_feature = vh[0]
    
    return combined_feature

def combine_features(features, use_svd=True):
    if use_svd:
        return combine_features_svd(features)
    else:
        return np.mean(features, axis=0)

class KalmanBoxTrackerv2(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, temp_feat, delta_t=3, orig=False, buffer_size=20, args=None, n_clusters=5):     # 'temp_feat' and 'buffer_size' for reid feature
        """
        Initialises a tracker using initial bounding box.

        """
        # define constant velocity model
        if not orig:
          from .kalmanfilter import KalmanFilterNew as KalmanFilter
          self.kf = KalmanFilter(dim_x=7, dim_z=4)
        else:
          from filterpy.kalman import KalmanFilter
          self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [
                            0, 0, 0, 1, 0, 0, 0],  [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTrackerv2.count
        KalmanBoxTrackerv2.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of 
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a 
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]), let's bear it for now.
        """
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.observations = dict()
        self.history_observations = []
        self.velocity = None
        self.delta_t = delta_t
        self.features = []
        self.use_clustering = args.use_clustering
        if self.use_clustering:
            assert n_clusters > 0 and buffer_size > 0 and buffer_size >= n_clusters
            self.n_clusters = n_clusters
            self.cluster_counts = None
            self.tmp_features = []
        self.buffer_size = buffer_size
        self.update_features(temp_feat)


    # ReID. for update embeddings during tracking
    def _compute_cosine_similarity(self, feat1, feat2):
        """Compute cosine similarity between two feature vectors."""
        return 1 - cosine(feat1, feat2)

    def _find_closest_cluster(self, feature):
        """Find the index of the closest cluster centroid using cosine similarity."""
        similarities = [self._compute_cosine_similarity(feature, centroid) 
                       for centroid in self.features]
        return np.argmax(similarities)

    def _update_cluster_centroid(self, cluster_idx, new_feature):
        """Update the centroid of a cluster with a new feature."""
        current_count = self.cluster_counts[cluster_idx]
        current_centroid = self.features[cluster_idx]
        
        # Update weighted mean
        updated_centroid = (current_centroid * current_count + new_feature) / (current_count + 1)
        self.features[cluster_idx] = updated_centroid
        self.cluster_counts[cluster_idx] += 1

    def update_features(self, feat):
        """
        Update feature buffer using clustering when buffer is full.
        For subsequent features, associate with closest cluster and update centroid.
        """
        if self.use_clustering:
            self.tmp_features.append(feat)
            
            if len(self.tmp_features) >= self.buffer_size:
                # Buffer is full, perform initial clustering
                features_array = np.concatenate((self.tmp_features, self.features), axis=0)
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(features_array)
                
                # Initialize cluster centroids and counts
                self.features = kmeans.cluster_centers_
                self.cluster_counts = np.zeros(self.n_clusters)
                self.tmp_features = []
                
                # Count samples in each cluster
                for label in cluster_labels:
                    self.cluster_counts[label] += 1
            if len(self.features) < self.n_clusters:
                self.features.append(feat)
                
        else:
            if len(self.features) < self.buffer_size:
                self.features.append(feat)
            else:
                self.features.append(feat)
                # average the features
                self.features = [np.mean(self.features, axis=0)]

    def get_feature_representation(self):
        return self.features
        
    def update(self, bbox, id_feature, update_feature=True):
        """
        Updates the state vector with observed bbox.
        """
        if bbox is not None:
            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age-dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                """
                  Estimate the track speed direction with observations \Delta t steps away
                """
                self.velocity = speed_direction(previous_box, bbox)
            
            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(convert_bbox_to_z(bbox))
            # add interface for update feature or not
            if update_feature:
                    self.update_features(id_feature)
        else:
            self.kf.update(bbox)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


"""
    We support multiple ways for association cost calculation, by default
    we use IoU. GIoU may have better performance in some situations. We note 
    that we hardly normalize the cost by all methods to (0,1) which may not be 
    the best practice.
"""
ASSO_FUNCS = {  "iou": iou_batch,
                "giou": giou_batch,
                "ciou": ciou_batch,
                "diou": diou_batch,
                "ct_dist": ct_dist,
                "Height_Modulated_IoU": hmiou
                }


class OCSORT_ReIDv2(object):
    def __init__(self, args, det_thresh, max_age=30, min_hits=3,
        iou_threshold=0.3, delta_t=3, asso_func="iou", inertia=0.2):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.embed_threshold = 0.95
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.use_byte = args.use_byte
        self.args = args
        KalmanBoxTrackerv2.count = 0

    def update(self, output_results, img_info, img_size, id_feature=None, warp_matrix=None):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        if output_results is None:
            return np.empty((0, 5))

        self.frame_count += 1
        # post_process detections
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale
        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)
        detect_distance = iou_batch(dets, dets) - np.eye(dets.shape[0])
        detection_overlap = np.mean(detect_distance, axis=0)
        scores = scores * (1 - detection_overlap)
        
        inds_low = scores > 0.1
        inds_high = scores < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)  # self.det_thresh > score > 0.1, for second matching
        dets_second = dets[inds_second]  # detections for second matching
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]
        id_feature = np.asarray([track.detach().cpu().numpy() for track in id_feature],dtype=np.float64)
        id_feature_keep = id_feature[remain_inds]  # ID feature of 1st stage matching
        id_feature_second = id_feature[inds_second]  # ID feature of 2nd stage matching

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array(
            [trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array(
            [k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        """
            First round of association
        """
        association_weights = [1.0, 0.7]
        max_len = max([len(track.features) for track in self.trackers]) if self.trackers else 0
        track_features = np.asarray([combine_features(np.vstack(list(track.features))) for track in self.trackers],
                                                 dtype=np.float64)

        
        detect_distance = iou_batch(dets, dets) - np.eye(dets.shape[0])
        detection_overlap = np.mean(detect_distance, axis=0)
        detection_overlapping = np.array(list(set(np.argwhere(detect_distance > 0.7).flatten())))
        
        emb_dists = embedding_distance(track_features, id_feature_keep).T
        
        # matched, unmatched_dets, unmatched_trks = associate(dets, trks, self.iou_threshold, velocities, k_observations, self.inertia)
        matched, unmatched_dets, unmatched_trks = associate_CLIP_reID(dets, trks, self.iou_threshold, velocities, k_observations, self.inertia, emb_dists, association_weights)
        # measure the overlap between the detections in the current frame

        # update with id feature
        for m in matched:
            if self.args.scale_embedding_with_confidence:
                tmp_feature = id_feature_keep[m[0], :]*(1-detection_overlap[m[0]])
                # tmp_feature = id_feature_keep[m[0], :]*dets[m[0], :][-1]*(1-detection_overlap[m[0]])
                update_feature = m[0] not in detection_overlapping
            else:
                tmp_feature = id_feature_keep[m[0], :]
                update_feature = m[0] not in detection_overlapping
            self.trackers[m[1]].update(dets[m[0], :], tmp_feature, update_feature=update_feature) # update feature only when the detection is not overlapping with others

        """
            Second round of associaton by OCR
        """
        association_weights = [1.0, 1.0]
        # BYTE association
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]
            iou_left = self.asso_func(dets_second, u_trks)          # iou between low score detections and unmatched tracks
            iou_left = np.array(iou_left)
            # measure the overlap between the detections in the current frame
            detect_distance = iou_batch(dets_second, dets_second) - np.eye(dets_second.shape[0])
            second_detection_distance_weight = (1-np.max(detect_distance, axis=0))
            detection_overlapping = np.array(list(set(np.argwhere(detect_distance > 0.7).flatten())))
            detection_overlap = np.mean(detect_distance, axis=0)
            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                emb_dists_second = embedding_distance(track_features[unmatched_trks], id_feature_second).T
                matched_indices = linear_assignment((association_weights[0] * (-(iou_left))) + (-association_weights[1] * (1-(emb_dists_second.T*second_detection_distance_weight).T)))
                to_remove_trk_indices = []
                for m in matched_indices:
                    det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    if self.args.scale_embedding_with_confidence:
                        tmp_feature = id_feature_second[det_ind, :]*(1-detection_overlap[det_ind])
                        # tmp_feature = id_feature_second[det_ind, :]*dets_second[det_ind, :][-1]*(1-detection_overlap[det_ind])
                        update_feature = det_ind not in detection_overlapping
                    else:
                        tmp_feature = id_feature_second[det_ind, :]
                        update_feature = det_ind not in detection_overlapping
                    self.trackers[trk_ind].update(dets_second[det_ind, :], tmp_feature, update_feature=update_feature)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_tracks = track_features[unmatched_trks]
            left_feature = id_feature_keep[unmatched_dets]
            left_embed = embedding_distance(left_tracks,left_feature).T
            # id_feature_left = id_feature[unmatched_dets]
            # measure the overlap between the detections in the current frame
            detect_distance = iou_batch(left_dets, left_dets) - np.eye(left_dets.shape[0])
            detection_overlapping = np.array(list(set(np.argwhere(detect_distance > 0.7).flatten())))
            detection_overlap = np.mean(detect_distance, axis=0)
            iou_left = self.asso_func(left_dets, left_trks)
            iou_left = np.array(iou_left)
            
            if (1-left_embed).max() >= self.embed_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                rematched_indices = linear_assignment((1-left_embed)*(-association_weights[1]))
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if (1-left_embed)[m[0],m[1]] < self.embed_threshold:
                        continue
                    if self.args.scale_embedding_with_confidence:
                        tmp_feature = id_feature_keep[det_ind, :]*(1-detection_overlap[det_ind])
                        # tmp_feature = id_feature_keep[det_ind, :]*dets[det_ind, :][-1]*(1-detection_overlap[det_ind])
                        update_feature = det_ind not in detection_overlapping
                    else:
                        tmp_feature = id_feature_keep[det_ind, :]
                        update_feature = det_ind not in detection_overlapping
                    self.trackers[trk_ind].update(dets[det_ind, :], tmp_feature, update_feature=update_feature)
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks:
            self.trackers[m].update(None, None)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTrackerv2(dets[i, :], id_feature_keep[i, :], delta_t=self.delta_t, args=self.args)
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0][:4]
            else:
                """
                    this is optional to use the recent observation or the kalman filter prediction,
                    we didn't notice significant difference here
                """
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            # remove dead tracklet (no observation for long time, try to recover the lost tracklet)
            # if(trk.time_since_update > self.max_age):
            #     self.trackers.pop(i)
        if(len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))

class KalmanBoxTrackerv1(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, temp_feat, delta_t=3, orig=False, buffer_size=10, args=None):     # 'temp_feat' and 'buffer_size' for reid feature
        """
        Initialises a tracker using initial bounding box.

        """
        # define constant velocity model
        if not orig:
          from .kalmanfilter import KalmanFilterNew as KalmanFilter
          self.kf = KalmanFilter(dim_x=7, dim_z=4)
        else:
          from filterpy.kalman import KalmanFilter
          self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [
                            0, 0, 0, 1, 0, 0, 0],  [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTrackerv1.count
        KalmanBoxTrackerv1.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of 
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a 
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]), let's bear it for now.
        """
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.observations = dict()
        self.history_observations = []
        self.velocity = None
        self.delta_t = delta_t
        self.features = []
        self.buffer_size = buffer_size
        self.update_features(temp_feat)


    # ReID. for update embeddings during tracking
    def update_features(self, feat):
        tmp  = self.features
        
        # measure similarity and remove the most similar one
        if len(tmp) >= self.buffer_size:
            emb_dists = embedding_distance(np.vstack(tmp), np.array([feat]))
            idx = np.argmin(emb_dists)
            tmp.pop(idx)
        tmp.append(feat)
        self.features = tmp
            

    def update(self, bbox, id_feature, update_feature=True):
        """
        Updates the state vector with observed bbox.
        """
        if bbox is not None:
            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age-dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                """
                  Estimate the track speed direction with observations \Delta t steps away
                """
                self.velocity = speed_direction(previous_box, bbox)
            
            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(convert_bbox_to_z(bbox))
            # add interface for update feature or not
            if update_feature:
                    self.update_features(id_feature)
        else:
            self.kf.update(bbox)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


"""
    We support multiple ways for association cost calculation, by default
    we use IoU. GIoU may have better performance in some situations. We note 
    that we hardly normalize the cost by all methods to (0,1) which may not be 
    the best practice.
"""
ASSO_FUNCS = {  "iou": iou_batch,
                "giou": giou_batch,
                "ciou": ciou_batch,
                "diou": diou_batch,
                "ct_dist": ct_dist,
                "Height_Modulated_IoU": hmiou
                }


class OCSORT_ReIDv1(object):
    def __init__(self, args, det_thresh, max_age=30, min_hits=3,
        iou_threshold=0.3, delta_t=3, asso_func="iou", inertia=0.2):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.use_byte = args.use_byte
        self.args = args
        KalmanBoxTrackerv1.count = 0

    def update(self, output_results, img_info, img_size, id_feature=None, get_embeddings=None):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        if get_embeddings is not None:
            player_embeddings = {}
        if output_results is None:
            if get_embeddings is not None:
                return np.empty((0, 5)), player_embeddings
            else:
                return np.empty((0, 5))

        self.frame_count += 1
        # post_process detections
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale
        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)
        inds_low = scores > 0.1
        inds_high = scores < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)  # self.det_thresh > score > 0.1, for second matching
        dets_second = dets[inds_second]  # detections for second matching
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]
        id_feature = np.asarray([track.detach().cpu().numpy() for track in id_feature],dtype=np.float64)
        if len(id_feature.shape)<2:
            id_feature = np.expand_dims(id_feature,0)
        id_feature_keep = id_feature[remain_inds]  # ID feature of 1st stage matching
        id_feature_second = id_feature[inds_second]  # ID feature of 2nd stage matching

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array(
            [trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array(
            [k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        """
            First round of association
        """
        matched, unmatched_dets, unmatched_trks = associate(dets, trks, self.iou_threshold, velocities, k_observations, self.inertia)

        # update with id feature
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :], id_feature_keep[m[0], :])
            if get_embeddings is not None:
                player_embeddings[m[1]] = id_feature_keep[m[0], :]

        """
            Second round of associaton by OCR
        """
        # BYTE association
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]
            iou_left = self.asso_func(dets_second, u_trks)          # iou between low score detections and unmatched tracks
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                matched_indices = linear_assignment(-iou_left)
                to_remove_trk_indices = []
                for m in matched_indices:
                    det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets_second[det_ind, :], id_feature_second[det_ind, :])
                    if get_embeddings is not None:
                        player_embeddings[trk_ind] = id_feature_second[det_ind, :]
                    to_remove_trk_indices.append(trk_ind)
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = self.asso_func(left_dets, left_trks)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :], id_feature_keep[det_ind, :])
                    if get_embeddings is not None:
                        player_embeddings[trk_ind] = id_feature_keep[det_ind, :]
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks:
            self.trackers[m].update(None, None)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTrackerv1(dets[i, :], id_feature_keep[i, :], delta_t=self.delta_t, args=self.args)
            self.trackers.append(trk)
            if get_embeddings is not None:
                player_embeddings[trk.id] = id_feature_keep[i, :]
        i = len(self.trackers)
        if get_embeddings is not None:
            tmp_emb = {}
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0][:4]
            else:
                """
                    this is optional to use the recent observation or the kalman filter prediction,
                    we didn't notice significant difference here
                """
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
                if get_embeddings is not None:
                    tmp_emb[trk.id+1] = player_embeddings[trk.id]
            i -= 1
            # remove dead tracklet (no observation for long time, try to recover the lost tracklet)
            # if(trk.time_since_update > self.max_age):
            #     self.trackers.pop(i)
        if get_embeddings is not None:
            player_embeddings = tmp_emb
            if(len(ret) > 0):
                return np.concatenate(ret), player_embeddings
            return np.empty((0, 5)), player_embeddings
        if(len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))



