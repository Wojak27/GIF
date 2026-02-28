from __future__ import print_function, absolute_import
import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np

from .utils import mkdir_if_missing, write_json, read_json

"""Dataset classes"""
# For CC MuPNIT
def extract_player_id(filename):
    # Extract the player id from the first token.
    return int(filename.split('_')[0])
# For CC MuPNIT
def extract_cloth_label(filename):
    # Extract the cloth label from the second token.
    parts = filename.split('_')
    if len(parts) < 2:
        return None
    return parts[1]

class Mars(object):
    def __init__(self, root='../../mevid/', min_seq_len=0):
        self.root = root
        self.train_name_path = osp.join(self.root, 'train_name.txt')
        self.test_name_path = osp.join(self.root, 'test_name.txt')
        self.track_train_info_path = osp.join(self.root, 'track_train_info.txt')
        self.track_test_info_path = osp.join(self.root, 'track_test_info.txt')
        self.query_IDX_path = osp.join(self.root, 'query_IDX.txt')
        
        self._check_before_run()

        # prepare meta data
        train_names = self._get_names(self.train_name_path)
        test_names = self._get_names(self.test_name_path)
        track_train = np.loadtxt(self.track_train_info_path).astype(np.int)
        track_test = np.loadtxt(self.track_test_info_path).astype(np.int)
        query_IDX = np.loadtxt(self.query_IDX_path).astype(np.int) 
        #query_IDX -= 1 # index from 0
        track_query = track_test[query_IDX,:]
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX,:]
        # track_gallery = track_test

        train, num_train_tracklets, num_train_pids, num_train_imgs = \
          self._process_data(train_names, track_train, home_dir='bbox_train', relabel=True, min_seq_len=min_seq_len)

        query, num_query_tracklets, num_query_pids, num_query_imgs = \
          self._process_data(test_names, track_query, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = \
          self._process_data(test_names, track_gallery, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        num_imgs_per_tracklet = num_train_imgs + num_gallery_imgs + num_query_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_gallery_pids
        num_total_tracklets = num_train_tracklets + num_gallery_tracklets + num_query_tracklets

        print("=> MEVID loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _process_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0):
        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,2].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            start_index, end_index, pid, oid, camid = data
            if pid == -1: continue # junk images are just ignored
            #assert 1 <= camid <= 6
            if relabel: pid = pid2label[pid]
            #camid -= 1 # index starts from 0
            img_names = names[start_index:end_index+1]
            if len(img_names) == 0:
                continue

            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera
            camnames = [img_name[9:12] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid, oid))
                num_imgs_per_tracklet.append(len(img_paths))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

# With CC   
class MuPNIT(object):
    """
    MuPNIT Dataset for video-based person re-identification.
    
    Folder structure:
      root/
         all_train/       --> training images
         query/           --> query images (assigned camid = 0)
         gallery_test/    --> gallery images (assigned camid = 1)
    
    Cloth label is extracted from the second token in the filename.
    Images with the same (player id, cloth label) are grouped into one tracklet.
    """
    def __init__(self, root, min_seq_len=1, relabel=True):
        self.root = root  # reid dataset root
        self.train_dir = osp.join(self.root, 'all_train')
        self.query_dir = osp.join(self.root, 'query')
        self.gallery_dir = osp.join(self.root, 'gallery_test')
        self.min_seq_len = min_seq_len
        
        # Process each split using the helper function:
        self.train, self.num_train_tracklets, self.num_train_pids, _ = \
            self._process_dir(self.train_dir, relabel=relabel, camid=0)
        self.query, self.num_query_tracklets, self.num_query_pids, _ = \
            self._process_dir(self.query_dir, relabel=False, camid=0)
        self.gallery, self.num_gallery_tracklets, self.num_gallery_pids, _ = \
            self._process_dir(self.gallery_dir, relabel=False, camid=1)
        
        print("=> MuPNIT loaded")
        print("Dataset statistics:")
        print("  train    | # ids: {} | # tracklets: {}".format(self.num_train_pids, self.num_train_tracklets))
        print("  query    | # ids: {} | # tracklets: {}".format(self.num_query_pids, self.num_query_tracklets))
        print("  gallery  | # ids: {} | # tracklets: {}".format(self.num_gallery_pids, self.num_gallery_tracklets))
    
    def _process_dir(self, dir_path, relabel=False, camid=0):
        img_files = [f for f in os.listdir(dir_path) if f.endswith('.jpg')]
        img_files.sort()
        
        # Group images by (player id, cloth label)
        tracklet_dict = {}
        for fname in img_files:
            pid = extract_player_id(fname)
            cloth = extract_cloth_label(fname)
            key = (pid, cloth)
            if key not in tracklet_dict:
                tracklet_dict[key] = []
            tracklet_dict[key].append(osp.join(dir_path, fname))
        
        tracklets = []
        pid_set = set()
        for (pid, cloth), paths in tracklet_dict.items():
            if len(paths) < self.min_seq_len:
                continue
            pid_set.add(pid)
            paths.sort()
            # Create the tracklet tuple: (tuple(img_paths), pid, camid, cloth)
            tracklets.append((tuple(paths), pid, camid, cloth))
        num_tracklets = len(tracklets)
        num_pids = len(pid_set)
        if relabel:
            pid_container = sorted(list(pid_set))
            pid2label = {pid: label for label, pid in enumerate(pid_container)}
            new_tracklets = []
            for (paths, pid, camid, cloth) in tracklets:
                new_tracklets.append((paths, pid2label[pid], camid, cloth))
            tracklets = new_tracklets
            num_pids = len(pid_container)
        return tracklets, num_tracklets, num_pids, [len(paths) for paths in tracklet_dict.values()]

# ==== NEW: SoccerNetV3 ReID (image-based, grouped into per-player tracklets) ====
class SoccerNetV3ReID(object):
    """
    SoccerNet ReID 2023

    Folder structure (root points to .../reid-2023):
      root/
         train/                      -> images (no query/gallery split)
         valid/                      -> query/ + gallery/
         test/                       -> query/ + gallery/
         train_bbox_info.json        -> in train/
         valid/bbox_info.json
         test/bbox_info.json

    Key points:
    - Identity consistency only holds *within* each action.
    - We therefore build global pids as (action_idx, person_uid).
    - camid: set to frame_idx so that query (frame 0) vs gallery (>0) matches aren't suppressed.
    - oid: use the "id" field (jersey number if visible, else a letter, else "None") for clothes eval.
    """

    def __init__(self, root, min_seq_len=1, relabel=True, eval_split='test'):
        """
        Args:
            root (str): path to 'reid-2023' directory
            min_seq_len (int): min frames per tracklet for training
            relabel (bool): relabel training pids to 0..N-1
            eval_split (str): which split to evaluate on: 'valid' or 'test'
        """
        import os
        import os.path as osp
        from collections import defaultdict

        self.root = root
        self.train_dir = osp.join(self.root, 'train')
        self.valid_dir = osp.join(self.root, 'valid')
        self.test_dir = osp.join(self.root, 'test')
        self.train_json = osp.join(self.train_dir, 'train_bbox_info.json')
        self.valid_json = osp.join(self.valid_dir, 'bbox_info.json')
        self.test_json = osp.join(self.test_dir, 'bbox_info.json')

        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.train_json):
            raise RuntimeError("'{}' is not available".format(self.train_json))
        if eval_split not in ['valid', 'test']:
            raise ValueError("eval_split must be 'valid' or 'test', got {}".format(eval_split))

        # ---------- helpers ----------
        def _compose_filename(m):
            # Matches SoccerNet filename convention exactly
            return "{}-{}-{}-{}-{}-{}-{}-{}x{}.png".format(
                m['bbox_idx'], m['action_idx'], m['person_uid'], m['frame_idx'],
                m['clazz'], m['id'], m['UAI'], m['height'], m['width']
            )

        def _parse_filename(basename):
            # Robust parse of "<bbox>-<action>-<person>-<frame>-<class>-<id>-<UAI>-<HxW>.png"
            # class part is underscore-separated, so simple split('-') is safe.
            name = basename[:-4]  # strip .png
            parts = name.split('-')
            if len(parts) < 8:
                raise ValueError("Unexpected filename format: {}".format(basename))
            bbox_idx = int(parts[0])
            action_idx = int(parts[1])
            person_uid = int(parts[2])
            frame_idx = int(parts[3])
            clazz = parts[4]                      # e.g., Player_team_left
            id_field = parts[5]                   # jersey number / letter / None
            uai = parts[6]
            hw = parts[7]
            if 'x' not in hw:
                # Sometimes size is the 8th part; if there are extra hyphens (rare),
                # rejoin the tail and split the last token.
                hw = parts[-1]
            h_str, w_str = hw.split('x')
            height, width = int(h_str), int(w_str)
            return {
                'bbox_idx': bbox_idx,
                'action_idx': action_idx,
                'person_uid': person_uid,
                'frame_idx': frame_idx,
                'clazz': clazz,
                'id': id_field,
                'UAI': uai,
                'height': height,
                'width': width
            }

        def _group_train_tracklets():
            # Build tracklets for training by grouping all images with same (action_idx, person_uid).
            meta = read_json(self.train_json)
            groups = defaultdict(list)  # key: (action_idx, person_uid) -> list of dicts with paths & meta

            for k, m in meta.items():
                rel = m['relative_path']  # e.g., "england_epl/2014-2015/.../0"
                rel_dir = osp.join(self.train_dir, m['relative_path'])
                prefix = "{}-{}-{}-{}".format(m['bbox_idx'], m['action_idx'], m['person_uid'], m['frame_idx'])
                candidates = [f for f in os.listdir(rel_dir) if f.startswith(prefix) and f.endswith('.png')]
                if len(candidates) == 0:
                    raise RuntimeError("Could not find file for meta {} in {}".format(m, rel_dir))
                fname = candidates[0]
                fpath = osp.join(rel_dir, fname)

                key = (int(m['action_idx']), int(m['person_uid']))
                groups[key].append({
                    'frame_idx': int(m['frame_idx']),
                    'path': fpath,
                    'id': str(m['id'])  # jersey info as string
                })

            # Now convert to tracklets [(tuple(paths), pid, camid, oid)]
            tracklets = []
            pid_map = {}  # (action_idx, person_uid) -> new label
            next_pid = 0
            num_imgs_per_tracklet = []

            for key, items in groups.items():
                # sort by frame index for temporal order
                items.sort(key=lambda x: (x['frame_idx'], x['path']))
                paths = [it['path'] for it in items]
                if len(paths) < min_seq_len:
                    # skip short tracklets
                    continue

                if relabel:
                    if key not in pid_map:
                        pid_map[key] = next_pid
                        next_pid += 1
                    pid = pid_map[key]
                else:
                    # build a stable integer label from key (should not be used during training)
                    pid = hash(key) % (10**9)

                # training camid not used; set to 0
                camid = 0
                # choose first jersey string as oid (for clothes eval); safe as strings
                oid = items[0]['id'] if len(items) > 0 else 'None'

                tracklets.append((tuple(paths), pid, camid, oid))
                num_imgs_per_tracklet.append(len(paths))

            num_tracklets = len(tracklets)
            num_pids = len(pid_map) if relabel else len(groups)
            return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

        def _walk_images(root_dir):
            for dp, _, files in os.walk(root_dir):
                for f in files:
                    if f.endswith('.png'):
                        yield osp.join(dp, f)

        def _build_eval(split_dir):
            """
            Build query/gallery lists for evaluation from file names only.
            Each sample is treated as a single-frame tracklet: ( (img_path,), pid, camid, oid ).
            """
            query_dir = osp.join(split_dir, 'query')
            gallery_dir = osp.join(split_dir, 'gallery')
            if not osp.exists(query_dir) or not osp.exists(gallery_dir):
                raise RuntimeError("Missing query/gallery in '{}'".format(split_dir))

            # Build a consistent pid mapping across query and gallery
            pid_map = {}
            next_pid = 0

            def encode_pid(action_idx, person_uid):
                key = (action_idx, person_uid)
                nonlocal pid_map, next_pid
                if key not in pid_map:
                    pid_map[key] = next_pid
                    next_pid += 1
                return pid_map[key]

            def build_side(side_dir, is_query):
                items = []
                for img_path in _walk_images(side_dir):
                    meta = _parse_filename(osp.basename(img_path))
                    # pid = f"{meta['action_idx']}_{meta['person_uid']}"  # Could be string
                    pid = encode_pid(meta['action_idx'], meta['person_uid'])  # use ints
                    camid = int(meta['frame_idx'])  # 0 for action frame (query), >0 for replays (gallery)
                    oid = str(meta['id'])          # jersey info as string
                    # one-image "tracklet"
                    items.append(((img_path,), pid, camid, oid))
                return items

            queryset = build_side(query_dir, is_query=True)
            galleryset = build_side(gallery_dir, is_query=False)
            num_query_pids = len(set([it[1] for it in queryset]))
            num_gallery_pids = len(set([it[1] for it in galleryset]))
            num_query_tracklets = len(queryset)
            num_gallery_tracklets = len(galleryset)
            return queryset, num_query_tracklets, num_query_pids, galleryset, num_gallery_tracklets, num_gallery_pids

        # ---------- construct splits ----------
        train, num_train_tracklets, num_train_pids, num_imgs_train = _group_train_tracklets()
        eval_dir = self.test_dir if eval_split == 'test' else self.valid_dir
        query, num_query_tracklets, num_query_pids, gallery, num_gallery_tracklets, num_gallery_pids = _build_eval(eval_dir)

        # --- logs (simple) ---
        import numpy as np
        num_imgs_per_tracklet = list(num_imgs_train)
        if len(num_imgs_per_tracklet) == 0:
            min_num = max_num = avg_num = 0
        else:
            min_num = np.min(num_imgs_per_tracklet)
            max_num = np.max(num_imgs_per_tracklet)
            avg_num = np.mean(num_imgs_per_tracklet)

        print("=> SoccerNetV3 ReID loaded (train on 'train', eval on '{}')".format(eval_split))
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  number of images per tracklet (train): {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids


class iLIDSVID(object):
    """
    iLIDS-VID

    Reference:
    Wang et al. Person Re-Identification by Video Ranking. ECCV 2014.
    
    Dataset statistics:
    # identities: 300
    # tracklets: 600
    # cameras: 2

    Args:
        split_id (int): indicates which split to use. There are totally 10 splits.
    """
    def __init__(self, root='/data/datasets/', split_id=9):
        self.root = osp.join(root, 'iLIDS-VID')
        self.dataset_url = 'http://www.eecs.qmul.ac.uk/~xiatian/iLIDS-VID/iLIDS-VID.tar'
        self.data_dir = osp.join(self.root, 'i-LIDS-VID')
        self.split_dir = osp.join(self.root, 'train-test people splits')
        self.split_mat_path = osp.join(self.split_dir, 'train_test_splits_ilidsvid.mat')
        self.split_path = osp.join(self.root, 'splits.json')
        self.cam_1_path = osp.join(self.root, 'i-LIDS-VID/sequences/cam1')
        self.cam_2_path = osp.join(self.root, 'i-LIDS-VID/sequences/cam2')
        # self._download_data()
        self._check_before_run()

        self._prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError("split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits)-1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
          self._process_data(train_dirs, cam1=True, cam2=True)
        train_dense, _, _, _ = self._process_train_data(train_dirs, cam1=True, cam2=True)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
          self._process_data(test_dirs, cam1=True, cam2=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
          self._process_data(test_dirs, cam1=False, cam2=True)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> iLIDS-VID loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.train_dense = train_dense
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _download_data(self):
        if osp.exists(self.root):
            print("This dataset has been downloaded.")
            return

        mkdir_if_missing(self.root)
        fpath = osp.join(self.root, osp.basename(self.dataset_url))

        print("Downloading iLIDS-VID dataset")
        url_opener = urllib.URLopener()
        url_opener.retrieve(self.dataset_url, fpath)

        print("Extracting files")
        tar = tarfile.open(fpath)
        tar.extractall(path=self.root)
        tar.close()

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))
        if not osp.exists(self.split_dir):
            raise RuntimeError("'{}' is not available".format(self.split_dir))

    def _prepare_split(self):
        if not osp.exists(self.split_path):
            print("Creating splits")
            mat_split_data = loadmat(self.split_mat_path)['ls_set']
            
            num_splits = mat_split_data.shape[0]
            num_total_ids = mat_split_data.shape[1]
            assert num_splits == 10
            assert num_total_ids == 300
            num_ids_each = num_total_ids/2

            # pids in mat_split_data are indices, so we need to transform them
            # to real pids
            person_cam1_dirs = os.listdir(self.cam_1_path)
            person_cam2_dirs = os.listdir(self.cam_2_path)

            # make sure persons in one camera view can be found in the other camera view
            assert set(person_cam1_dirs) == set(person_cam2_dirs)

            splits = []
            for i_split in range(num_splits):
                # first 50% for testing and the remaining for training, following Wang et al. ECCV'14.
                train_idxs = sorted(list(mat_split_data[i_split,num_ids_each:]))
                test_idxs = sorted(list(mat_split_data[i_split,:num_ids_each]))
                
                train_idxs = [int(i)-1 for i in train_idxs]
                test_idxs = [int(i)-1 for i in test_idxs]
                
                # transform pids to person dir names
                train_dirs = [person_cam1_dirs[i] for i in train_idxs]
                test_dirs = [person_cam1_dirs[i] for i in test_idxs]
                
                split = {'train': train_dirs, 'test': test_dirs}
                splits.append(split)

            print("Totally {} splits are created, following Wang et al. ECCV'14".format(len(splits)))
            print("Split file is saved to {}".format(self.split_path))
            write_json(splits, self.split_path)

        print("Splits created")

    def _process_data(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        num_imgs_per_tracklet = []
        dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}
        
        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_1_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))
                num_imgs_per_tracklet.append(len(img_names))

            if cam2:
                person_dir = osp.join(self.cam_2_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))
                num_imgs_per_tracklet.append(len(img_names))

        num_tracklets = len(tracklets)
        num_pids = len(dirnames)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

    def _process_train_data(self, dirnames, cam1=True, cam2=True, sampling_step=32):
        tracklets = []
        num_imgs_per_tracklet = []
        dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}
        
        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_1_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                # tracklets.append((img_names, pid, 1))
                # dense sampling
                num_sampling = len(img_names)//sampling_step
                if num_sampling == 0:
                    tracklets.append((img_names, pid, 1))
                else:
                    for idx in range(num_sampling):
                        if idx == num_sampling - 1:
                            tracklets.append((img_names[idx*sampling_step:], pid, 1))
                        else:
                            tracklets.append((img_names[idx*sampling_step : (idx+2)*sampling_step], pid, 1))

                num_imgs_per_tracklet.append(len(img_names))

            if cam2:
                person_dir = osp.join(self.cam_2_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                # tracklets.append((img_names, pid, 1))
                # dense sampling
                num_sampling = len(img_names)//sampling_step
                if num_sampling == 0:
                    tracklets.append((img_names, pid, 1))
                else:
                    for idx in range(num_sampling):
                        if idx == num_sampling - 1:
                            tracklets.append((img_names[idx*sampling_step:], pid, 1))
                        else:
                            tracklets.append((img_names[idx*sampling_step : (idx+2)*sampling_step], pid, 1))

                num_imgs_per_tracklet.append(len(img_names))

        num_tracklets = len(tracklets)
        num_pids = len(dirnames)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


class DukeMTMCVidReID(object):
    """
    DukeMTMCVidReID
    Reference:
    Wu et al. Exploit the Unknown Gradually: One-Shot Video-Based Person
    Re-Identification by Stepwise Learning. CVPR 2018.
    URL: https://github.com/Yu-Wu/DukeMTMC-VideoReID
    
    Dataset statistics:
    # identities: 702 (train) + 702 (test)
    # tracklets: 2196 (train) + 2636 (test)
    """

    def __init__(self, root='/data/datasets/', min_seq_len=0, verbose=True, **kwargs):
        self.dataset_dir = osp.join(root, 'DukeMTMC-VideoReID')
        self.dataset_url = 'http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-VideoReID.zip'
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
        self.split_train_json_path = osp.join(self.dataset_dir, 'split_train.json')
        self.split_train_dense_json_path = osp.join(self.dataset_dir, 'split_train_dense.json')
        self.split_query_json_path = osp.join(self.dataset_dir, 'split_query.json')
        self.split_gallery_json_path = osp.join(self.dataset_dir, 'split_gallery.json')


        self.min_seq_len = min_seq_len
        # self._download_data()
        self._check_before_run()
        print("Note: if root path is changed, the previously generated json files need to be re-generated (so delete them first)")

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
          self._process_dir(self.train_dir, self.split_train_json_path, relabel=True)
        train_dense, num_train_tracklets_dense, num_train_pids_dense, num_imgs_train_dense = \
          self._process_dir_dense(self.train_dir, self.split_train_dense_json_path, relabel=True, sampling_step=64)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
          self._process_dir(self.query_dir, self.split_query_json_path, relabel=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
          self._process_dir(self.gallery_dir, self.split_gallery_json_path, relabel=False)

        print("the number of tracklets under dense sampling for train set: {}".format(num_train_tracklets_dense))

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        if verbose:
            print("=> DukeMTMC-VideoReID loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset   | # ids | # tracklets")
            print("  ------------------------------")
            print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
            print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
            print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
            print("  ------------------------------")
            print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
            print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
            print("  ------------------------------")

        self.train = train
        self.train_dense = train_dense
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, json_path, relabel):
        if osp.exists(json_path):
            print("=> {} generated before, awesome!".format(json_path))
            split = read_json(json_path)
            return split['tracklets'], split['num_tracklets'], split['num_pids'], split['num_imgs_per_tracklet']

        print("=> Automatically generating split (might take a while for the first time, have a coffe)")
        pdirs = glob.glob(osp.join(dir_path, '*')) # avoid .DS_Store
        print("Processing {} with {} person identities".format(dir_path, len(pdirs)))

        pid_container = set()
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        tracklets = []
        num_imgs_per_tracklet = []
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            if relabel: pid = pid2label[pid]
            tdirs = glob.glob(osp.join(pdir, '*'))
            for tdir in tdirs:
                raw_img_paths = glob.glob(osp.join(tdir, '*.jpg'))
                num_imgs = len(raw_img_paths)

                if num_imgs < self.min_seq_len:
                    continue

                num_imgs_per_tracklet.append(num_imgs)
                img_paths = []
                for img_idx in range(num_imgs):
                    # some tracklet starts from 0002 instead of 0001
                    img_idx_name = 'F' + str(img_idx+1).zfill(4)
                    res = glob.glob(osp.join(tdir, '*' + img_idx_name + '*.jpg'))
                    if len(res) == 0:
                        print("Warn: index name {} in {} is missing, jump to next".format(img_idx_name, tdir))
                        continue
                    img_paths.append(res[0])
                img_name = osp.basename(img_paths[0])
                if img_name.find('_') == -1:
                    # old naming format: 0001C6F0099X30823.jpg
                    camid = int(img_name[5]) - 1
                else:
                    # new naming format: 0001_C6_F0099_X30823.jpg
                    camid = int(img_name[6]) - 1
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))

        num_pids = len(pid_container)
        num_tracklets = len(tracklets)

        print("Saving split to {}".format(json_path))
        split_dict = {
            'tracklets': tracklets,
            'num_tracklets': num_tracklets,
            'num_pids': num_pids,
            'num_imgs_per_tracklet': num_imgs_per_tracklet,
        }
        write_json(split_dict, json_path)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

    def _process_dir_dense(self, dir_path, json_path, relabel, sampling_step=32):
        if osp.exists(json_path):
            print("=> {} generated before, awesome!".format(json_path))
            split = read_json(json_path)
            return split['tracklets'], split['num_tracklets'], split['num_pids'], split['num_imgs_per_tracklet']

        print("=> Automatically generating split (might take a while for the first time, have a coffe)")
        pdirs = glob.glob(osp.join(dir_path, '*')) # avoid .DS_Store
        print("Processing {} with {} person identities".format(dir_path, len(pdirs)))

        pid_container = set()
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        tracklets = []
        num_imgs_per_tracklet = []
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            if relabel: pid = pid2label[pid]
            tdirs = glob.glob(osp.join(pdir, '*'))
            for tdir in tdirs:
                raw_img_paths = glob.glob(osp.join(tdir, '*.jpg'))
                num_imgs = len(raw_img_paths)

                if num_imgs < self.min_seq_len:
                    continue

                num_imgs_per_tracklet.append(num_imgs)
                img_paths = []
                for img_idx in range(num_imgs):
                    # some tracklet starts from 0002 instead of 0001
                    img_idx_name = 'F' + str(img_idx+1).zfill(4)
                    res = glob.glob(osp.join(tdir, '*' + img_idx_name + '*.jpg'))
                    if len(res) == 0:
                        print("Warn: index name {} in {} is missing, jump to next".format(img_idx_name, tdir))
                        continue
                    img_paths.append(res[0])
                img_name = osp.basename(img_paths[0])
                if img_name.find('_') == -1:
                    # old naming format: 0001C6F0099X30823.jpg
                    camid = int(img_name[5]) - 1
                else:
                    # new naming format: 0001_C6_F0099_X30823.jpg
                    camid = int(img_name[6]) - 1
                img_paths = tuple(img_paths)

                # dense sampling
                num_sampling = len(img_paths)//sampling_step
                if num_sampling == 0:
                    tracklets.append((img_paths, pid, camid))
                else:
                    for idx in range(num_sampling):
                        if idx == num_sampling - 1:
                            tracklets.append((img_paths[idx*sampling_step:], pid, camid))
                        else:
                            tracklets.append((img_paths[idx*sampling_step : (idx+1)*sampling_step], pid, camid))

        num_pids = len(pid_container)
        num_tracklets = len(tracklets)

        print("Saving split to {}".format(json_path))
        split_dict = {
            'tracklets': tracklets,
            'num_tracklets': num_tracklets,
            'num_pids': num_pids,
            'num_imgs_per_tracklet': num_imgs_per_tracklet,
        }
        write_json(split_dict, json_path)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


"""Create dataset"""

__factory = {
    'mars': Mars,
    'ilidsvid': iLIDSVID,
    'duke': DukeMTMCVidReID,
    'mupnit': MuPNIT,
    'soccernetv3': SoccerNetV3ReID,   # <-- NEW
}

def get_names():
    return __factory.keys()

def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](*args, **kwargs)
