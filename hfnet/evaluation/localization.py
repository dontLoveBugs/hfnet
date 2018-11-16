import numpy as np
import logging
from pathlib import Path
import pickle
from scipy.spatial import cKDTree
import sys
from tqdm import tqdm
import cv2
import yaml

from hfnet.datasets import get_dataset
from .utils import db_management
from .utils.db_management import (
    read_query_list, extract_query, build_localization_dbs,
    colmap_image_to_pose)
from .utils.localization import (
    covis_clustering, match_against_place, do_pnp, preprocess_globaldb,
    preprocess_localdb, loc_failure, LocResult)
from hfnet.datasets.colmap_utils.read_model import read_model
from hfnet.utils.tools import Timer  # noqa: F401 (profiling)
from hfnet.settings import DATA_PATH

sys.modules['hfnet.evaluation.db_management'] = db_management  # backward comp


class Localization:
    def __init__(self, dataset_name, model_name, config, build_db=False):
        base_path = Path(DATA_PATH, dataset_name)

        logging.info(f'Importing COLMAP model {model_name}')
        self.cameras, self.images, self.points = read_model(
            path=Path(base_path, 'models', model_name).as_posix(), ext='.bin')
        self.db_ids = np.array(list(self.images.keys()))
        self.db_names = [self.images[i].name for i in self.db_ids]

        # Statistics for debugging
        kpts_per_image = np.median(np.array(
            [len(i.point3D_ids) for i in self.images.values()]))
        obs_per_image = np.nanmean(np.array(
            [np.mean(i.point3D_ids > 0) for i in self.images.values()]))
        logging.info(
            f'Number of images: {len(self.images)}\n'
            f'Number of points: {len(self.points)}\n'
            f'Median keypoints per image: {kpts_per_image}\n'
            f'Ratio of matched keypoints: {obs_per_image:.3f}\n'
        )

        # Resolve paths in config
        if 'colmap_db' in config['local']:
            db_path = Path(base_path, config['local']['colmap_db'])
            config['local']['colmap_db'] = db_path.as_posix()
        if 'colmap_db_queries' in config['local']:
            db_path = Path(base_path, config['local']['colmap_db_queries'])
            config['local']['colmap_db_queries'] = db_path.as_posix()
        global_path = Path(base_path, 'databases', config['global']['db_name'])
        local_path = Path(base_path, 'databases', config['local']['db_name'])

        # Build databases if necessary
        ok_global, ok_local = global_path.exists(), local_path.exists()
        if not ok_global or not ok_local:
            if build_db:
                logging.info('Starting to build databases: '
                             f'global: {not ok_global}, local: {not ok_local}')
                global_descriptors, local_db = build_localization_dbs(
                    self.db_ids, self.images, self.cameras,
                    config_global=None if ok_global else config['global'],
                    config_local=None if ok_local else config['local'])
                if not ok_global:
                    with open(global_path, 'wb') as f:
                        pickle.dump((self.db_names, global_descriptors), f)
                if not ok_local:
                    with open(local_path, 'wb') as f:
                        pickle.dump(local_db, f)
            else:
                raise IOError('Database files do not exist, '
                              'build must be enabled with --build_db')

        logging.info('Importing global and local databases')
        with open(global_path, 'rb') as f:
            globaldb_names, global_descriptors = pickle.load(f)
            assert isinstance(globaldb_names[0], str)
            name_to_id = {name: i for i, name in enumerate(globaldb_names)}
            mapping = np.array([name_to_id[n] for n in self.db_names])
            global_descriptors = global_descriptors[mapping]
        with open(local_path, 'rb') as f:
            local_db = pickle.load(f)

        logging.info('Indexing descriptors')
        self.global_descriptors, self.global_transform = preprocess_globaldb(
            global_descriptors, config['global'])
        self.global_index = cKDTree(self.global_descriptors)
        self.local_db, self.local_transform = preprocess_localdb(
            local_db, config['local'])

        self.base_path = base_path
        self.dataset_name = dataset_name
        self.config = config

        self.use_cpp = config.get('use_cpp')
        if self.use_cpp:
            self.init_cpp()

    def init_cpp(self):
        import _hfnetcpp
        self.cpp_backend = _hfnetcpp.HfNet()
        id_to_idx = {}
        old_to_new_kpt = {}
        for idx, i in enumerate(self.db_ids):
            #logging.info('Adding image %s', idx)
            item = self.local_db[i]
            keypoints = item.keypoints.T.astype(np.float32).copy()
            local_desc = item.descriptors.T.astype(np.float32).copy()
            global_desc = self.global_descriptors[idx].astype(np.float32).copy()
            # keypoints are NOT undistorted or nomalized
            idx = self.cpp_backend.addImage(global_desc, keypoints, local_desc)
            id_to_idx[i] = idx
            old_to_new_kpt[i] = {k: j for j, k in enumerate(np.where(self.images[i].point3D_ids>=0)[0])}
        for i, point in self.points.items():
            observations = np.array([
                [id_to_idx[im_id], old_to_new_kpt[im_id][kpt_id]]
                for im_id, kpt_id in zip(point.image_ids, point.point2D_idxs)], dtype=np.int32)
            self.cpp_backend.add3dPoint(
                point.xyz.astype(np.float32).copy(), observations.copy())
        self.cpp_backend.buildIndex()

    def init_queries(self, query_file, query_config, prefix=''):
        queries = read_query_list(
            Path(self.base_path, query_file), prefix=prefix)
        Dataset = get_dataset(query_config.get('name', self.dataset_name))
        query_config = {
            **query_config, 'image_names': [q.name for q in queries]}
        query_dataset = Dataset(**query_config)
        return queries, query_dataset

    def localize_cpp(self, query_info, query_item):
        assert self.config['use_cpp']
        global_desc = self.global_transform(
            query_item.global_desc[np.newaxis])[0].astype(np.float32)
        local_desc = self.local_transform(
            query_item.local_desc).astype(np.float32).T.copy()
        keypoints = cv2.undistortPoints(
            query_item.keypoints[np.newaxis], query_info.K,
            np.array([query_info.dist, 0, 0, 0]))[0].astype(np.float32).T.copy()

        logging.info('Localizing image %s', query_info.name)
        success, num_components_total, num_components_tested, last_component_size, num_db_landmarks, num_matches, \
            num_inliers, num_iters, \
        global_ms, covis_ms, local_ms, pnp_ms = self.cpp_backend.localize(
               global_desc, keypoints, local_desc)

        result = LocResult(success, num_inliers, 0, np.eye(4))
        stats = {
            'success': success,
            'num_components_total': num_components_total,
            'num_components_tested': num_components_tested,
            'last_component_size': last_component_size,
            'num_db_landmarks': num_db_landmarks,
            'num_matches': num_matches,
            'num_inliers': num_inliers,
            'num_ransac_iters': num_iters,
            'timings': {
                'global': global_ms,
                'covis': covis_ms,
                'local': local_ms,
                'pnp': pnp_ms,
            }
        }
        return (result, stats)

    def localize(self, query_info, query_data, debug=False):
        config_global = self.config['global']
        config_local = self.config['local']
        config_pose = self.config['pose']

        # Fetch data
        query_item = extract_query(
            query_data, query_info, config_global, config_local)

        if self.config.get('use_cpp', False):
            assert not debug
            return self.localize_cpp(query_info, query_item)

        # Global matching
        global_desc = self.global_transform(
            query_item.global_desc[np.newaxis])[0]
        dist, indices = self.global_index.query(
            global_desc, k=config_global['num_prior'])
        prior_ids = self.db_ids[indices]

        # Local matching
        clustered_frames = covis_clustering(
            prior_ids, self.local_db, self.points)
        local_desc = self.local_transform(query_item.local_desc)

        # Iterative pose estimation
        dump = []
        results = []
        for place in clustered_frames:
            matches_data = {} if debug else None
            matches, place_lms = match_against_place(
                place, self.local_db, local_desc, config_local['ratio_thresh'],
                debug_dict=matches_data)

            if len(matches) > 3:
                matched_kpts = query_item.keypoints[matches[:, 0]]
                matched_lms = np.stack(
                    [self.points[place_lms[i]].xyz for i in matches[:, 1]])
                result, inliers = do_pnp(
                    matched_kpts, matched_lms, query_info, config_pose)
            else:
                result = loc_failure
                inliers = np.empty((0,), np.int32)

            results.append(result)
            if debug:
                dump.append({
                    'query_item': query_item,
                    'prior_ids': prior_ids,
                    'places': clustered_frames,
                    'matching': matches_data,
                    'matches': matches,
                    'inliers': inliers,
                })
            if result.success:
                break

        # In case of failure we return the pose of the first retrieved prior
        if not result.success:
            result = results[0]
            result = LocResult(False, result.num_inliers, result.inlier_ratio,
                               colmap_image_to_pose(self.images[prior_ids[0]]))

        if debug:
            debug_data = {
                **(dump[-1 if result.success else 0]),
                'index_success': (len(dump)-1) if result.success else -1,
                'dumps': dump,
                'results': results,
            }
            return result, debug_data
        else:
            return result, None


def evaluate(loc, queries, query_dataset, max_iter=None):
    results = []
    all_stats = []
    query_iter = query_dataset.get_test_set()

    for query_info, query_data in tqdm(zip(queries, query_iter)):
        result, stats = loc.localize(query_info, query_data, debug=False)
        results.append(result)
        all_stats.append(stats)

        if max_iter is not None:
            if len(results) == max_iter:
                break

    success = np.array([r.success for r in results])
    num_inliers = np.array([r.num_inliers for r in results])
    ratios = np.array([r.inlier_ratio for r in results])

    metrics = {
        'success': np.mean(success),
        'inliers': np.mean(num_inliers[success]),
        'inlier_ratios': np.mean(ratios[success]),
        'failure': np.arange(len(success))[np.logical_not(success)],
    }
    metrics = {k: v.tolist() for k, v in metrics.items()}
    metrics['all_stats'] = all_stats
    return metrics, results
