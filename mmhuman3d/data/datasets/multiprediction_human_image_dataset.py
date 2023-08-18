import json
import os
import os.path
import pickle as pkl
from abc import ABCMeta
from collections import OrderedDict
from typing import Any, List, Optional, Union
from yacs.config import CfgNode as CN
from functools import partial

import mmcv
import numpy as np
from scipy.sparse import csr_matrix
import torch
from torch.utils.data import Dataset
import math

from mmhuman3d.core.conventions import constants

from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_num,
)
from mmhuman3d.core.evaluation import (
    keypoint_3d_auc,
    keypoint_3d_pck,
    keypoint_mpjpe,
    vertice_pve,
    keypoint_2d_pckh,
)

from mmhuman3d.utils.geometry import (
    convert_weak_perspective_to_perspective,
)

from mmhuman3d.utils.neural_renderer import (
    get_cameras)

from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.models.builder import build_body_model
from mmhuman3d.utils.keypoint_utils import remove_outside_batch
from .base_dataset import BaseDataset
from .builder import DATASETS, build_dataset

def split_dict(d, n):
    """
    split dict into n sub dict
    """
    sub_ds = []
    for i in range(n):
        sub_d = {}
        for k in d.keys():
            if isinstance(d[k], (list, tuple, np.ndarray)):
                sub_d[k] = d[k][i::n]
        sub_ds.append(sub_d)
    return sub_ds


@DATASETS.register_module()
class MultiPredictionHumanImageDataset(Dataset):
    """Occluded Human Image Dataset. A wrapper for HumanImageDataset to 
    generate sliding window occlusion.

    Args:
        orig_cfg (dict): the config of the wrapped dataset
        occ_size (int): size of occlusion patch
        occ_stride (int): stride of sliding window
        textures_file (str): path to the texture list file for occlusion
        test_mode (bool, optional): in train mode or test mode.
            Default: False.
    """
    # metric
    ALLOWED_METRICS = {
        'mpjpe', 'pa-mpjpe', 'pve', '3dpck', 'pa-3dpck', '3dauc', 'pa-3dauc', 'pckh'
    }

    OCC_INFO_KEYS = ['occ_idx', 'occ_size', 'occ_stride', 'texture_file', 'texture_crop_tl']
    def __init__(self,
                 orig_cfg, 
                 n_predictions=1,
                 textures_file=None,
                 test_mode=False,
                 hparams=None,
                 ):
        if test_mode:
            orig_cfg.test_mode = True
        self.dataset = build_dataset(orig_cfg)
        # self.occ_size = occ_size
        # self.occ_stride = occ_stride
        # self.n_grid = int(math.ceil((constants.IMG_RES - self.occ_size) / self.occ_stride + 1))
        # self.n_grid_2 = self.n_grid ** 2

        self.n_predictions = n_predictions
        self.num_data = len(self.dataset) * self.n_predictions
        if hparams is not None:
            self.hparams = CN.load_cfg(str(hparams))
        else:
            self.hparams = None
        if textures_file is not None:
            self.texture_files = [x.strip() for x in open(textures_file).readlines()]
        else:
            self.texture_files = None

        if hasattr(self.hparams, 'pred_initialization'):
            if self.hparams.pred_initialization.endswith('.json'):
                self.pred_initialization = json.load(open(self.hparams.pred_initialization))
            elif self.hparams.pred_initialization.endswith('.pkl'):
                self.pred_initialization = pkl.load(open(self.hparams.pred_initialization, 'rb'))
            else:
                self.pred_initialization = None
        else:
            self.pred_initialization = None

        if hasattr(self.hparams, 'saved_partseg') and self.hparams.saved_partseg is not None:
            if self.hparams.saved_partseg.endswith('.pkl'):
                self.saved_partseg = pkl.load(open(self.hparams.saved_partseg, 'rb'))
            else:
                self.saved_partseg = self.hparams.saved_partseg
        else:
            self.saved_partseg = None

        self.occ_info = json.load(open(self.hparams.occ_info_file)) \
                if hasattr(self.hparams, 'occ_info_file') else None
                

    def __getitem__(self, idx):
        idx_org = idx // self.n_predictions
        info = self.dataset.prepare_raw_data(self.dataset.get_indices()[idx_org])
        info['idx'] = idx
            
        return self.dataset.pipeline(info)

    def __len__(self):
        return self.num_data

    def evaluate(self,
                 outputs: list,
                 res_folder: str,
                 metric: Optional[Union[str, List[str]]] = 'pa-mpjpe',
                 **kwargs: dict):
        """Evaluate 3D keypoint results.

        Args:
            outputs (list): results from model inference.
            res_folder (str): path to store results.
            metric (Optional[Union[str, List(str)]]):
                the type of metric. Default: 'pa-mpjpe'
            kwargs (dict): other arguments.
        Returns:
            dict:
                A dict of all evaluation results.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        for metric in metrics:
            if metric not in self.ALLOWED_METRICS:
                raise KeyError(f'metric {metric} is not supported')

        if 'eval_saved_results' in kwargs and kwargs['eval_saved_results']:
            res = outputs
        else:
            res_file = os.path.join(res_folder, 'result_keypoints.json')
            # for keeping correctness during multi-gpu test, we sort all results

            res_dict = {}
            for out in outputs:
                target_id = out['image_idx']
                batch_size = len(out['keypoints_3d'])
                for i in range(batch_size):
                    occ_size = out['meta_info']['occ_size'][i]
                    occ_level_idx = self.map_occ_level_idx.get(occ_size)
                    idx = int(target_id[i]) * self.n_occ_levels + occ_level_idx
                    res_dict[idx] = dict(
                        keypoints=out['keypoints_3d'][i],
                        poses=out['smpl_pose'][i],
                        betas=out['smpl_beta'][i],
                        cameras=out['camera'][i],
                        pred_segm_mask=out['pred_segm_mask'][i] if 'pred_segm_mask' in out else None,
                        keypoints2d=out['meta_info']['keypoints_2d'][i]
                    )
                    for key in self.OCC_INFO_KEYS:
                        if key in out['meta_info']:
                            res_dict[idx][key] = out['meta_info'][key][i]

            keypoints, poses, betas, cameras, keypoints2d, occ_info = [], [], [], [], [], {}
            pred_segm_mask = []
            for j in range(self.num_data):
                i = self.dataset.get_indices()[j // self.n_occ_levels] * self.n_occ_levels + j % self.n_occ_levels
                keypoints.append(res_dict[i]['keypoints'])
                poses.append(res_dict[i]['poses'])
                betas.append(res_dict[i]['betas'])
                cameras.append(res_dict[i]['cameras'])
                pred_segm_mask.append(res_dict[i]['pred_segm_mask'])
                keypoints2d.append(res_dict[i]['keypoints2d'])

                for key in self.OCC_INFO_KEYS:
                    if key in res_dict[i]:
                        occ_info.setdefault(key, []).append(res_dict[i][key])

            # Use a lot of memory when dataset is large; now save during forward
            # if 'save_partseg' in kwargs and kwargs['save_partseg']:
            #     # pkl.dump(pred_segm_mask, open(os.path.join(res_folder, 'result_pred_segm_mask.pkl'), 'wb'))
            #     os.makedirs(os.path.join(res_folder, 'result_pred_segm_mask'), exist_ok=True)
            #     for i, mask in enumerate(pred_segm_mask):
            #         dst = os.path.join(res_folder, 'result_pred_segm_mask', f'{i}.npy')
            #         np.save(dst, mask)

            # keypoints: (B, 17, 3) predicted 3D keypoints
            # keypoints2d: (B, 17, 2) gt 2D keypoints
            res = dict(keypoints=keypoints, poses=poses, betas=betas, cameras=cameras, 
                        keypoints2d=keypoints2d, occ_info=occ_info)
            mmcv.dump(res, res_file)
            mmcv.dump(occ_info, os.path.join(res_folder, f'result_occ_info.json'))

        name_value_tuples = []
        for _metric in metrics:
            if _metric == 'mpjpe':
                _nv_tuples, pred_info_argmin, error_elementwise = self._report_mpjpe(res)
            elif _metric == 'pa-mpjpe':
                _nv_tuples, pred_info_argmin, error_elementwise = self._report_mpjpe(res, metric='pa-mpjpe')
            elif _metric == '3dpck':
                _nv_tuples = self._report_3d_pck(res)
            elif _metric == 'pa-3dpck':
                _nv_tuples = self._report_3d_pck(res, metric='pa-3dpck')
            elif _metric == '3dauc':
                _nv_tuples = self._report_3d_auc(res)
            elif _metric == 'pa-3dauc':
                _nv_tuples = self._report_3d_auc(res, metric='pa-3dauc')
            elif _metric == 'pve':
                _nv_tuples = self._report_pve(res)
            elif _metric == 'pckh':
                _nv_tuples, pred_info_argmin, error_elementwise = self._report_2d_pckh(res)
            else:
                raise NotImplementedError
            name_value_tuples.extend(_nv_tuples)
            mmcv.dump(pred_info_argmin, os.path.join(res_folder, f'result_pred_info_{_metric}.json'))
            np.save(os.path.join(res_folder, f'{_metric}_elementwise.npy'), error_elementwise)

        name_value = OrderedDict(name_value_tuples)
        return name_value

    def _report_mpjpe(self, res_file, metric='mpjpe'):
        """Calculate mean per joint position error (MPJPE) or its variants PA-
        MPJPE.

        Report mean per joint position error (MPJPE) and mean per joint
        position error after rigid alignment (PA-MPJPE)
        """
        pred_keypoints3d, gt_keypoints3d, gt_keypoints3d_mask = \
                    self._parse_result(res_file, partial(self.dataset._parse_result, mode='keypoint'))

        err_name = metric.upper()
        if metric == 'mpjpe':
            alignment = 'none'
        elif metric == 'pa-mpjpe':
            alignment = 'procrustes'
        else:
            raise ValueError(f'Invalid metric: {metric}')

        error, error_elementwise = keypoint_mpjpe(pred_keypoints3d, gt_keypoints3d,
                               gt_keypoints3d_mask, alignment, return_elementwise=True)

        pred_idx_argmin = np.argmin(error_elementwise.reshape(-1, self.n_predictions), axis=1)
        pred_info_argmin = dict(pred_idx_argmin=pred_idx_argmin)

        info_str = [(err_name, error)]

        return info_str, pred_info_argmin, error_elementwise

    def _report_3d_pck(self, res_file, metric='3dpck'):
        """Calculate Percentage of Correct Keypoints (3DPCK) w. or w/o
        Procrustes alignment.
        Args:
            keypoint_results (list): Keypoint predictions. See
                'Body3DMpiInf3dhpDataset.evaluate' for details.
            metric (str): Specify mpjpe variants. Supported options are:
                - ``'3dpck'``: Standard 3DPCK.
                - ``'pa-3dpck'``:
                    3DPCK after aligning prediction to groundtruth
                    via a rigid transformation (scale, rotation and
                    translation).
        """

        pred_keypoints3d, gt_keypoints3d, gt_keypoints3d_mask = \
            self._parse_result(res_file)

        err_name = metric.upper()
        if metric == '3dpck':
            alignment = 'none'
        elif metric == 'pa-3dpck':
            alignment = 'procrustes'
        else:
            raise ValueError(f'Invalid metric: {metric}')

        error = keypoint_3d_pck(pred_keypoints3d, gt_keypoints3d,
                                gt_keypoints3d_mask, alignment)
        name_value_tuples = [(err_name, error)]

        return name_value_tuples

    def _report_3d_auc(self, res_file, metric='3dauc'):
        """Calculate the Area Under the Curve (AUC) computed for a range of
        3DPCK thresholds.
        Args:
            keypoint_results (list): Keypoint predictions. See
                'Body3DMpiInf3dhpDataset.evaluate' for details.
            metric (str): Specify mpjpe variants. Supported options are:
                - ``'3dauc'``: Standard 3DAUC.
                - ``'pa-3dauc'``: 3DAUC after aligning prediction to
                    groundtruth via a rigid transformation (scale, rotation and
                    translation).
        """

        pred_keypoints3d, gt_keypoints3d, gt_keypoints3d_mask = \
            self._parse_result(res_file)

        err_name = metric.upper()
        if metric == '3dauc':
            alignment = 'none'
        elif metric == 'pa-3dauc':
            alignment = 'procrustes'
        else:
            raise ValueError(f'Invalid metric: {metric}')

        error = keypoint_3d_auc(pred_keypoints3d, gt_keypoints3d,
                                gt_keypoints3d_mask, alignment)
        name_value_tuples = [(err_name, error)]

        return name_value_tuples

    def _report_pve(self, res_file):
        """Calculate per vertex error."""
        pred_verts, gt_verts, _ = \
            self._parse_result(res_file, mode='vertice')
        error = vertice_pve(pred_verts, gt_verts)
        return [('PVE', error)]

    def _report_2d_pckh(self, res_file):
        """Calculate PCKh metric. """
        pred_keypoints2d, gt_keypoints2d, gt_keypoints2d_mask = self._parse_result(res_file, self.dataset._parse_result_2d)

        if self.dataset.dataset_name in ['pw3d', '3doh50k', 'h36m']:
            thr_dist_ids = (12, 13)
            thr = 0.5
        pckh, pckh_elementwise = keypoint_2d_pckh(pred_keypoints2d, gt_keypoints2d, gt_keypoints2d_mask, 
                        thr=thr, thr_dist_ids=thr_dist_ids, return_elementwise=True)
        pred_idx_argmin = np.argmax(pckh_elementwise.reshape(-1, self.n_predictions), axis=1)
        pred_info_argmin = dict(pred_idx_argmin=pred_idx_argmin)
        return [('PCKh', pckh)], pred_info_argmin, pckh_elementwise

    def _merge(self, xs):
        """
        merge lists
        """
        merged = []
        for x_i in zip(*xs):
            merged.extend(x_i)
        return np.array(merged)

    def _parse_result(self, res_file, parse_fn):
        pred_keypoints = []
        gt_keypoints = []
        gt_keypoints_mask = []
        res_files = split_dict(res_file, self.n_predictions)
        for pred_idx in range(self.n_predictions):
            res_file = res_files[pred_idx]
            pred_keypoints_i, gt_keypoints_i, gt_keypoints_mask_i = parse_fn(res_file)
            pred_keypoints.append(pred_keypoints_i)
            gt_keypoints.append(gt_keypoints_i)
            gt_keypoints_mask.append(gt_keypoints_mask_i)
        pred_keypoints = self._merge(pred_keypoints) #np.concatenate(pred_keypoints, axis=0)
        gt_keypoints = self._merge(gt_keypoints) #np.concatenate(gt_keypoints, axis=0)
        gt_keypoints_mask = self._merge(gt_keypoints_mask) #np.concatenate(gt_keypoints_mask, axis=0)
        return pred_keypoints, gt_keypoints, gt_keypoints_mask