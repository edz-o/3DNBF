import json
import os
import os.path
import pickle as pkl
from abc import ABCMeta
from collections import OrderedDict
from typing import Any, List, Optional, Union
from yacs.config import CfgNode as CN
import ipdb
import mmcv
import cv2
import numpy as np
from scipy.sparse import csr_matrix
import torch

try:
    from pytorch3d.renderer.cameras import get_ndc_to_screen_transform
except:
    pass

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
    keypoint_2d_pck,
)

from mmhuman3d.utils.geometry import (
    convert_weak_perspective_to_perspective,
)

from mmhuman3d.utils.neural_renderer import get_cameras

from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.models.builder import build_body_model
from mmhuman3d.utils.keypoint_utils import remove_outside_batch
from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class HumanImageDataset(BaseDataset, metaclass=ABCMeta):
    """Human Image Dataset.

    Args:
        data_prefix (str): the prefix of data path.
        pipeline (list): a list of dict, where each element represents
            a operation defined in `mmhuman3d.datasets.pipelines`.
        dataset_name (str | None): the name of dataset. It is used to
            identify the type of evaluation metric. Default: None.
        body_model (dict | None, optional): the config for body model,
            which will be used to generate meshes and keypoints.
            Default: None.
        ann_file (str | None, optional): the annotation file. When ann_file
            is str, the subclass is expected to read from the ann_file.
            When ann_file is None, the subclass is expected to read
            according to data_prefix.
        convention (str, optional): keypoints convention. Keypoints will be
            converted from "human_data" to the given one.
            Default: "human_data"
        test_mode (bool, optional): in train mode or test mode.
            Default: False.
    """

    # metric
    ALLOWED_METRICS = {
        'mpjpe',
        'pa-mpjpe',
        'pve',
        '3dpck',
        'pa-3dpck',
        '3dauc',
        'pa-3dauc',
        'pckh',
        'pck_vert',
        'pck',
    }

    def __init__(
        self,
        data_prefix: str,
        pipeline: list,
        dataset_name: str,
        body_model: Optional[Union[dict, None]] = None,
        ann_file: Optional[Union[str, None]] = None,
        convention: Optional[str] = 'human_data',
        test_mode: Optional[bool] = False,
        hparams: Optional[Union[dict, None]] = None,
    ):
        self.convention = convention
        self.num_keypoints = get_keypoint_num(convention)
        if hparams is not None:
            self.hparams = CN.load_cfg(str(hparams))
        else:
            self.hparams = CN.load_cfg(str({}))  # None
        super(HumanImageDataset, self).__init__(
            data_prefix, pipeline, ann_file, test_mode, dataset_name
        )
        if body_model is not None:
            self.body_model = build_body_model(body_model)
        else:
            self.body_model = None
        self.sample_indices(self.hparams.get('test_indices', []))

    def __len__(self):
        return len(self._indices)

    def get_annotation_file(self):
        """Get path of the annotation file."""
        ann_prefix = os.path.join(self.data_prefix, 'preprocessed_datasets')
        self.ann_file = os.path.join(ann_prefix, self.ann_file)

    def load_annotations(self):
        """Load annotation from the annotation file.

        Here we simply use :obj:`HumanData` to parse the annotation.
        """
        self.get_annotation_file()
        # change keypoint from 'human_data' to the given convention
        self.human_data = HumanData.fromfile(self.ann_file)
        if self.human_data.check_keypoints_compressed():
            self.human_data.decompress_keypoints()
        if 'keypoints3d' in self.human_data:
            keypoints3d = self.human_data['keypoints3d']
            assert 'keypoints3d_mask' in self.human_data
            keypoints3d_mask = self.human_data['keypoints3d_mask']
            keypoints3d, keypoints3d_mask = convert_kps(
                keypoints3d,
                src='human_data',
                dst=self.convention,
                mask=keypoints3d_mask,
            )
            self.human_data.__setitem__('keypoints3d', keypoints3d)
            self.human_data.__setitem__('keypoints3d_mask', keypoints3d_mask)
        if 'keypoints2d' in self.human_data:
            keypoints2d = self.human_data['keypoints2d']
            assert 'keypoints2d_mask' in self.human_data
            keypoints2d_mask = self.human_data['keypoints2d_mask']
            keypoints2d, keypoints2d_mask = convert_kps(
                keypoints2d,
                src='human_data',
                dst=self.convention,
                mask=keypoints2d_mask,
            )
            self.human_data.__setitem__('keypoints2d', keypoints2d)
            self.human_data.__setitem__('keypoints2d_mask', keypoints2d_mask)
        self.num_data = self.human_data.data_len

        if (
            hasattr(self.hparams, 'saved_partseg')
            and self.hparams.saved_partseg is not None
        ):
            if self.hparams.saved_partseg.endswith('.pkl'):
                self.saved_partseg = pkl.load(open(self.hparams.saved_partseg, 'rb'))
            else:
                self.saved_partseg = self.hparams.saved_partseg
        else:
            self.saved_partseg = None

        self.occ_info = (
            json.load(open(self.hparams.occ_info_file))
            if (self.hparams.get('occ_info_file', None) is not None) and \
                (self.hparams.get('occ_info_file', None) != 'None')
            else None
        )

        if hasattr(self.hparams, 'pred_initialization'):
            if self.hparams.pred_initialization.endswith('.json'):
                self.pred_initialization = json.load(
                    open(self.hparams.pred_initialization)
                )
            elif self.hparams.pred_initialization.endswith('.pkl'):
                self.pred_initialization = pkl.load(
                    open(self.hparams.pred_initialization, 'rb')
                )
            else:
                self.pred_initialization = None
        else:
            self.pred_initialization = None

    def sample_indices(self, indices=None):
        if indices:
            if len(indices) == 3:
                self._indices = np.arange(*indices)
            else:
                self._indices = indices
        else:
            self._indices = np.arange(self.num_data)

    def get_indices(self):
        return self._indices

    def prepare_raw_data(self, idx: int):
        """Get item from self.human_data."""
        info = {}
        info['idx'] = int(idx)
        info['img_prefix'] = None
        image_path = self.human_data['image_path'][idx]
        # print(self.human_data['image_path'])
        info['image_path'] = os.path.join(
            self.data_prefix, 'datasets', self.dataset_name, image_path
        )
        if hasattr(self.hparams, 'load_mask') and self.hparams.load_mask:
            info['mask_prefix'] = None
            # from humor mask
            mask_path = image_path.replace('/raw_frames/', '/masks/')
            info['mask_path'] = os.path.join(
                self.data_prefix, 'datasets', self.dataset_name, mask_path
            )
            info['mask'] = 255 - cv2.imread(info['mask_path'], cv2.IMREAD_GRAYSCALE)
        if image_path.endswith('smc'):
            device, device_id, frame_id = self.human_data['image_id'][idx]
            info['image_id'] = (device, int(device_id), int(frame_id))
        info['dataset_name'] = self.dataset_name
        info['sample_idx'] = int(idx)
        if 'bbox_xywh' in self.human_data:
            info['bbox_xywh'] = self.human_data['bbox_xywh'][idx]
            x, y, w, h, s = info['bbox_xywh']
            cx = x + w / 2
            cy = y + h / 2
            w = h = max(w, h)
            info['center'] = np.array([cx, cy])
            info['scale'] = np.array([w, h])
        else:
            info['bbox_xywh'] = np.zeros((5))
            info['center'] = np.zeros((2))
            info['scale'] = np.zeros((2))

        # in later modules, we will check validity of each keypoint by
        # its confidence. Therefore, we do not need the mask of keypoints.

        if 'keypoints2d' in self.human_data:
            info['keypoints2d'] = self.human_data['keypoints2d'][idx]
        else:
            info['keypoints2d'] = np.zeros((self.num_keypoints, 3))

        if 'keypoints3d' in self.human_data:
            info['has_kp3d'] = 1
            info['keypoints3d'] = self.human_data['keypoints3d'][idx]
        else:
            info['has_kp3d'] = 0
            info['keypoints3d'] = np.zeros((self.num_keypoints, 4))

        if 'smpl' in self.human_data:
            smpl_dict = self.human_data['smpl']
        else:
            smpl_dict = {}

        if 'meta' in self.human_data:
            if self.human_data['meta']['gender'][idx] == 'm':
                info['gender'] = 0
            else:
                info['gender'] = 1

        if 'smpl' in self.human_data and smpl_dict:
            if 'has_smpl' in self.human_data:
                info['has_smpl'] = int(self.human_data['has_smpl'][idx])
            else:
                info['has_smpl'] = 1
        else:
            info['has_smpl'] = 0
        if 'body_pose' in smpl_dict:
            info['smpl_body_pose'] = smpl_dict['body_pose'][idx]
        else:
            info['smpl_body_pose'] = np.zeros((23, 3))

        if 'global_orient' in smpl_dict:
            info['smpl_global_orient'] = smpl_dict['global_orient'][idx]
        else:
            info['smpl_global_orient'] = np.zeros((3))

        if 'betas' in smpl_dict:
            info['smpl_betas'] = smpl_dict['betas'][idx]
        else:
            info['smpl_betas'] = np.zeros((10))

        if 'transl' in smpl_dict:
            info['smpl_transl'] = smpl_dict['transl'][idx]
        else:
            info['smpl_transl'] = np.zeros((3))

        if 'misc' in self.human_data:
            # Backward compatibility
            info['poses_init'] = (
                self.human_data['misc']['poses_init'][idx]
                if 'poses_init' in self.human_data['misc']
                else np.repeat(np.eye(3)[None], 24, axis=0)
            )
            info['betas_init'] = (
                self.human_data['misc']['betas_init'][idx]
                if 'betas_init' in self.human_data['misc']
                else np.zeros(10)
            )
            info['cameras_init'] = (
                self.human_data['misc']['cameras_init'][idx]
                if 'cameras_init' in self.human_data['misc']
                else np.array([1.0, 0.0, 0.0])
            )
            info['coke_features'] = (
                self.human_data['misc']['coke_features'][idx]
                if 'coke_features' in self.human_data['misc']
                else None
            )  # might have issues with None
            info['keypoints3d_init'] = (
                self.human_data['misc']['keypoints3d_init'][idx]
                if 'keypoints3d_init' in self.human_data['misc']
                else np.zeros((self.num_keypoints, 3))
            )

        if self.pred_initialization is not None:
            info['poses_init'] = (
                np.array(self.pred_initialization['poses'])[idx]
                if 'poses' in self.pred_initialization
                else np.concatenate(
                    (
                        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])[None],
                        np.repeat(np.eye(3)[None], 23, axis=0),
                    ),
                    axis=0,
                )
            )
            info['betas_init'] = (
                np.array(self.pred_initialization['betas'])[idx]
                if 'betas' in self.pred_initialization
                else np.zeros(10)
            )
            info['cameras_init'] = (
                np.array(self.pred_initialization['cameras'])[idx]
                if 'cameras' in self.pred_initialization
                else np.array([1.0, 0.0, 0.0])
            )
            info['keypoints3d_init'] = (
                np.array(self.pred_initialization['keypoints'])[idx]
                if 'keypoints' in self.pred_initialization
                else np.zeros((self.num_keypoints, 3))
            )

        if self.saved_partseg is not None:
            if isinstance(self.saved_partseg, str):
                pred_segm_mask = np.load(os.path.join(self.saved_partseg, f'{idx}.npy'))
            else:
                pred_segm_mask = self.saved_partseg[idx]
            info['pred_segm_mask'] = pred_segm_mask  # self.saved_partseg[idx]

        if self.occ_info is not None:
            for k, v in self.occ_info.items():
                info[k] = v[idx]
        return info

    def prepare_data(self, idx: int):
        """Generate and transform data."""
        info = self.prepare_raw_data(self.get_indices()[idx])
        return self.pipeline(info)

    def evaluate(
        self,
        outputs: list,
        res_folder: str,
        metric: Optional[Union[str, List[str]]] = 'pa-mpjpe',
        **kwargs: dict,
    ):
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
                    res_dict[int(target_id[i])] = dict(
                        keypoints=out['keypoints_3d'][i],
                        poses=out['smpl_pose'][i],
                        betas=out['smpl_beta'][i],
                        cameras=out['camera'][i],
                        pred_segm_mask=out['pred_segm_mask'][i]
                        if 'pred_segm_mask' in out
                        else None,
                        keypoints2d=out['meta_info']['keypoints_2d'][i]
                        if 'meta_info' in out
                        else None,
                        verts2d=out['vertices2d_det'][i]
                        if 'vertices2d_det' in out
                        else None,
                    )

            keypoints, poses, betas, cameras, keypoints2d = [], [], [], [], []
            pred_segm_mask = []
            verts2d = []
            for i in self.get_indices():
                keypoints.append(res_dict[i]['keypoints'])
                poses.append(res_dict[i]['poses'])
                betas.append(res_dict[i]['betas'])
                cameras.append(res_dict[i]['cameras'])
                pred_segm_mask.append(res_dict[i]['pred_segm_mask'])
                keypoints2d.append(res_dict[i]['keypoints2d'])
                verts2d.append(res_dict[i]['verts2d'])

            if 'save_partseg' in kwargs and kwargs['save_partseg']:
                os.makedirs(
                    os.path.join(res_folder, 'result_pred_segm_mask'), exist_ok=True
                )
                for i, mask in enumerate(pred_segm_mask):
                    if mask is not None:
                        dst = os.path.join(
                            res_folder, 'result_pred_segm_mask', f'{i}.npy'
                        )
                        np.save(dst, mask)

            res = dict(keypoints=keypoints, poses=poses, betas=betas, cameras=cameras)
            if 'meta_info' in outputs[0] and 'keypoints_2d' in outputs[0]['meta_info']:
                res.update(dict(keypoints2d=keypoints2d))
            res['verts2d'] = verts2d
            mmcv.dump(res, res_file)

        name_value_tuples = []
        for _metric in metrics:
            if _metric == 'mpjpe':
                _nv_tuples, err_ew = self._report_mpjpe(res, return_elementwise=True)
            elif _metric == 'pa-mpjpe':
                _nv_tuples, err_ew = self._report_mpjpe(
                    res, metric='pa-mpjpe', return_elementwise=True
                )
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
                _nv_tuples, err_ew = self._report_2d_pckh(res, return_elementwise=True)
            elif _metric == 'pck':
                _nv_tuples, err_ew = self._report_2d_pck(res, return_elementwise=True)
            elif _metric == 'pck_vert':
                _nv_tuples, err_ew = self._report_2d_vert_pck(
                    res, thr=0.10, return_elementwise=True
                )
            else:
                raise NotImplementedError
            name_value_tuples.extend(_nv_tuples)
            np.save(os.path.join(res_folder, f'{_metric}_elementwise.npy'), err_ew)

        name_value = OrderedDict(name_value_tuples)
        return name_value

    @staticmethod
    def _write_keypoint_results(keypoints: Any, res_file: str):
        """Write results into a json file."""

        with open(res_file, 'w') as f:
            json.dump(keypoints, f, sort_keys=True, indent=4)

    def _parse_result(self, res, mode='keypoint'):
        """Parse results."""

        if mode == 'vertice':
            # gt
            gt_beta, gt_pose, gt_global_orient, gender = [], [], [], []
            gt_smpl_dict = self.human_data['smpl']
            for idx in self.get_indices():
                gt_beta.append(gt_smpl_dict['betas'][idx])
                gt_pose.append(gt_smpl_dict['body_pose'][idx])
                gt_global_orient.append(gt_smpl_dict['global_orient'][idx])
                if self.human_data['meta']['gender'][idx] == 'm':
                    gender.append(0)
                else:
                    gender.append(1)
            gt_beta = torch.FloatTensor(gt_beta)
            gt_pose = torch.FloatTensor(gt_pose).view(-1, 69)
            gt_global_orient = torch.FloatTensor(gt_global_orient)
            gender = torch.Tensor(gender)
            gt_output = self.body_model(
                betas=gt_beta,
                body_pose=gt_pose,
                global_orient=gt_global_orient,
                gender=gender,
            )
            gt_vertices = gt_output['vertices'].detach().cpu().numpy() * 1000.0
            gt_mask = np.ones(gt_vertices.shape[:-1])
            # pred
            pred_pose = torch.FloatTensor(res['poses'])
            pred_beta = torch.FloatTensor(res['betas'])
            pred_output = self.body_model(
                betas=pred_beta[:],
                body_pose=pred_pose[:, 1:][:],
                global_orient=pred_pose[:, 0].unsqueeze(1)[:],
                pose2rot=False,
                gender=gender,
            )
            pred_vertices = pred_output['vertices'].detach().cpu().numpy() * 1000.0

            assert len(pred_vertices) == len(self)

            return pred_vertices, gt_vertices, gt_mask
        elif mode == 'keypoint':
            pred_keypoints3d = res['keypoints'][:]
            # assert len(pred_keypoints3d) == self.num_data
            # (B, 17, 3)
            pred_keypoints3d = np.array(pred_keypoints3d)

            if self.dataset_name in ['pw3d', 'briar_synthetic_part_allShape', 'coco']:
                betas = []
                body_pose = []
                global_orient = []
                gender = []
                smpl_dict = self.human_data['smpl']
                # TODO remove for loop
                for idx in self.get_indices():
                    betas.append(smpl_dict['betas'][idx])
                    body_pose.append(smpl_dict['body_pose'][idx])
                    global_orient.append(smpl_dict['global_orient'][idx])
                    if self.dataset_name == 'coco':
                        gender.append(-1)
                    elif self.human_data['meta']['gender'][idx] == 'm':
                        gender.append(0)
                    else:
                        gender.append(1)
                betas = torch.FloatTensor(betas)
                body_pose = torch.FloatTensor(body_pose).view(-1, 69)
                global_orient = torch.FloatTensor(global_orient)
                gender = torch.Tensor(gender)

                chunk_size = 128
                i_chunk = 0
                gt_keypoints3d = []
                while i_chunk * chunk_size < len(self):
                    start = i_chunk * chunk_size
                    end = min((i_chunk + 1) * chunk_size, len(self))
                    gt_output_chunk = self.body_model(
                        betas=betas[start:end],
                        body_pose=body_pose[start:end],
                        global_orient=global_orient[start:end],
                        gender=gender[start:end],
                    )

                    gt_keypoints3d.append(
                        gt_output_chunk['joints'].detach().cpu().numpy()
                    )
                    i_chunk += 1
                gt_keypoints3d = np.concatenate(gt_keypoints3d, axis=0)

                eval_visible_joints = self.hparams.get('eval_visible_joints', False)

                if eval_visible_joints:
                    keypoints2d_gt = self.human_data['keypoints2d'][self.get_indices()]
                    if 'keypoints2d' in res:
                        keypoints2d_vis = np.array(res['keypoints2d'])[:, :, 2]  # DEBUG
                        keypoints2d_gt[:, :, 2] = keypoints2d_vis[:]

                    bbox_xywh = self.human_data['bbox_xywh'][self.get_indices()]
                    tl = (
                        bbox_xywh[:, :2]
                        + 0.5 * bbox_xywh[:, 2:4]
                        - 0.5 * np.max(bbox_xywh[:, 2:4], axis=1, keepdims=True)
                    )
                    bbox_xywh_square = np.concatenate(
                        (tl, np.maximum(bbox_xywh[:, 2:4], bbox_xywh[:, 3:1:-1])),
                        axis=1,
                    )
                    keypoints2d_gt = remove_outside_batch(
                        keypoints2d_gt, bbox_xywh_square
                    )

                    keypoints2d, keypoints2d_mask = convert_kps(
                        keypoints2d_gt,
                        src=self.convention,  #'smpl_49',
                        dst='h36m',
                        mask=self.human_data['keypoints2d_mask'],
                    )
                    gt_keypoints3d_mask = keypoints2d[:, :, 2]
                else:
                    # All visible
                    gt_keypoints3d_mask = np.ones(
                        (len(pred_keypoints3d), gt_keypoints3d.shape[1])
                    )
            elif self.dataset_name in ['3doh50k']:
                betas = []
                body_pose = []
                global_orient = []
                gender = []
                smpl_dict = self.human_data['smpl']
                for idx in self.get_indices():
                    betas.append(smpl_dict['betas'][idx])
                    body_pose.append(smpl_dict['body_pose'][idx])
                    global_orient.append(smpl_dict['global_orient'][idx])

                betas = torch.FloatTensor(betas)
                body_pose = torch.FloatTensor(body_pose).view(-1, 69)
                global_orient = torch.FloatTensor(global_orient)

                chunk_size = 128
                i_chunk = 0
                gt_keypoints3d = []
                while i_chunk * chunk_size < len(self):
                    start = i_chunk * chunk_size
                    end = min((i_chunk + 1) * chunk_size, len(self))
                    gt_output_chunk = self.body_model(
                        betas=betas[start:end],
                        body_pose=body_pose[start:end],
                        global_orient=global_orient[start:end],
                    )

                    gt_keypoints3d.append(
                        gt_output_chunk['joints'].detach().cpu().numpy()
                    )
                    i_chunk += 1
                gt_keypoints3d = np.concatenate(gt_keypoints3d, axis=0)

                eval_visible_joints = (
                    False if self.hparams is None else self.hparams.eval_visible_joints
                )
                if eval_visible_joints:
                    # self.human_data['keypoints2d'][:,:,2] = 1
                    bbox_xywh = self.human_data['bbox_xywh']
                    tl = (
                        bbox_xywh[:, :2]
                        + 0.5 * bbox_xywh[:, 2:4]
                        - 0.5 * np.max(bbox_xywh[:, 2:4], axis=1, keepdims=True)
                    )
                    bbox_xywh_square = np.concatenate(
                        (tl, np.maximum(bbox_xywh[:, 2:4], bbox_xywh[:, 3:1:-1])),
                        axis=1,
                    )
                    self.human_data['keypoints2d'] = remove_outside_batch(
                        self.human_data['keypoints2d'], bbox_xywh_square
                    )

                    keypoints2d, keypoints2d_mask = convert_kps(
                        self.human_data['keypoints2d'][self.get_indices()],
                        src=self.convention,  #'smpl_49',
                        dst='h36m',
                        mask=self.human_data['keypoints2d_mask'],
                    )
                    gt_keypoints3d_mask = keypoints2d[:, :, 2]

                else:
                    # All visible
                    gt_keypoints3d_mask = np.ones((len(pred_keypoints3d), 24))
            elif self.dataset_name in ['h36m', 'humman', 'mpi_inf_3dhp']:
                # mpi_inf_3dhp
                gt_keypoints3d, _ = convert_kps(
                    self.human_data['keypoints3d'][self.get_indices(), :, :3],
                    src=self.convention,  #'smpl_49',
                    dst='h36m',
                )
                gt_keypoints3d_mask = np.ones((len(pred_keypoints3d), 17))
            elif self.dataset_name in ['mpi_inf_3dhp']:
                # mpi_inf_3dhp train
                gt_keypoints3d, _ = convert_kps(
                    self.human_data['keypoints3d'][self.get_indices(), :, :],
                    src=self.convention,  #'smpl_49',
                    dst='smpl_49',
                )
                pred_keypoints3d, _ = convert_kps(
                    pred_keypoints3d,
                    src='h36m',
                    dst='smpl_49',
                )

                gt_keypoints3d_mask = gt_keypoints3d[:, :, 3].astype(np.bool)
                gt_keypoints3d = gt_keypoints3d[:, :, :3]

            else:
                raise NotImplementedError()

            # SMPL_49 only!
            if gt_keypoints3d.shape[1] == 49:
                assert pred_keypoints3d.shape[1] == 49

                gt_keypoints3d = gt_keypoints3d[:, 25:, :]
                pred_keypoints3d = pred_keypoints3d[:, 25:, :]
                gt_keypoints3d_mask = gt_keypoints3d_mask[:, 25:]

                joint_mapper = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]
                gt_keypoints3d = gt_keypoints3d[:, joint_mapper, :]
                pred_keypoints3d = pred_keypoints3d[:, joint_mapper, :]

                # we only evaluate on 14 lsp joints
                pred_pelvis = (pred_keypoints3d[:, 2] + pred_keypoints3d[:, 3]) / 2
                gt_pelvis = (gt_keypoints3d[:, 2] + gt_keypoints3d[:, 3]) / 2

            # H36M for testing!
            elif gt_keypoints3d.shape[1] == 17:
                # assert pred_keypoints3d.shape[1] == 17
                if pred_keypoints3d.shape[1] == 49:
                    pred_keypoints3d, _ = convert_kps(
                        pred_keypoints3d, src='smpl_49', dst='h36m'
                    )

                H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
                H36M_TO_J14 = H36M_TO_J17[:14]
                joint_mapper = H36M_TO_J14

                if pred_keypoints3d.shape[1] == 17:
                    pred_pelvis = pred_keypoints3d[:, 0]
                    pred_keypoints3d = pred_keypoints3d[:, joint_mapper, :]
                else:
                    pred_pelvis = None

                gt_pelvis = gt_keypoints3d[:, 0]

                gt_keypoints3d = gt_keypoints3d[:, joint_mapper, :]
                pred_pelvis = (pred_keypoints3d[:, 2] + pred_keypoints3d[:, 3]) / 2

            # keypoint 24
            elif gt_keypoints3d.shape[1] == 24:
                assert pred_keypoints3d.shape[1] == 24

                joint_mapper = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]
                gt_keypoints3d = gt_keypoints3d[:, joint_mapper, :]
                pred_keypoints3d = pred_keypoints3d[:, joint_mapper, :]

                # we only evaluate on 14 lsp joints
                pred_pelvis = (pred_keypoints3d[:, 2] + pred_keypoints3d[:, 3]) / 2
                gt_pelvis = (gt_keypoints3d[:, 2] + gt_keypoints3d[:, 3]) / 2

            else:
                pass

            if pred_pelvis is not None:
                pred_keypoints3d = (pred_keypoints3d - pred_pelvis[:, None, :]) * 1000
            else:
                pred_keypoints3d = pred_keypoints3d * 1000
            gt_keypoints3d = (gt_keypoints3d - gt_pelvis[:, None, :]) * 1000

            gt_keypoints3d_mask = gt_keypoints3d_mask[:, joint_mapper] > 0

            return pred_keypoints3d, gt_keypoints3d, gt_keypoints3d_mask

    def calculate_vertices2d(self):
        from mmhuman3d.utils.image_utils import get_mask_and_visibility_voge
        from mmhuman3d.utils.geometry import estimate_translation
        from mmhuman3d.utils.neural_renderer_voge import build_neural_renderer_voge
        from mmhuman3d.data.data_converters.pw3d import set_up_renderer

        from pytorch3d.renderer import PerspectiveCameras

        bbox_xywh = self.human_data['bbox_xywh'][self.get_indices()]
        tl = (
            bbox_xywh[:, :2]
            + 0.5 * bbox_xywh[:, 2:4]
            - 0.5 * np.max(bbox_xywh[:, 2:4], axis=1, keepdims=True)
        )
        bbox_xywh_square = np.concatenate(
            (tl, np.maximum(bbox_xywh[:, 2:4], bbox_xywh[:, 3:1:-1])), axis=1
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        body_model = build_body_model(
            dict(
                type='SMPL',
                keypoint_src='smpl_54',
                keypoint_dst='smpl_49',
                keypoint_approximate=True,
                model_path='data/body_models/smpl',
                extra_joints_regressor='data/body_models/J_regressor_extra.npy',
            )
        ).to(device)

        RENDERER_GT = CN.load_cfg(
            str(
                dict(
                    SIGMA=0,
                    GAMMA=1e-2,
                    FACES_PER_PIXEL=1,
                    RENDER_RES=self.hparams.IMG_RES // 4,
                )
            )
        )
        neural_renderer_voge_gt = build_neural_renderer_voge(RENDERER_GT)

        betas = []
        body_pose = []
        global_orient = []
        gender = []
        cam_params = []
        smpl_dict = self.human_data['smpl']

        # TODO remove for loop
        for idx in self.get_indices():
            betas.append(smpl_dict['betas'][idx])
            body_pose.append(smpl_dict['body_pose'][idx])
            global_orient.append(smpl_dict['global_orient'][idx])
            if self.human_data['meta']['gender'][idx] == 'm':
                gender.append(0)
            else:
                gender.append(1)

        betas = torch.FloatTensor(betas).to(device)
        body_pose = torch.FloatTensor(body_pose).view(-1, 69).to(device)
        global_orient = torch.FloatTensor(global_orient).to(device)
        gender = torch.Tensor(gender).to(device)
        # cam_params = self.human_data['cam_param'][self.get_indices()]
        keypoints2d_gt = self.human_data['keypoints2d'][self.get_indices()]

        keypoints2d_gt[:, :, :2] = (
            keypoints2d_gt[:, :, :2] - bbox_xywh_square[:, None, :2]
        )
        keypoints2d_gt[:, :, :2] = (
            keypoints2d_gt[:, :, :2]
            / bbox_xywh_square[:, None, 2:]
            * self.hparams.IMG_RES
            // 4
        )
        keypoints2d_gt = torch.from_numpy(keypoints2d_gt).float().to(device)
        mesh_sample_data = np.load(
            'data/sample_params/uniform/sample_data_8-2021-04-05.npz'
        )  # hard-coded
        mesh_sample_data = dict(mesh_sample_data)
        ds_indices = mesh_sample_data['indices']
        chunk_size = 128
        i_chunk = 0

        gt_vertices2d = []
        while i_chunk * chunk_size < len(self):
            start = i_chunk * chunk_size
            end = min((i_chunk + 1) * chunk_size, len(self))
            gt_output_chunk = body_model(
                betas=betas[start:end],
                body_pose=body_pose[start:end],
                global_orient=global_orient[start:end],
                gender=gender[start:end],
            )
            gt_vertices = gt_output_chunk['vertices']
            gt_verts_down = gt_vertices[:, ds_indices]
            gt_keypoints2d = keypoints2d_gt[start:end]

            with torch.no_grad():
                gt_cam_t = estimate_translation(
                    gt_output_chunk['joints'],
                    gt_keypoints2d,
                    focal_length=5000,
                    img_size=self.hparams.IMG_RES // 4,
                    use_all_joints=True
                    if '3dpw' in self.hparams.DATASETS_AND_RATIOS
                    else False,
                )
                new_cam = get_cameras(
                    self.hparams.FOCAL_LENGTH, self.hparams.IMG_RES // 4, gt_cam_t
                )

                neural_renderer_voge_gt.cameras = new_cam

                vertices2d_proj = new_cam.transform_points_screen(
                    gt_verts_down,
                    image_size=(
                        (self.hparams.IMG_RES // 4, self.hparams.IMG_RES // 4),
                    ),
                )[
                    :, :, :2
                ]  # image_size is (W, H)

                mask, iskpvisible, _ = get_mask_and_visibility_voge(
                    vertices=gt_verts_down,
                    neural_renderer=neural_renderer_voge_gt.to(gt_vertices.device),
                    img_size=self.hparams.IMG_RES // 4,
                    have_occ=torch.zeros_like(gt_verts_down[:, 0, 0]),
                )

                vertices2d_proj = torch.cat(
                    (vertices2d_proj, iskpvisible[:, :, None]), dim=2
                )
            gt_vertices2d.append(vertices2d_proj.detach().cpu().numpy())
            i_chunk += 1
        gt_vertices2d = np.concatenate(gt_vertices2d, axis=0)
        return gt_vertices2d

    def _parse_result_2d(self, res):
        """Parse results."""
        bbox_xywh = self.human_data['bbox_xywh'][self.get_indices()]
        tl = (
            bbox_xywh[:, :2]
            + 0.5 * bbox_xywh[:, 2:4]
            - 0.5 * np.max(bbox_xywh[:, 2:4], axis=1, keepdims=True)
        )
        bbox_xywh_square = np.concatenate(
            (tl, np.maximum(bbox_xywh[:, 2:4], bbox_xywh[:, 3:1:-1])), axis=1
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if 'keypoints2d_pred' in res:
            # for other methods
            scales = bbox_xywh_square[:, 2]
            pred_keypoints2d = res['keypoints2d_pred'][:] * scales[:, None, None] / 224

        elif 'cameras' in res:
            # Project 3D keypoints to 2D (TODO wrap this in a function)
            pred_cam = torch.tensor(res['cameras'][:], device=device)
            # (B, 17, 3)
            pred_keypoints3d = torch.tensor(res['keypoints'][:], device=device)
            assert len(pred_keypoints3d) == len(self)

            pred_cam_t = convert_weak_perspective_to_perspective(
                pred_cam,
                focal_length=self.hparams.FOCAL_LENGTH,
                img_res=self.hparams.IMG_RES,
            )
            # pred_cam_t = pred_cam # prohmr
            pred_cam = get_cameras(
                self.hparams.FOCAL_LENGTH, self.hparams.IMG_RES, pred_cam_t
            )

            # For PyTorch3D >= 0.5.0, transform_points_screen is not behaving as expected
            # when camera is not defined in NDC space. The transform is composed of
            # world -> screen and screen -> NDC, where the latter should not use the input
            # `image_size` but it does.
            pred_keypoints2d = pred_cam.transform_points_screen(
                pred_keypoints3d, image_size=bbox_xywh_square[:, 2:4]
            )[
                :, :, :2
            ]  # image_size is (W, H)

            pred_keypoints2d = pred_keypoints2d.cpu().numpy()

        if self.dataset_name in [
            'pw3d',
            '3doh50k',
            'h36m',
            'briar_synthetic_part_allShape',
        ]:
            keypoints2d_gt = self.human_data['keypoints2d'][self.get_indices()]
            eval_visible_joints = (
                False if self.hparams is None else self.hparams.eval_visible_joints
            )
            if eval_visible_joints:
                if 'keypoints2d' in res:
                    keypoints2d_vis = np.array(res['keypoints2d'])[:, :, 2]
                    keypoints2d_gt[:, :, 2] = keypoints2d_vis[:]
                keypoints2d_gt = remove_outside_batch(keypoints2d_gt, bbox_xywh_square)

            gt_keypoints2d, keypoints2d_mask = convert_kps(
                keypoints2d_gt,
                src=self.convention,  #'smpl_49',
                dst='h36m',
                mask=self.human_data['keypoints2d_mask'],
            )
            gt_keypoints2d[:, :, :2] = (
                gt_keypoints2d[:, :, :2] - bbox_xywh_square[:, None, :2]
            )  # convert to local coordinates
            gt_keypoints2d_mask = gt_keypoints2d[:, :, 2]
        else:
            raise NotImplementedError(f'Does not support datasets {self.dataset_name}')

        if gt_keypoints2d.shape[1] == 17:
            # assert pred_keypoints2d.shape[1] == 17
            if pred_keypoints2d.shape[1] == 49:
                pred_keypoints2d, _ = convert_kps(
                    pred_keypoints2d, src='smpl_49', dst='h36m'
                )
            H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
            H36M_TO_J14 = H36M_TO_J17[:14]
            joint_mapper = H36M_TO_J14

            gt_keypoints2d = gt_keypoints2d[:, joint_mapper, :]
            if pred_keypoints2d.shape[1] == 17:
                pred_keypoints2d = pred_keypoints2d[:, joint_mapper, :]

        else:
            raise ValueError(f'Invalid shape: {gt_keypoints2d.shape}')

        gt_keypoints2d_mask = gt_keypoints2d_mask[:, joint_mapper] > 0
        return pred_keypoints2d, gt_keypoints2d, gt_keypoints2d_mask

    def _parse_result_vert2d(self, res):
        """Parse results."""
        bbox_xywh = self.human_data['bbox_xywh'][self.get_indices()]
        tl = (
            bbox_xywh[:, :2]
            + 0.5 * bbox_xywh[:, 2:4]
            - 0.5 * np.max(bbox_xywh[:, 2:4], axis=1, keepdims=True)
        )
        bbox_xywh_square = np.concatenate(
            (tl, np.maximum(bbox_xywh[:, 2:4], bbox_xywh[:, 3:1:-1])), axis=1
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        gt_vertices2d = self.calculate_vertices2d()  # [0, img_size//4]

        pred_vert2d = np.array(res['verts2d'])[:, :, :2]  # [0, img_size//4]

        return (
            pred_vert2d / (self.hparams.IMG_RES // 4),
            gt_vertices2d[:, :, :2] / (self.hparams.IMG_RES // 4),
            gt_vertices2d[:, :, 2],
        )

    def _report_mpjpe(self, res_file, metric='mpjpe', return_elementwise=False):
        """Calculate mean per joint position error (MPJPE) or its variants PA-
        MPJPE.

        Report mean per joint position error (MPJPE) and mean per joint
        position error after rigid alignment (PA-MPJPE)
        """
        pred_keypoints3d, gt_keypoints3d, gt_keypoints3d_mask = self._parse_result(
            res_file, mode='keypoint'
        )

        err_name = metric.upper()  # notice this
        if metric == 'mpjpe':
            alignment = 'none'
        elif metric == 'pa-mpjpe':
            alignment = 'procrustes'
        else:
            raise ValueError(f'Invalid metric: {metric}')

        if return_elementwise:
            error, error_ew = keypoint_mpjpe(
                pred_keypoints3d,
                gt_keypoints3d,
                gt_keypoints3d_mask,
                alignment,
                return_elementwise=True,
            )
            info_str = [(err_name, error)]
            return info_str, error_ew
        else:
            error = keypoint_mpjpe(
                pred_keypoints3d,
                gt_keypoints3d,
                gt_keypoints3d_mask,
                alignment,
                return_elementwise=False,
            )
        info_str = [(err_name, error)]

        return info_str

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

        pred_keypoints3d, gt_keypoints3d, gt_keypoints3d_mask = self._parse_result(
            res_file
        )

        err_name = metric.upper()
        if metric == '3dpck':
            alignment = 'none'
        elif metric == 'pa-3dpck':
            alignment = 'procrustes'
        else:
            raise ValueError(f'Invalid metric: {metric}')

        error = keypoint_3d_pck(
            pred_keypoints3d, gt_keypoints3d, gt_keypoints3d_mask, alignment
        )
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

        pred_keypoints3d, gt_keypoints3d, gt_keypoints3d_mask = self._parse_result(
            res_file
        )

        err_name = metric.upper()
        if metric == '3dauc':
            alignment = 'none'
        elif metric == 'pa-3dauc':
            alignment = 'procrustes'
        else:
            raise ValueError(f'Invalid metric: {metric}')

        error = keypoint_3d_auc(
            pred_keypoints3d, gt_keypoints3d, gt_keypoints3d_mask, alignment
        )
        name_value_tuples = [(err_name, error)]

        return name_value_tuples

    def _report_pve(self, res_file):
        """Calculate per vertex error."""
        pred_verts, gt_verts, _ = self._parse_result(res_file, mode='vertice')
        error = vertice_pve(pred_verts, gt_verts)
        return [('PVE', error)]

    def _report_2d_pckh(self, res_file, return_elementwise=False):
        """Calculate PCKh metric."""
        pred_keypoints2d, gt_keypoints2d, gt_keypoints2d_mask = self._parse_result_2d(
            res_file
        )

        if self.dataset_name in [
            'pw3d',
            '3doh50k',
            'h36m',
            'briar_synthetic_part_allShape',
        ]:
            thr_dist_ids = (12, 13)
            thr = 0.5
        if return_elementwise:
            error, pckh_ew = keypoint_2d_pckh(
                pred_keypoints2d,
                gt_keypoints2d,
                gt_keypoints2d_mask,
                thr=thr,
                thr_dist_ids=thr_dist_ids,
                return_elementwise=True,
            )
            return [('PCKh', error)], pckh_ew
        else:
            error = keypoint_2d_pckh(
                pred_keypoints2d,
                gt_keypoints2d,
                gt_keypoints2d_mask,
                thr=thr,
                thr_dist_ids=thr_dist_ids,
            )
        return [('PCKh', error)]

    def _report_2d_pck(self, res_file, thr=0.10, return_elementwise=False):
        """Calculate PCKh metric."""
        pred_keypoints2d, gt_keypoints2d, gt_keypoints2d_mask = self._parse_result_2d(
            res_file
        )
        pred_keypoints2d, gt_keypoints2d = (
            pred_keypoints2d / self.hparams.IMG_RES,
            gt_keypoints2d / self.hparams.IMG_RES,
        )
        if return_elementwise:
            error, pck_ew = keypoint_2d_pck(
                pred_keypoints2d,
                gt_keypoints2d,
                gt_keypoints2d_mask,
                thr=thr,
                return_elementwise=True,
            )
            return [('PCK', error)], pck_ew
        else:
            error = keypoint_2d_pck(
                pred_keypoints2d, gt_keypoints2d, gt_keypoints2d_mask, thr=thr
            )
        return [('PCK', error)]

    def _report_2d_vert_pck(self, res_file, thr=0.05, return_elementwise=False):
        """Calculate PCK metric."""
        pred_vert2d, gt_vert2d, gt_vert2d_mask = self._parse_result_vert2d(res_file)

        if return_elementwise:
            error, pckh_ew = keypoint_2d_pck(
                pred_vert2d, gt_vert2d, gt_vert2d_mask, thr=thr, return_elementwise=True
            )
            return [('PCK vert', error)], pckh_ew
        else:
            error = keypoint_2d_pck(pred_vert2d, gt_vert2d, gt_vert2d_mask, thr=thr)
        return [('PCK vert', error)]
