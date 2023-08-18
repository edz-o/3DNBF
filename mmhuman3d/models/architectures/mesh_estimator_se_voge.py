from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple, Union
from yacs.config import CfgNode as CN
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import pytorch3d
import numpy as np
import cv2
from pytorch3d.transforms.rotation_conversions import matrix_to_axis_angle
from openpose_pytorch import torch_openpose
from mmcv import imdenormalize

import sys

sys.path.append('./smplx-master')
from mmhuman3d.core.conventions.keypoints_mapping import get_keypoint_idx
from mmhuman3d.core.conventions import constants
from mmhuman3d.models.utils import FitsDict
from mmhuman3d.models.heads import PareHeadwCoKeNeMoAttn
from mmhuman3d.utils.geometry import (
    batch_rodrigues,
    estimate_translation,
    project_points,
    rotation_matrix_to_angle_axis,
    convert_weak_perspective_to_perspective,
    convert_perspective_to_weak_perspective,
    rotate_aroundy,
)
from mmhuman3d.utils.image_utils import (
    generate_part_labels,
    get_vert_to_part,
    generate_part_labels_voge,
    get_mask_and_visibility,
    get_mask_and_visibility_voge,
    get_vert_orients,
)
from mmhuman3d.utils.neuralsmpl_utils import get_detected_2d_vertices
from mmhuman3d.utils.neural_renderer import (
    build_neural_renderer,
    get_blend_params,
    get_cameras,
)
from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_num,
)
from mmhuman3d.utils.neural_renderer_voge import build_neural_renderer_voge
from ..builder import (
    ARCHITECTURES,
    build_backbone,
    build_body_model,
    build_discriminator,
    build_head,
    build_loss,
    build_neck,
    build_registrant,
    build_feature_bank,
)
import json
from .base_architecture import BaseArchitecture
from ..registrants import NeuralSMPLFitting, NeuralSMPLFittingVoGE
from mmhuman3d.utils.vis_utils import (
    SMPLVisualizer,
)
from mmhuman3d.data.datasets.pipelines.transforms import _flip_smpl_pose_batch


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class VoGEBodyModelEstimatorSE(BaseArchitecture, metaclass=ABCMeta):
    """BodyModelEstimator Architecture.

    Args:
        backbone (dict | None, optional): Backbone config dict. Default: None.
        neck (dict | None, optional): Neck config dict. Default: None
        head (dict | None, optional): Regressor config dict. Default: None.
        disc (dict | None, optional): Discriminator config dict.
            Default: None.
        registrant ( dict | None, optional): Registrant config dict.
            Default: None.
        body_model_train (dict | None, optional): SMPL config dict during
            training. Default: None.
        body_model_test (dict | None, optional): SMPL config dict during
            test. Default: None.
        convention (str, optional): Keypoints convention. Default: "human_data"
        loss_keypoints2d (dict | None, optional): Losses config dict for
            2D keypoints. Default: None.
        loss_keypoints3d (dict | None, optional): Losses config dict for
            3D keypoints. Default: None.
        loss_vertex (dict | None, optional): Losses config dict for mesh
            vertices. Default: None
        loss_smpl_pose (dict | None, optional): Losses config dict for smpl
            pose. Default: None
        loss_smpl_betas (dict | None, optional): Losses config dict for smpl
            betas. Default: None
        loss_camera (dict | None, optional): Losses config dict for predicted
            camera parameters. Default: None
        loss_adv (dict | None, optional): Losses config for adversial
            training. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(
        self,
        backbone: Optional[Union[dict, None]] = None,
        neck: Optional[Union[dict, None]] = None,
        head: Optional[Union[dict, None]] = None,
        feature_bank: Optional[Union[dict, None]] = None,
        disc: Optional[Union[dict, None]] = None,
        registrant: Optional[Union[dict, None]] = None,
        body_model_train: Optional[Union[dict, None]] = None,
        body_model_test: Optional[Union[dict, None]] = None,
        convention: Optional[str] = 'human_data',
        loss_keypoints2d: Optional[Union[dict, None]] = None,
        loss_keypoints3d: Optional[Union[dict, None]] = None,
        loss_vertex: Optional[Union[dict, None]] = None,
        loss_smpl_pose: Optional[Union[dict, None]] = None,
        loss_smpl_betas: Optional[Union[dict, None]] = None,
        loss_camera: Optional[Union[dict, None]] = None,
        loss_segm_mask: Optional[Union[dict, None]] = None,
        loss_part_segm: Optional[Union[dict, None]] = None,
        loss_contrastive: Optional[Union[dict, None]] = None,
        # loss_noise_reg: Optional[Union[dict, None]] = None,
        loss_adv: Optional[Union[dict, None]] = None,
        init_cfg: Optional[Union[list, dict, None]] = None,
        hparams: Optional[Union[dict, None]] = None,
    ):
        super(VoGEBodyModelEstimatorSE, self).__init__(init_cfg)

        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.head = build_head(head)
        self.disc = build_discriminator(disc)
        self.hparams = CN.load_cfg(str(hparams))

        self.feature_bank = build_feature_bank(feature_bank)
        self.body_model_train = build_body_model(body_model_train)
        self.body_model_test = build_body_model(body_model_test)
        self.convention = convention

        # TODO: support HMR+

        self.loss_keypoints2d = build_loss(loss_keypoints2d)
        self.loss_keypoints3d = build_loss(loss_keypoints3d)
        self.loss_vertex = build_loss(loss_vertex)
        self.loss_smpl_pose = build_loss(loss_smpl_pose)
        self.loss_smpl_betas = build_loss(loss_smpl_betas)
        self.loss_adv = build_loss(loss_adv)
        self.loss_camera = build_loss(loss_camera)
        self.loss_segm_mask = build_loss(loss_segm_mask)
        self.loss_part_segm = build_loss(loss_part_segm)
        self.loss_contrastive = build_loss(loss_contrastive)

        set_requires_grad(self.body_model_train, False)
        set_requires_grad(self.body_model_test, False)

        self.vert_to_part = get_vert_to_part()

        self.neural_renderer_gt = build_neural_renderer(hparams.MODEL.RENDERER_GT)
        self.neural_renderer_voge_gt = build_neural_renderer_voge(
            hparams.MODEL.RENDERER_GT
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_mesh_info()

        self.registrant = build_registrant(registrant)
        if registrant is not None:
            self.fits = 'registration'
            self.fits_dict = FitsDict(fits='static')
            if isinstance(self.registrant, NeuralSMPLFitting) or isinstance(
                self.registrant, NeuralSMPLFittingVoGE
            ):
                self.registrant.set_mesh_info(
                    ds_indices=self.ds_indices, faces_down=self.faces_down
                )
                self.openpose_body_estimator = torch_openpose.torch_openpose(
                    'body_25', 'third_party/pytorch_openpose_body_25/model/body_25.pth'
                )

                if self.hparams.REGISTRANT.RUN_VISUALIZER:
                    visualizer = SMPLVisualizer(
                        self.body_model_test,
                        'cuda',
                        None,
                        image_size=(
                            self.hparams.VISUALIZER.IMG_RES,
                            self.hparams.VISUALIZER.IMG_RES,
                        ),
                        point_light_location=((0, 0, -3.0),),
                        shader_type=self.hparams.VISUALIZER.SHADER_TYPE,
                    )
                    self.registrant.set_visualizer(visualizer)
                    self.writer = SummaryWriter(
                        log_dir=self.hparams.VISUALIZER.DEBUG_LOG_DIR
                    )
                else:
                    self.writer = None

    def load_mesh_info(self):
        mesh_sample_data = np.load(self.hparams.mesh_sample_param_path)
        mesh_sample_data = dict(mesh_sample_data)
        self.ds_indices = mesh_sample_data['indices']
        n = len(self.ds_indices)

        self.faces_down = torch.tensor(
            mesh_sample_data['faces_downsampled'], device=self.device
        )
        adj_mat = torch.zeros((n, n)).long()
        for face in self.faces_down:
            x1, x2, x3 = sorted(face)
            adj_mat[x1, x2] = 1
            adj_mat[x1, x3] = 1
            adj_mat[x2, x3] = 1
        self.adj_mat = (adj_mat + adj_mat.transpose(1, 0)).unsqueeze(0)

    def train_step(self, data_batch, optimizer, **kwargs):
        """
        Train step function.
        In this function, the detector will finish the train step following
        the pipeline:
        1. get fake and real SMPL parameters
        2. optimize discriminator (if have)
        3. optimize generator
        If `self.train_cfg.disc_step > 1`, the train step will contain multiple
        iterations for optimizing discriminator with different input data and
        only one iteration for optimizing generator after `disc_step`
        iterations for discriminator.
        Args:
            data_batch (torch.Tensor): Batch of data as input.
            optimizer (dict[torch.optim.Optimizer]): Dict with optimizers for
                generator and discriminator (if have).
        Returns:
            outputs (dict): Dict with loss, information for logger,
            the number of samples.
        """
        # mmcv/runner/epoch_based_runner.py#L29, add kwargs['iters'] = self.iter
        self.eval()
        _iter = kwargs['iter']

        # get basic predictions and GT
        if self.backbone is not None:
            img = data_batch['img']
            features = self.backbone(img)
        else:
            features = data_batch['features']
        if self.neck is not None:
            features = self.neck(features)
        if isinstance(self.head, PareHeadwCoKeNeMoAttn):
            predictions = self.head(
                features, self.feature_bank.get_feature_banks_original_order()
            )
        else:
            predictions = self.head(features)
        targets = self.prepare_targets(data_batch)

        # optimize discriminator we don't have, so continue
        if self.disc is not None:
            self.optimize_discrinimator(predictions, data_batch, optimizer)

        # calculating all loss
        losses = self.compute_losses(predictions, targets)

        # optimize discriminator we don't have, so continue
        if self.disc is not None:
            adv_loss = self.optimize_generator(predictions)
            losses.update(adv_loss)

        # after train
        loss, log_vars = self._parse_losses(losses)

        loss = loss / self.hparams.cumulative_iters
        loss.backward()
        if _iter % self.hparams.cumulative_iters == 0:
            if self.backbone is not None:
                optimizer['backbone'].step()
            if self.neck is not None:
                optimizer['neck'].step()
            if self.head is not None:
                optimizer['head'].step()

            if self.backbone is not None:
                optimizer['backbone'].zero_grad()
            if self.neck is not None:
                optimizer['neck'].zero_grad()
            if self.head is not None:
                optimizer['head'].zero_grad()

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))),
        )

        return outputs

    def run_registration_test_voge(
        self,
        predictions: dict,
        targets: dict,
        threshold: Optional[float] = 10.0,
        focal_length: Optional[float] = 5000.0,
        img_res: Optional[Union[Tuple[int], int]] = 224,
        img: Optional[torch.Tensor] = None,
    ) -> dict:
        """Run registration on 2D keypoints in predictions to obtain SMPL
        parameters as pseudo ground truth.

        Args:
            predictions (dict): predicted SMPL parameters are used for
                initialization.
            targets (dict): existing ground truths with 2D keypoints
            threshold (float, optional): the threshold to update fits
                dictionary. Default: 10.0.
            focal_length (tuple(int) | int, optional): camera focal_length
            img_res (int, optional): image resolution

        Returns:
            targets: contains additional SMPL parameters
        """

        img_metas = targets['img_metas']
        ds_rate = self.hparams.REGISTRANT.downsample_rate
        dataset_name = [
            meta['dataset_name'] for meta in img_metas
        ]  # name of the dataset the image comes from

        indices = targets['sample_idx'].squeeze()
        if 'is_flipped' not in targets:
            is_flipped = torch.zeros_like(targets['sample_idx']).bool()
        else:
            is_flipped = (
                targets['is_flipped'].squeeze().bool()
            )  # flag that indicates whether image was flipped
        # during data augmentation
        rot_angle = targets[
            'rotation'
        ].squeeze()  # rotation angle used for data augmentation Q

        if self.hparams.REGISTRANT.use_other_init:
            pred_rotmat = targets['poses_init'].float()
            pred_betas = targets['betas_init'].float()
            pred_cam = targets['cameras_init'].float()
            keypoinst3d = targets['keypoints3d_init'].float()
        else:
            pred_rotmat = predictions['pred_pose'].detach().clone()
            pred_betas = predictions['pred_shape'].detach().clone()
            pred_cam = predictions['pred_cam'].detach().clone()

        if (
            hasattr(self.hparams.MODEL, 'NON_STANDARD_WEAK_CAM')
        ) and self.hparams.MODEL.NON_STANDARD_WEAK_CAM:
            pred_cam[:, 1:] = pred_cam[:, 1:] / pred_cam[:, 0:1]

        pred_cam_t = convert_weak_perspective_to_perspective(
            pred_cam,
            focal_length=focal_length,
            img_res=img_res,
        )
        if 'pred_segm_mask' in predictions:
            pred_segm_mask = predictions['pred_segm_mask'].detach().clone()
        elif self.hparams.REGISTRANT.use_saved_partseg:
            pred_segm_mask = targets['pred_segm_mask']
            if pred_segm_mask.shape[-1] != img_res // ds_rate:
                pred_segm_mask = F.interpolate(
                    pred_segm_mask,
                    (img_res // ds_rate, img_res // ds_rate),
                    mode='bilinear',
                    align_corners=True,
                )
        else:
            pred_segm_mask = None

        if 'mask' in targets:
            gt_mask = targets['mask']
            if gt_mask.shape[-1] != img_res // ds_rate:
                gt_mask = (
                    F.interpolate(
                        gt_mask.unsqueeze(1),
                        (img_res // ds_rate, img_res // ds_rate),
                        mode='nearest',
                    )
                    .squeeze(1)
                    .long()
                )
        else:
            gt_mask = None

        if self.hparams.REGISTRANT.use_saved_coke:
            coke_features = targets['coke_features'].float()
        elif 'coke_features' in predictions:
            coke_features = predictions['coke_features'].detach().clone()
        else:
            coke_features = None

        # TODO (simplify this) now only work for testing
        bbox_xywh = targets['bbox_xywh']
        tl = (
            bbox_xywh[:, :2]
            + 0.5 * bbox_xywh[:, 2:4]
            - 0.5 * bbox_xywh[:, 2:4].max(dim=1, keepdim=True)[0]
        )
        bbox_xyxy = torch.cat(
            (bbox_xywh[:, :2] - tl, bbox_xywh[:, :2] + bbox_xywh[:, 2:4] - tl), dim=1
        )
        scale = bbox_xywh[:, 2:4].max(dim=1)[0] / self.hparams.DATASET.IMG_RES
        bbox_xyxy = bbox_xyxy / scale[:, None]
        vertices2d_det, vertices2d_det_conf = get_detected_2d_vertices(
            coke_features,
            bbox_xyxy,
            self.registrant.neural_mesh_model_voge.features,
            self.hparams.MODEL.downsample_rate,
            n_orient=self.registrant.hparams.n_orient,
        )
        vertices2d_det = torch.cat(
            (vertices2d_det, vertices2d_det_conf.unsqueeze(2)), dim=2
        )

        gt_keypoints_2d = targets['keypoints2d'].float()
        num_keypoints = gt_keypoints_2d.shape[1]
        has_smpl = (
            targets['has_smpl'].view(-1).bool()
        )  # flag that indicates whether SMPL parameters are valid
        batch_size = has_smpl.shape[0]
        device = has_smpl.device

        # Get inital fits from the prediction
        opt_pose = matrix_to_axis_angle(pred_rotmat).flatten(1)
        # opt_pose[:, 3:] = torch.randn_like(opt_pose[:, 3:]) * 0.4
        opt_pose = opt_pose.to(device)
        opt_betas = pred_betas.to(device)
        opt_cam_t = pred_cam_t.clone().to(device)

        opt_output = self.body_model_train(
            betas=opt_betas, body_pose=opt_pose[:, 3:], global_orient=opt_pose[:, :3]
        )

        # (TODO) opt_joints, opt_vertices, opt_output are not used?
        if num_keypoints == 49:
            opt_joints = opt_output['joints']
            opt_vertices = opt_output['vertices']
        else:
            opt_joints = opt_output['joints'][:, 25:, :]
            opt_vertices = opt_output['vertices']

        # TODO: current pipeline, the keypoints are already in the pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        # remove gt keypoints, only use openpose keypoints
        gt_keypoints_2d_orig[:, 25:, 2] = 0

        with torch.no_grad():
            loss_dict = self.registrant.evaluate(
                global_orient=opt_pose[:, :3],
                body_pose=opt_pose[:, 3:],
                betas=opt_betas,
                transl=opt_cam_t,
                keypoints2d=gt_keypoints_2d_orig[:, :, :2],
                keypoints2d_conf=gt_keypoints_2d_orig[:, :, 2],
                vertices2d=vertices2d_det[:, :, :2],
                vertices2d_conf=vertices2d_det[:, :, 2],
                reduction_override='none',
                pred_segm_mask=pred_segm_mask,
                gt_mask=gt_mask,
                predicted_map=coke_features,
            )
        opt_loss = loss_dict['total_loss']

        init_pose = opt_pose.clone()
        init_betas = opt_betas.clone()
        init_cam_t = opt_cam_t.clone()

        def _optimization(
            gt_keypoints_2d_orig,
            vertices2d_det,
            pred_segm_mask,
            gt_mask,
            coke_features,
            opt_pose,
            opt_betas,
            opt_cam_t,
        ):
            registrant_output = self.registrant(
                keypoints2d=gt_keypoints_2d_orig[:, :, :2],
                keypoints2d_conf=gt_keypoints_2d_orig[:, :, 2],
                # keypoints3d=keypoinst3d,
                # keypoints3d_conf=torch.ones_like(keypoinst3d[:, :, 0]),
                vertices2d=vertices2d_det[:, :, :2],
                vertices2d_conf=vertices2d_det[:, :, 2],
                pred_segm_mask=pred_segm_mask,
                gt_mask=gt_mask,
                predicted_map=coke_features,
                init_global_orient=opt_pose[:, :3],
                init_transl=opt_cam_t,  # only correct when Rotation is None
                init_body_pose=opt_pose[:, 3:],
                init_betas=opt_betas,
                img=img,
                img_meta=img_metas,
                return_joints=True,
                return_verts=True,
                return_losses=True,
            )

            new_opt_vertices = registrant_output['vertices']
            new_opt_joints = registrant_output['joints']

            new_opt_global_orient = registrant_output['global_orient']
            new_opt_body_pose = registrant_output['body_pose']
            new_opt_pose = torch.cat([new_opt_global_orient, new_opt_body_pose], dim=1)

            new_opt_betas = registrant_output['betas']
            new_opt_cam_t = registrant_output['transl']
            new_opt_loss = registrant_output['total_loss']
            return (
                new_opt_loss,
                new_opt_vertices,
                new_opt_joints,
                new_opt_pose,
                new_opt_betas,
                new_opt_cam_t,
            )

        def _update(
            opt_loss,
            new_opt_loss,
            opt_vertices,
            new_opt_vertices,
            opt_joints,
            new_opt_joints,
            opt_pose,
            new_opt_pose,
            opt_betas,
            new_opt_betas,
            opt_cam_t,
            new_opt_cam_t,
        ):
            # Will update the dictionary for the examples where the new loss
            # is less than the current one
            update = new_opt_loss < opt_loss
            # update = torch.ones_like(update).bool()
            opt_loss[update] = new_opt_loss[update]
            opt_vertices[update, :] = new_opt_vertices[update, :]
            opt_joints[update, :] = new_opt_joints[update, :]
            opt_pose[update, :] = new_opt_pose[update, :]
            opt_betas[update, :] = new_opt_betas[update, :]
            opt_cam_t[update, :] = new_opt_cam_t[update, :]

            # Replace extreme betas with zero betas
            opt_betas[(opt_betas.abs() > 3).any(dim=-1)] = 0.0

        global_orient_rot = rotate_aroundy(init_pose[:, :3], 180)
        init_pose_rotflip = _flip_smpl_pose_batch(
            torch.cat([global_orient_rot, init_pose[:, 3:]], dim=1)
        )
        with torch.no_grad():
            loss_dict_rot = self.registrant.evaluate(
                global_orient=init_pose_rotflip[:, :3],
                body_pose=init_pose_rotflip[:, 3:],
                betas=opt_betas,
                transl=opt_cam_t,
                keypoints2d=gt_keypoints_2d_orig[:, :, :2],
                keypoints2d_conf=gt_keypoints_2d_orig[:, :, 2],
                vertices2d=vertices2d_det[:, :, :2],
                vertices2d_conf=vertices2d_det[:, :, 2],
                reduction_override='none',
                pred_segm_mask=pred_segm_mask,
                gt_mask=gt_mask,
                predicted_map=coke_features,
            )
        opt_loss_rot = loss_dict_rot['total_loss']  # loss for initial prediction
        update = opt_loss_rot < opt_loss
        opt_pose[update] = init_pose_rotflip[update]
        opt_loss[update] = opt_loss_rot[update]

        # Evaluate GT SMPL parameters
        if (
            hasattr(self.hparams.REGISTRANT, 'use_gt_smpl')
            and self.hparams.REGISTRANT.use_gt_smpl
        ):
            gt_pose, gt_betas, gt_cam_t, _ = self.get_gt_smpl(
                targets, self.body_model_train
            )
            with torch.no_grad():
                loss_dict_gt = self.registrant.evaluate(
                    global_orient=gt_pose[:, :3],
                    body_pose=gt_pose[:, 3:],
                    betas=gt_betas,
                    transl=gt_cam_t,
                    gt_mask=gt_mask,
                    keypoints2d=gt_keypoints_2d_orig[:, :, :2],
                    keypoints2d_conf=gt_keypoints_2d_orig[:, :, 2],
                    vertices2d=vertices2d_det[:, :, :2],
                    vertices2d_conf=vertices2d_det[:, :, 2],
                    reduction_override='none',
                    pred_segm_mask=pred_segm_mask,
                    predicted_map=coke_features,
                )
            opt_loss_gt = loss_dict_gt['total_loss']  # loss for initial prediction
            update = opt_loss_gt < opt_loss
            opt_pose[update] = gt_pose[update]
            opt_loss[update] = opt_loss_gt[update]
            opt_cam_t[update] = gt_cam_t[update]
            opt_pose = gt_pose
            opt_cam_t = gt_cam_t
            opt_betas = gt_betas

            init_pose = opt_pose.clone()
            init_betas = opt_betas.clone()
            init_cam_t = opt_cam_t.clone()

        self.registrant.set_summary_writer(self.writer)

        hypotheses = [
            (init_pose.clone(), init_betas.clone(), init_cam_t.clone()),
        ]
        if self.hparams.REGISTRANT.get('optimize_twoside', False):
            hypotheses.append(
                (init_pose_rotflip, init_betas.clone(), init_cam_t.clone())
            )
        for _init_pose, _init_betas, _init_cam_t in hypotheses:
            for lr in self.hparams.REGISTRANT.get(
                'LRS', [self.registrant.optimizer.lr]
            ):
                self.registrant.optimizer.lr = lr
                (
                    new_opt_loss,
                    new_opt_vertices,
                    new_opt_joints,
                    new_opt_pose,
                    new_opt_betas,
                    new_opt_cam_t,
                ) = _optimization(
                    gt_keypoints_2d_orig,
                    vertices2d_det,
                    pred_segm_mask,
                    gt_mask,
                    coke_features,
                    _init_pose,
                    _init_betas,
                    _init_cam_t,
                )

                _update(
                    opt_loss,
                    new_opt_loss,
                    opt_vertices,
                    new_opt_vertices,
                    opt_joints,
                    new_opt_joints,
                    opt_pose,
                    new_opt_pose,
                    opt_betas,
                    new_opt_betas,
                    opt_cam_t,
                    new_opt_cam_t,
                )

        # Replace extreme betas with zero betas
        opt_betas[(opt_betas.abs() > 3).any(dim=-1)] = 0.0

        # Replace the optimized parameters with the ground truth parameters,
        # if available

        # Assert whether a fit is valid by comparing the joint loss with
        # the threshold
        valid_fit = (opt_loss < threshold).to(device)
        valid_fit = valid_fit | has_smpl
        targets['valid_fit'] = valid_fit

        targets['opt_vertices'] = opt_vertices
        targets['opt_cam_t'] = opt_cam_t
        targets['opt_joints'] = opt_joints
        targets['opt_pose'] = opt_pose
        targets['opt_betas'] = opt_betas

        targets['vertices2d_det'] = vertices2d_det

        return targets

    def optimize_discrinimator(
        self, predictions: dict, data_batch: dict, optimizer: dict
    ):
        """Optimize discrinimator during adversarial training."""
        set_requires_grad(self.disc, True)
        fake_data = self.make_fake_data(predictions, requires_grad=False)
        real_data = self.make_real_data(data_batch)
        fake_score = self.disc(fake_data)
        real_score = self.disc(real_data)

        disc_losses = {}
        disc_losses['real_loss'] = self.loss_adv(
            real_score, target_is_real=True, is_disc=True
        )
        disc_losses['fake_loss'] = self.loss_adv(
            fake_score, target_is_real=False, is_disc=True
        )
        loss_disc, log_vars_d = self._parse_losses(disc_losses)

        optimizer['disc'].zero_grad()
        loss_disc.backward()
        optimizer['disc'].step()

    def optimize_generator(self, predictions: dict):
        """Optimize generator during adversarial training."""
        set_requires_grad(self.disc, False)
        fake_data = self.make_fake_data(predictions, requires_grad=True)
        pred_score = self.disc(fake_data)
        loss_adv = self.loss_adv(pred_score, target_is_real=True, is_disc=False)
        loss = dict(adv_loss=loss_adv)
        return loss

    def compute_keypoints3d_loss(
        self, pred_keypoints3d: torch.Tensor, gt_keypoints3d: torch.Tensor
    ):
        """Compute loss for 3d keypoints."""
        keypoints3d_conf = gt_keypoints3d[:, :, 3].float().unsqueeze(-1)
        keypoints3d_conf = keypoints3d_conf.repeat(1, 1, 3)
        pred_keypoints3d = pred_keypoints3d.float()
        gt_keypoints3d = gt_keypoints3d[:, :, :3].float()

        # currently, only mpi_inf_3dhp and h36m have 3d keypoints
        # both datasets have right_hip_extra and left_hip_extra
        right_hip_idx = get_keypoint_idx('right_hip_extra', self.convention)
        left_hip_idx = get_keypoint_idx('left_hip_extra', self.convention)
        gt_pelvis = (
            gt_keypoints3d[:, right_hip_idx, :] + gt_keypoints3d[:, left_hip_idx, :]
        ) / 2
        pred_pelvis = (
            pred_keypoints3d[:, right_hip_idx, :] + pred_keypoints3d[:, left_hip_idx, :]
        ) / 2

        gt_keypoints3d = gt_keypoints3d - gt_pelvis[:, None, :]
        pred_keypoints3d = pred_keypoints3d - pred_pelvis[:, None, :]
        loss = self.loss_keypoints3d(
            pred_keypoints3d, gt_keypoints3d, reduction_override='none'
        )

        # import numpy as np; np.savez('debug_normal_h36m.npz', pred=pred_keypoints3d.detach().cpu().numpy(), gt=gt_keypoints3d.detach().cpu().numpy(), conf=keypoints3d_conf.detach().cpu().numpy())
        valid_pos = keypoints3d_conf > 0
        if keypoints3d_conf[valid_pos].numel() == 0:
            return torch.Tensor([0]).type_as(gt_keypoints3d)
        loss = torch.sum(loss * keypoints3d_conf)
        loss /= keypoints3d_conf[valid_pos].numel()
        return loss

    def compute_keypoints2d_loss(
        self,
        pred_keypoints3d: torch.Tensor,
        pred_cam: torch.Tensor,
        gt_keypoints2d: torch.Tensor,
        img_res: Optional[int] = 224,
        focal_length: Optional[int] = 5000,
    ):
        """Compute loss for 2d keypoints."""
        keypoints2d_conf = gt_keypoints2d[:, :, 2].float().unsqueeze(-1)
        keypoints2d_conf = keypoints2d_conf.repeat(1, 1, 2)
        gt_keypoints2d = gt_keypoints2d[:, :, :2].float()
        pred_keypoints2d = project_points(
            pred_keypoints3d, pred_cam, focal_length=focal_length, img_res=img_res
        )
        # Normalize keypoints to [-1,1]
        # The coordinate origin of pred_keypoints_2d is
        # the center of the input image.
        pred_keypoints2d = 2 * pred_keypoints2d / (img_res - 1)
        # The coordinate origin of gt_keypoints_2d is
        # the top left corner of the input image.
        gt_keypoints2d = 2 * gt_keypoints2d / (img_res - 1) - 1
        loss = self.loss_keypoints2d(
            pred_keypoints2d, gt_keypoints2d, reduction_override='none'
        )
        valid_pos = keypoints2d_conf > 0
        if keypoints2d_conf[valid_pos].numel() == 0:
            return torch.Tensor([0]).type_as(gt_keypoints2d)
        loss = torch.sum(loss * keypoints2d_conf)
        loss /= keypoints2d_conf[valid_pos].numel()
        return loss

    def compute_vertex_loss(
        self,
        pred_vertices: torch.Tensor,
        gt_vertices: torch.Tensor,
        has_smpl: torch.Tensor,
    ):
        """Compute loss for vertices."""
        gt_vertices = gt_vertices.float()
        conf = has_smpl.float().view(-1, 1, 1)
        conf = conf.repeat(1, gt_vertices.shape[1], gt_vertices.shape[2])
        loss = self.loss_vertex(pred_vertices, gt_vertices, reduction_override='none')
        valid_pos = conf > 0
        if conf[valid_pos].numel() == 0:
            return torch.Tensor([0]).type_as(gt_vertices)
        loss = torch.sum(loss * conf) / conf[valid_pos].numel()
        return loss

    def compute_smpl_pose_loss(
        self, pred_rotmat: torch.Tensor, gt_pose: torch.Tensor, has_smpl: torch.Tensor
    ):
        """Compute loss for smpl pose."""
        conf = has_smpl.float().view(-1, 1, 1, 1).repeat(1, 24, 3, 3)
        gt_rotmat = batch_rodrigues(gt_pose.view(-1, 3)).view(-1, 24, 3, 3)
        loss = self.loss_smpl_pose(pred_rotmat, gt_rotmat, reduction_override='none')
        valid_pos = conf > 0
        if conf[valid_pos].numel() == 0:
            return torch.Tensor([0]).type_as(gt_pose)
        loss = torch.sum(loss * conf) / conf[valid_pos].numel()
        return loss

    def compute_smpl_betas_loss(
        self, pred_betas: torch.Tensor, gt_betas: torch.Tensor, has_smpl: torch.Tensor
    ):
        """Compute loss for smpl betas."""
        conf = has_smpl.float().view(-1, 1).repeat(1, 10)
        loss = self.loss_smpl_betas(pred_betas, gt_betas, reduction_override='none')
        valid_pos = conf > 0
        if conf[valid_pos].numel() == 0:
            return torch.Tensor([0]).type_as(gt_betas)
        loss = torch.sum(loss * conf) / conf[valid_pos].numel()
        return loss

    def compute_camera_loss(self, cameras: torch.Tensor):
        """Compute loss for predicted camera parameters."""
        loss = self.loss_camera(cameras)
        return loss

    def compute_segm_mask_loss(
        self, pred_segm_mask: torch.Tensor, gt_segm_mask: torch.Tensor
    ):
        loss = self.loss_segm_mask(pred_segm_mask, gt_segm_mask)
        return loss

    def compute_part_segm_loss(self, pred_segm_rgb, gt_segm_rgb):
        loss = self.loss_part_segm(pred_segm_rgb, gt_segm_rgb)
        return loss

    def compute_coke_loss(
        self,
        coke_features,
        keypoint_positions,
        has_smpl,
        iskpvisible,
        feature_bank,
        adj_mat,
        vert_orients=None,
        bg_mask=None,
        mask=None,
        vert_coke_features=None,
    ):
        """Verbose"""

        losses = self.loss_contrastive(
            coke_features,
            keypoint_positions,
            has_smpl,
            iskpvisible,
            feature_bank,
            adj_mat,
            vert_orients=vert_orients,
            bg_mask=bg_mask,
            mask=mask,
            vert_coke_features=vert_coke_features,
        )
        return losses

    def compute_losses(self, predictions: dict, targets: dict):
        """Compute losses."""
        bs = targets['keypoints3d'].shape[0]
        if self.hparams.disable_inference_loss is False:
            pred_betas = predictions['pred_shape'].view(-1, 10)
            pred_pose = predictions['pred_pose'].view(
                -1, 24, 3, 3
            )  # pred_pose N, 24, 3, 3
            pred_cam = predictions['pred_cam'].view(-1, 3)
            gt_keypoints3d = targets['keypoints3d']
        gt_keypoints2d = targets['keypoints2d']
        coke_features = predictions['coke_features']

        have_occ = targets['have_occ']
        # if have_occ:
        occ_size = targets['occ_size']
        occ_stride = targets['occ_stride']
        occ_idx = targets['occ_idx']
        occ_mask = targets['occ_mask']
        # get pred kp3d & vertice based on pred betas & pose
        if (
            self.body_model_train is not None
            and self.hparams.disable_inference_loss is False
        ):
            pred_output = self.body_model_train(
                betas=pred_betas,
                body_pose=pred_pose[:, 1:],
                global_orient=pred_pose[:, 0].unsqueeze(1),
                pose2rot=False,
                num_joints=gt_keypoints2d.shape[1],
            )
            pred_keypoints3d = pred_output['joints']
            pred_vertices = pred_output['vertices']

        # have registrant: valid_fit that we don't have, go else
        if 'valid_fit' in targets:
            has_smpl = targets['valid_fit'].view(-1)
            gt_pose = targets['opt_pose']
            gt_betas = targets['opt_betas']
            gt_vertices = targets['opt_vertices']
            if self.body_model_train is not None:
                gt_output = self.body_model_train(
                    betas=gt_betas,
                    body_pose=gt_pose[:, 3:],
                    global_orient=gt_pose[:, :3],
                    num_joints=gt_keypoints2d.shape[1],
                )
        else:
            has_smpl = targets['has_smpl'].view(-1)
            gt_pose = targets['smpl_body_pose']  # B, 23, 3
            global_orient = targets['smpl_global_orient'].view(-1, 1, 3)
            gt_pose = (
                torch.cat((global_orient, gt_pose), dim=1).float().flatten(1)
            )  # gt_pose N, 72
            gt_betas = targets['smpl_betas'].float()
            # get GT kp3d & vertice based on GT betas & pose
            if self.body_model_train is not None:
                gt_output = self.body_model_train(
                    betas=gt_betas,
                    body_pose=gt_pose[:, 3:],
                    global_orient=gt_pose[:, :3],
                    num_joints=gt_keypoints2d.shape[1],
                )
                gt_keypoints3d_smpl = gt_output['joints']
                gt_vertices = gt_output['vertices']

                has_kp3d = targets['has_kp3d'].view(-1)
                gt_keypoints3d = gt_keypoints3d.float()
                gt_keypoints3d[
                    (has_smpl == 1) & (has_kp3d == 0), :, :3
                ] = gt_keypoints3d_smpl[(has_smpl == 1) & (has_kp3d == 0), :, :]
                gt_keypoints3d[(has_smpl == 1) & (has_kp3d == 0), :, 3] = 1.0
            # downsample
            gt_verts_down = gt_vertices[:, self.ds_indices]  # for both if/else

        losses = {}

        # kp3d
        if (
            self.loss_keypoints3d is not None
            and self.hparams.disable_inference_loss is False
        ):
            losses['keypoints3d_loss'] = self.compute_keypoints3d_loss(
                pred_keypoints3d, gt_keypoints3d
            )

        # kp2d
        if (
            self.loss_keypoints2d is not None
            and self.hparams.disable_inference_loss is False
        ):
            losses['keypoints2d_loss'] = self.compute_keypoints2d_loss(
                pred_keypoints3d,
                pred_cam,
                gt_keypoints2d,
                img_res=self.hparams.DATASET.IMG_RES,
                focal_length=self.hparams.DATASET.FOCAL_LENGTH,
            )

        # vertex
        if (
            self.loss_vertex is not None
            and self.hparams.disable_inference_loss is False
        ):
            losses['vertex_loss'] = self.compute_vertex_loss(
                pred_vertices, gt_vertices, has_smpl
            )

        # pose(smpl)
        if (
            self.loss_smpl_pose is not None
            and self.hparams.disable_inference_loss is False
        ):
            losses['smpl_pose_loss'] = self.compute_smpl_pose_loss(
                pred_pose, gt_pose, has_smpl
            )

        # betas(smpl)
        if (
            self.loss_smpl_betas is not None
            and self.hparams.disable_inference_loss is False
        ):
            losses['smpl_betas_loss'] = self.compute_smpl_betas_loss(
                pred_betas, gt_betas, has_smpl
            )

        # camera
        if (
            self.loss_camera is not None
            and self.hparams.disable_inference_loss is False
        ):
            losses['camera_loss'] = self.compute_camera_loss(pred_cam)

        # The coordinate origin of gt_keypoints_2d is the top left corner of the input image.
        # get GT camera & render
        with torch.no_grad():
            gt_cam_t = estimate_translation(
                gt_output['joints'],
                gt_keypoints2d,
                focal_length=self.hparams.DATASET.FOCAL_LENGTH,
                img_size=self.hparams.DATASET.IMG_RES,
                use_all_joints=True
                if '3dpw' in self.hparams.DATASET.DATASETS_AND_RATIOS
                else False,
            )

        new_cam = get_cameras(
            self.hparams.DATASET.FOCAL_LENGTH / 4,
            self.hparams.MODEL.RENDERER_GT.RENDER_RES,  # self.hparams.DATASET.IMG_RES,
            gt_cam_t,
        )

        self.neural_renderer_voge_gt.cameras = new_cam
        # part segm
        if self.loss_segm_mask is not None:
            self.neural_renderer_gt.rasterizer.cameras = new_cam
            gt_segm_mask, gt_segm_rgb = generate_part_labels(
                vertices=gt_vertices,
                faces=self.body_model_train.faces_tensor[None].expand(bs, -1, -1),
                vert_to_part=self.vert_to_part.to(gt_vertices.device),
                neural_renderer=self.neural_renderer_gt.to(gt_vertices.device),
            )

            # segm_mask loss
            pred_segm_mask = predictions['pred_segm_mask'][has_smpl == 1]
            gt_segm_mask = gt_segm_mask[has_smpl == 1]
            losses['loss_segm_mask'] = self.compute_segm_mask_loss(
                pred_segm_mask, gt_segm_mask
            )

        # coke loss
        if self.loss_contrastive is not None:
            if self.hparams.renderer_type == 'VoGE':
                # get mask & visibility based on GT render
                mask, iskpvisible, vert_coke_features = get_mask_and_visibility_voge(
                    vertices=gt_verts_down,
                    neural_renderer=self.neural_renderer_voge_gt.to(gt_vertices.device),
                    img_size=self.hparams.DATASET.IMG_RES
                    // self.hparams.MODEL.downsample_rate,
                    occ_stride=occ_stride,
                    occ_size=occ_size,
                    occ_idx=occ_idx,
                    have_occ=have_occ,
                    coke_features=coke_features
                    if self.hparams.MODEL.VOGE_SAMPLE
                    else None,
                )
                with torch.no_grad():
                    vert_orients = get_vert_orients(
                        gt_keypoints3d_smpl[:, :25], self.hparams.MODEL.N_ORIENT
                    )
                    bg_mask = 1.0 - mask
                    bg_mask[have_occ] = (
                        bg_mask
                        + F.interpolate(
                            occ_mask.unsqueeze(0),
                            (
                                self.hparams.DATASET.IMG_RES
                                // self.hparams.MODEL.downsample_rate,
                                self.hparams.DATASET.IMG_RES
                                // self.hparams.MODEL.downsample_rate,
                            ),
                        ).squeeze(0)
                    )[have_occ].float()

                    # -----------------------------------------------------------------------------

                    # proj 2d based on camera & GT vertices & features
                    vertices2d_proj = new_cam.transform_points_screen(
                        gt_verts_down, image_size=(coke_features.shape[3:1:-1],)
                    )[
                        :, :, :2
                    ]  # image_size is (W, H)

                if self.loss_contrastive is not None:
                    if isinstance(vert_coke_features, (list, tuple)):
                        loss_coke = dict()
                        for vert_coke_features_sub in vert_coke_features:
                            loss_coke_sub = self.compute_coke_loss(
                                coke_features,
                                torch.flip(vertices2d_proj, [2]),
                                targets['has_smpl'].view(-1).bool(),
                                iskpvisible,
                                self.feature_bank,
                                self.adj_mat.to(coke_features.device),
                                vert_orients=vert_orients,
                                bg_mask=bg_mask,
                                vert_coke_features=vert_coke_features_sub
                                if self.hparams.MODEL.VOGE_SAMPLE
                                else None,
                                mask=mask,
                            )
                            for k, v in loss_coke_sub.items():
                                loss_coke[k] = loss_coke.get(k, 0) + v / len(
                                    vert_coke_features
                                )
                    else:
                        loss_coke = self.compute_coke_loss(
                            coke_features,
                            torch.flip(vertices2d_proj, [2]),
                            targets['has_smpl'].view(-1).bool(),
                            iskpvisible,
                            self.feature_bank,
                            self.adj_mat.to(coke_features.device),
                            vert_orients=vert_orients,
                            bg_mask=bg_mask,
                            vert_coke_features=vert_coke_features
                            if self.hparams.MODEL.VOGE_SAMPLE
                            else None,
                            mask=mask,
                        )
                    losses.update(loss_coke)
            else:
                with torch.no_grad():
                    vertices2d_proj = new_cam.transform_points_screen(
                        gt_verts_down, image_size=(coke_features.shape[3:1:-1],)
                    )[
                        :, :, :2
                    ]  # image_size is (W, H)

                    mask, iskpvisible = get_mask_and_visibility(
                        vertices=gt_verts_down,
                        faces=self.faces_down,
                        rasterizer=self.neural_renderer_gt.rasterizer,
                    )
                    vert_orients = get_vert_orients(
                        gt_keypoints3d_smpl[:, :25], self.hparams.MODEL.N_ORIENT
                    )
                if self.loss_contrastive is not None:
                    (
                        losses['loss_contrastive'],
                        losses['loss_noise_reg'],
                    ) = self.compute_coke_loss(
                        coke_features,
                        torch.flip(vertices2d_proj, [2]),
                        targets['has_smpl'].view(-1).bool(),
                        iskpvisible,
                        self.feature_bank,
                        self.adj_mat.to(coke_features.device),
                        vert_orients=vert_orients,
                        bg_mask=bg_mask,
                    )

        # segm loss
        if (
            self.loss_part_segm is not None
            and self.hparams.disable_inference_loss is False
        ):
            # get pred cam & render
            pred_cam_t = convert_weak_perspective_to_perspective(
                pred_cam,
                focal_length=self.hparams.DATASET.FOCAL_LENGTH,
                img_res=self.hparams.DATASET.IMG_RES,
            )
            new_cam = get_cameras(
                self.hparams.DATASET.FOCAL_LENGTH,
                self.hparams.DATASET.IMG_RES,
                pred_cam_t,
            )

            if self.hparams.renderer_type == 'VoGE':
                # -----------------------------------------------------------------------------
                self.neural_renderer_pred.cameras = new_cam

                # get pred segm para based on pred vertices & render
                _, pred_segm_rgb = generate_part_labels_voge(
                    vertices=pred_vertices,
                    faces=self.body_model_train.faces_tensor[None].expand(bs, -1, -1),
                    vert_to_part=self.vert_to_part.to(gt_vertices.device),
                    neural_renderer=self.neural_renderer_pred.to(gt_vertices.device),
                )
                # -----------------------------------------------------------------------------
            else:
                self.neural_renderer_pred.rasterizer.cameras = new_cam
                _, pred_segm_rgb = generate_part_labels(
                    vertices=pred_vertices,
                    faces=self.body_model_train.faces_tensor[None].expand(bs, -1, -1),
                    vert_to_part=self.vert_to_part.to(gt_vertices.device),
                    neural_renderer=self.neural_renderer_pred.to(gt_vertices.device),
                )
            # segm loss
            losses['loss_part_segm'] = self.compute_part_segm_loss(
                pred_segm_rgb[has_smpl == 1], gt_segm_rgb[has_smpl == 1]
            )

        return losses

    def run_openpose_estimator(self, imgs, img_metas, match_gt_kp2d=True, **kwargs):
        openpose_dets = []
        if match_gt_kp2d:
            gt_keypoints_batch = kwargs['kp2ds_gt'].cpu().numpy()
            for img, img_meta, gt_keypoints in zip(
                imgs.cpu().numpy(), img_metas, gt_keypoints_batch
            ):
                img_norm_cfg = img_meta['img_norm_cfg']
                img = img.transpose((1, 2, 0))
                img_np = imdenormalize(
                    img,
                    mean=img_norm_cfg['mean'],
                    std=img_norm_cfg['std'],
                    to_bgr=img_norm_cfg['to_rgb'],
                )
                openpose_keypoints = np.array(
                    self.openpose_body_estimator(img_np)
                )  # N, 25, 3

                # match with groundtruth
                # use mean keypoint distance to decide center human
                if openpose_keypoints.any():
                    gt_keypoints = gt_keypoints[:25]
                    openpose_keypoints[openpose_keypoints[:, :, 2] == 0] = np.nan
                    joint_dists = np.nanmean(
                        np.linalg.norm(
                            openpose_keypoints[:, :, :2] - gt_keypoints[None, :, :2],
                            axis=2,
                        ),
                        axis=1,
                    )
                    joint_dists[np.isnan(joint_dists)] = float('inf')
                    eval_person_id = np.argmin(joint_dists)
                    openpose_kp = openpose_keypoints[eval_person_id]
                    openpose_kp[np.isnan(openpose_kp)] = 0.0
                else:
                    openpose_kp = np.zeros((25, 3))
                openpose_dets.append(openpose_kp)
        else:
            for img, img_meta in zip(imgs.cpu().numpy(), img_metas):
                img_norm_cfg = img_meta['img_norm_cfg']
                img = img.transpose((1, 2, 0))
                img_np = imdenormalize(
                    img,
                    mean=img_norm_cfg['mean'],
                    std=img_norm_cfg['std'],
                    to_bgr=img_norm_cfg['to_rgb'],
                )
                openpose_keypoints = np.array(
                    self.openpose_body_estimator(img_np)
                )  # N, 25, 3

                # use mean keypoint distance to decide center human
                if openpose_keypoints.any():
                    openpose_keypoints[openpose_keypoints[:, :, 2] == 0] = np.nan
                    center = np.array(img.shape[:2]) / 2
                    joint_dists = np.nanmean(
                        np.linalg.norm(
                            openpose_keypoints[:, :, :2] - center[None, None, :2],
                            axis=2,
                        ),
                        axis=1,
                    )
                    joint_dists[np.isnan(joint_dists)] = float('inf')
                    eval_person_id = np.argmin(joint_dists)
                    openpose_kp = openpose_keypoints[eval_person_id]
                    openpose_kp[np.isnan(openpose_kp)] = 0.0
                else:
                    openpose_kp = np.zeros((25, 3))
                openpose_dets.append(openpose_kp)
        openpose_dets = torch.from_numpy(np.array(openpose_dets)).type_as(imgs)
        return openpose_dets

    def get_gt_smpl(self, targets, body_model):
        gt_keypoints2d = targets['keypoints2d'].clone()

        # recover occlusion (hack: remove for mpi_inf_3dhp)
        gt_keypoints2d, _ = convert_kps(
            gt_keypoints2d,
            src='smpl_49',
            dst='h36m',
        )
        gt_keypoints2d[:, :, 2] = 1
        gt_keypoints2d, _ = convert_kps(
            gt_keypoints2d,
            src='h36m',
            dst='smpl_49',
        )

        gt_pose = targets['smpl_body_pose']
        global_orient = targets['smpl_global_orient'].view(-1, 1, 3)
        gt_pose = torch.cat((global_orient, gt_pose), dim=1).float().flatten(1)
        gt_betas = targets['smpl_betas'].float()
        gt_output = body_model(
            betas=gt_betas,
            body_pose=gt_pose[:, 3:],
            global_orient=gt_pose[:, :3],
            num_joints=gt_keypoints2d.shape[1],
        )
        gt_keypoints3d_smpl = gt_output['joints']

        # The coordinate origin of gt_keypoints_2d is
        # the top left corner of the input image.
        with torch.no_grad():
            gt_cam_t = estimate_translation(
                gt_output['joints'],
                gt_keypoints2d,
                focal_length=self.hparams.DATASET.FOCAL_LENGTH,
                img_size=self.hparams.DATASET.IMG_RES,
                use_all_joints=False,
            )

        new_cam = get_cameras(
            self.hparams.DATASET.FOCAL_LENGTH / self.hparams.MODEL.downsample_rate,
            self.hparams.MODEL.RENDERER_GT.RENDER_RES,
            gt_cam_t,
        )

        with torch.no_grad():
            kp2ds_gt = new_cam.transform_points_screen(
                gt_keypoints3d_smpl,
                image_size=(
                    (self.hparams.DATASET.IMG_RES, self.hparams.DATASET.IMG_RES),
                ),
            )[
                :, :, :2
            ]  # image_size is (W, H)

        return gt_pose, gt_betas, gt_cam_t, kp2ds_gt

    @abstractmethod
    def make_fake_data(self, predictions, requires_grad):
        pass

    @abstractmethod
    def make_real_data(self, data_batch):
        pass

    @abstractmethod
    def prepare_targets(self, data_batch):
        pass

    def forward_train(self, **kwargs):
        """Forward function for general training.

        For mesh estimation, we do not use this interface.
        """
        raise NotImplementedError(
            'This interface should not be used in '
            'current training schedule. Please use '
            '`train_step` for training.'
        )

    @abstractmethod
    def forward_test(self, img, img_metas, **kwargs):
        """Defines the computation performed at every call when testing."""
        pass


@ARCHITECTURES.register_module()
class ImageVoGEBodyModelEstimatorSE(VoGEBodyModelEstimatorSE):
    def make_fake_data(self, predictions: dict, requires_grad: bool):
        pred_cam = predictions['pred_cam']
        pred_pose = predictions['pred_pose']
        pred_betas = predictions['pred_shape']
        if requires_grad:
            fake_data = (pred_cam, pred_pose, pred_betas)
        else:
            fake_data = (pred_cam.detach(), pred_pose.detach(), pred_betas.detach())
        return fake_data

    def make_real_data(self, data_batch: dict):
        transl = data_batch['adv_smpl_transl'].float()
        global_orient = data_batch['adv_smpl_global_orient']
        body_pose = data_batch['adv_smpl_body_pose']
        betas = data_batch['adv_smpl_betas'].float()
        pose = torch.cat((global_orient, body_pose), dim=-1).float()
        real_data = (transl, pose, betas)
        return real_data

    def prepare_targets(self, data_batch: dict):
        # Image Mesh Estimator does not need extra process for ground truth
        return data_batch

    def get_openpose_det(self, img: torch.Tensor, img_metas: dict, **kwargs):
        img_norm_cfg = img_metas[0]['img_norm_cfg']
        mean = (
            torch.from_numpy(np.array(img_norm_cfg['mean']))
            .view(1, 3, 1, 1)
            .to(img.device)
        )
        std = (
            torch.from_numpy(np.array(img_norm_cfg['std']))
            .view(1, 3, 1, 1)
            .to(img.device)
        )
        img_tensor = img * std + mean
        img_tensor = F.interpolate(
            img_tensor,
            size=(self.hparams.VISUALIZER.IMG_RES, self.hparams.VISUALIZER.IMG_RES),
            mode='bilinear',
            align_corners=True,
        )
        img_tensor = img_tensor.permute(0, 2, 3, 1)

        if (
            not hasattr(self.hparams.REGISTRANT, 'use_otf_openpose')
        ) or self.hparams.REGISTRANT.use_otf_openpose:
            if not hasattr(self.hparams.REGISTRANT, 'match_gt_kp2d'):
                self.hparams.REGISTRANT.match_gt_kp2d = False
            if self.hparams.REGISTRANT.match_gt_kp2d:
                gt_pose, gt_betas, gt_cam_t, kp2ds_gt = self.get_gt_smpl(
                    kwargs, self.body_model_train
                )
                kwargs['kp2ds_gt'] = kp2ds_gt

            openpose_keypoints = self.run_openpose_estimator(
                img,
                img_metas,
                match_gt_kp2d=self.hparams.REGISTRANT.match_gt_kp2d,
                **kwargs,
            )
        return openpose_keypoints

    def forward_test(self, img: torch.Tensor, img_metas: dict, **kwargs):
        """Defines the computation performed at every call when testing."""

        if self.hparams.REGISTRANT.use_saved_coke:
            # (HACK)
            img = F.interpolate(img, (224, 224), mode='bilinear', align_corners=True)
        if self.backbone is not None:
            features = self.backbone(img)
        else:
            features = kwargs['features']

        if self.neck is not None:
            features = self.neck(features)

        if isinstance(self.head, PareHeadwCoKeNeMoAttn):
            predictions = self.head(
                features, self.feature_bank.get_feature_banks_original_order()
            )
        else:
            predictions = self.head(features)
        if self.registrant is not None:
            img_norm_cfg = img_metas[0]['img_norm_cfg']
            mean = (
                torch.from_numpy(np.array(img_norm_cfg['mean']))
                .view(1, 3, 1, 1)
                .to(img.device)
            )
            std = (
                torch.from_numpy(np.array(img_norm_cfg['std']))
                .view(1, 3, 1, 1)
                .to(img.device)
            )
            img_tensor = img * std + mean
            img_tensor = F.interpolate(
                img_tensor,
                size=(self.hparams.VISUALIZER.IMG_RES, self.hparams.VISUALIZER.IMG_RES),
                mode='bilinear',
                align_corners=True,
            )
            img_tensor = img_tensor.permute(0, 2, 3, 1)

            if (
                not hasattr(self.hparams.REGISTRANT, 'use_otf_openpose')
            ) or self.hparams.REGISTRANT.use_otf_openpose:
                if not hasattr(self.hparams.REGISTRANT, 'match_gt_kp2d'):
                    self.hparams.REGISTRANT.match_gt_kp2d = False
                if self.hparams.REGISTRANT.match_gt_kp2d:
                    gt_pose, gt_betas, gt_cam_t, kp2ds_gt = self.get_gt_smpl(
                        kwargs, self.body_model_train
                    )
                    kwargs['kp2ds_gt'] = kp2ds_gt

                openpose_keypoints = self.run_openpose_estimator(
                    img,
                    img_metas,
                    match_gt_kp2d=self.hparams.REGISTRANT.match_gt_kp2d,
                    **kwargs,
                )
                kwargs['keypoints2d'][:, :25] = openpose_keypoints

            kwargs['img_metas'] = img_metas
            if isinstance(self.registrant, NeuralSMPLFitting) or isinstance(
                self.registrant, NeuralSMPLFittingVoGE
            ):
                neural_features, clutter_bank = self.feature_bank.get_feature_banks()
                self.registrant.neural_mesh_model_voge.set_features(neural_features)
                # Mixture of VMF for clutter model
                if hasattr(self.hparams.REGISTRANT, 'mixture_path'):
                    clutter_bank = F.normalize(
                        torch.from_numpy(np.load(self.hparams.REGISTRANT.mixture_path))
                        .float()
                        .cuda(),
                        p=2,
                        dim=1,
                    )
                    self.registrant.set_clutter(clutter_bank[:, :])  #
                else:
                    self.registrant.set_clutter(clutter_bank)

            if self.hparams.renderer_type == 'VoGE':
                ret = self.run_registration_test_voge(
                    predictions,
                    kwargs,
                    focal_length=self.hparams.DATASET.FOCAL_LENGTH,
                    img_res=self.hparams.DATASET.IMG_RES,
                    img=img_tensor,
                )
            else:
                raise NotImplementedError()

            predictions = dict(
                pred_pose=pytorch3d.transforms.axis_angle_to_matrix(
                    ret['opt_pose'].view(-1, 24, 3)
                ),
                pred_shape=ret['opt_betas'],
                pred_cam=convert_perspective_to_weak_perspective(
                    ret['opt_cam_t'],
                    focal_length=self.hparams.DATASET.FOCAL_LENGTH,
                    img_res=self.hparams.DATASET.IMG_RES,
                ),
                pred_segm_mask=predictions.get('pred_segm_mask', None),
                vertices2d_det=ret['vertices2d_det'],
            )

        pred_pose = predictions['pred_pose']
        pred_betas = predictions['pred_shape']
        pred_cam = predictions['pred_cam']

        pred_output = self.body_model_test(
            betas=pred_betas,
            body_pose=pred_pose[:, 1:],
            global_orient=pred_pose[:, 0].unsqueeze(1),
            pose2rot=False,
        )

        pred_vertices = pred_output['vertices']
        pred_keypoints_3d = pred_output['joints']
        all_preds = {}
        all_preds['keypoints_3d'] = pred_keypoints_3d.detach().cpu().numpy()
        all_preds['smpl_pose'] = pred_pose.detach().cpu().numpy()
        all_preds['smpl_beta'] = pred_betas.detach().cpu().numpy()
        all_preds['camera'] = pred_cam.detach().cpu().numpy()
        image_path = [img_meta['image_path'] for img_meta in img_metas]
        # all_preds['vertices'] = pred_vertices.detach().cpu().numpy()
        all_preds['image_path'] = image_path
        all_preds['image_idx'] = kwargs['sample_idx']
        if 'pred_segm_mask' in predictions and self.hparams.get('save_partseg', False):
            all_preds['pred_segm_mask'] = (
                predictions['pred_segm_mask'].detach().cpu().numpy()
            )
        if 'vertices2d_det' in predictions:
            all_preds['vertices2d_det'] = (
                predictions['vertices2d_det'].detach().cpu().numpy()
            )

        meta_info = {}
        meta_info['keypoints_2d'] = kwargs['keypoints2d'].cpu().numpy()
        for img_meta in img_metas:
            for occ_info in [
                'occ_size',
                'occ_stride',
                'occ_idx',
                'texture_file',
                'texture_crop_tl',
            ]:
                if occ_info in img_meta:
                    meta_info.setdefault(occ_info, []).append(img_meta[occ_info])
        all_preds['meta_info'] = meta_info

        return all_preds
