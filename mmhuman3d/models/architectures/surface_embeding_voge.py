from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple, Union
from yacs.config import CfgNode as CN
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import pytorch3d
from pytorch3d.transforms.rotation_conversions import matrix_to_axis_angle
import ipdb
from mmcv import imdenormalize
from mmcv.runner import get_dist_info
from mmhuman3d.core.conventions.keypoints_mapping import convert_kps, get_keypoint_idx
from mmhuman3d.core.conventions import constants

from mmhuman3d.utils.dist_utils import allgather_tensor
from mmhuman3d.models.utils import FitsDict
from mmhuman3d.models.backbones import VisionTransformer
from mmhuman3d.utils.geometry import (
    batch_rodrigues,
    estimate_translation,
    project_points,
    rotation_matrix_to_angle_axis,
    convert_weak_perspective_to_perspective,
    convert_perspective_to_weak_perspective,
    rotate_aroundy
)
from mmhuman3d.utils.image_utils import (
    generate_part_labels, 
    generate_part_labels_voge,
    get_vert_to_part,
    get_mask_and_visibility,
    get_mask_and_visibility_voge,
    get_vert_orients)
from mmhuman3d.utils.neuralsmpl_utils import get_detected_2d_vertices
from mmhuman3d.utils.neural_renderer import (
    build_neural_renderer,
    get_blend_params,
    get_cameras)
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
from mmhuman3d.utils.neural_renderer_voge import build_neural_renderer_voge
from mmhuman3d.utils.vis_utils import SMPLVisualizer
from mmhuman3d.data.datasets.pipelines.transforms import _flip_smpl_pose_batch

from openpose_pytorch import torch_openpose
from .base_architecture import BaseArchitecture
from ..registrants import NeuralSMPLFittingVoGE

from loguru import logger

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

@ARCHITECTURES.register_module()
class SurfaceEmbedingModuleVoGE(BaseArchitecture, metaclass=ABCMeta):
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

    def __init__(self,
                 backbone: Optional[Union[dict, None]] = None,
                 neck: Optional[Union[dict, None]] = None,
                 head: Optional[Union[dict, None]] = None,
                 feature_bank: Optional[Union[dict, None]] = None,
                 disc: Optional[Union[dict, None]] = None,
                 registrant: Optional[Union[dict, None]] = None,
                 body_model_train: Optional[Union[dict, None]] = None,
                 body_model_test: Optional[Union[dict, None]] = None,
                 convention: Optional[str] = 'human_data',
                 loss_contrastive: Optional[Union[dict, None]] = None,
                 loss_noise_reg: Optional[Union[dict, None]] = None,
                 init_cfg: Optional[Union[list, dict, None]] = None,
                 hparams: Optional[Union[dict, None]] = None):
        super(SurfaceEmbedingModuleVoGE, self).__init__(init_cfg)
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
        self.registrant = build_registrant(registrant)
        if registrant is not None:
            self.fits = 'registration'
            self.fits_dict = FitsDict(fits='static')

        self.loss_contrastive = build_loss(loss_contrastive)

        set_requires_grad(self.body_model_train, False)
        set_requires_grad(self.body_model_test, False)

        self.vert_to_part = get_vert_to_part()
        self.neural_renderer_gt = build_neural_renderer_voge(hparams.MODEL.RENDERER_GT)
        # self.neural_renderer_gt1 = build_neural_renderer_voge(hparams.MODEL.RENDERER_GT1)
        # self.neural_renderer_pred = build_neural_renderer_voge(hparams.MODEL.RENDERER_PRED)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_mesh_info()

        if registrant is not None:
            self.fits = 'registration'
            self.fits_dict = FitsDict(fits='static')
            assert isinstance(self.registrant, NeuralSMPLFittingVoGE)
            self.registrant.set_mesh_info(ds_indices=self.ds_indices, 
                                            faces_down=self.faces_down)
            self.openpose_body_estimator = torch_openpose.torch_openpose('body_25', 'third_party/pytorch_openpose_body_25/model/body_25.pth')
            if self.hparams.REGISTRANT.RUN_VISUALIZER:
                visualizer = SMPLVisualizer(self.body_model_test, 'cuda', None,
                            image_size=(self.hparams.VISUALIZER.IMG_RES, self.hparams.VISUALIZER.IMG_RES),
                            point_light_location=((0,0,-3.0),)) 
                self.registrant.set_visualizer(visualizer)
                self.writer = SummaryWriter(log_dir=self.hparams.DEBUG.DEBUG_LOG_DIR)
            else:
                self.writer = None

        if not hasattr(self.hparams.MODEL, 'VOGE_SAMPLE'):
            self.hparams.MODEL.VOGE_SAMPLE = False

        if hasattr(self.hparams.DEBUG, 'eval_occ_seg') and self.hparams.DEBUG.eval_occ_seg:
            # from torchmetrics import Dice
            # from mmhuman3d.utils.eval_utils import Dice
            # self.dice_avg_meter = Dice(average='micro', threshold=-1)
            from mmhuman3d.utils.eval_utils import IOU
            self.dice_avg_meter = IOU(average='micro', threshold=-1.0)
            self.count = 0

    def load_mesh_info(self):
        mesh_sample_data = np.load(self.hparams.mesh_sample_param_path)
        mesh_sample_data = dict(mesh_sample_data)
        self.ds_indices = mesh_sample_data['indices']
        n = len(self.ds_indices)
        
        self.faces_down = torch.tensor(mesh_sample_data['faces_downsampled'], device=self.device)
        adj_mat = torch.zeros((n, n)).long()
        for face in self.faces_down:
            x1, x2, x3 = sorted(face)
            adj_mat[x1, x2] = 1
            adj_mat[x1, x3] = 1
            adj_mat[x2, x3] = 1
        self.adj_mat = (adj_mat + adj_mat.transpose(1,0)).unsqueeze(0)
        # n_dist = 1
        # for i in range(n_dist):
        #     self.adj_mat = self.adj_mat @ self.adj_mat
        

    def train_step(self, data_batch, optimizer, **kwargs):
        """Train step function.

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
        if self.backbone is not None:
            img = data_batch['img']
            if isinstance(self.backbone, VisionTransformer) or \
            (hasattr(self.backbone, 'module') and isinstance(self.backbone.module, VisionTransformer)):
                features = self.backbone(img, get_vit_features=True, layer_idx=self.hparams.MODEL.layer_idx)
            else:
                features = self.backbone(img)
        else:
            features = data_batch['features']

        if self.neck is not None:
            features = self.neck(features)
        if self.head is not None:
            predictions = self.head(features)
        else:
            predictions = dict(coke_features=features)
        targets = self.prepare_targets(data_batch)

        # optimize discriminator (if have)
        if self.disc is not None:
            self.optimize_discrinimator(predictions, data_batch, optimizer)

        if self.registrant is not None:
            targets = self.run_registration(predictions, targets)
        losses = self.compute_losses(predictions, targets)
        # optimizer generator part
        if self.disc is not None:
            adv_loss = self.optimize_generator(predictions)
            losses.update(adv_loss)

        loss, log_vars = self._parse_losses(losses)
        if optimizer is not None:
            if self.backbone is not None:
                optimizer['backbone'].zero_grad()
            if self.neck is not None:
                optimizer['neck'].zero_grad()
            if self.head is not None:
                optimizer['head'].zero_grad()
            loss.backward()
            if self.backbone is not None:
                optimizer['backbone'].step()
            if self.neck is not None:
                optimizer['neck'].step()
            if self.head is not None:
                optimizer['head'].step()
        # logger.debug(self.backbone.module.conv1.weight.grad.mean().item())
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))
        return outputs

    def compute_coke_loss(self, coke_features,
                keypoint_positions,
                has_smpl,
                iskpvisible,
                feature_bank,
                adj_mat,
                vert_orients=None,
                bg_mask=None,
                vert_coke_features=None):
        """
            coke_features: N, 128, res/4. res/4
        """
        loss_contrastive, loss_noise_reg = self.loss_contrastive(coke_features,
                keypoint_positions,
                has_smpl,
                iskpvisible,
                feature_bank,
                adj_mat,
                vert_orients=vert_orients,
                bg_mask=bg_mask,
                vert_coke_features=vert_coke_features)
        return loss_contrastive, loss_noise_reg

    def get_gt_smpl(self, targets, body_model):
        gt_keypoints2d = targets['keypoints2d'].clone()
        # recover occlusion (hack: remove for mpi_inf_3dhp)
        gt_keypoints2d, _ = convert_kps(gt_keypoints2d,
                                    src='smpl_49',
                                    dst='h36m',)
        gt_keypoints2d[:, :, 2] = 1
        gt_keypoints2d, _ = convert_kps(gt_keypoints2d,
                                    src='h36m',
                                    dst='smpl_49',)

        gt_pose = targets['smpl_body_pose']
        global_orient = targets['smpl_global_orient'].view(-1, 1, 3)
        gt_pose = torch.cat((global_orient, gt_pose), dim=1).float().flatten(1)
        gt_betas = targets['smpl_betas'].float()
        # gt_pose N, 72
        # if self.body_model_train is not None:


        gt_output = body_model(
            betas=gt_betas,
            body_pose=gt_pose[:, 3:],
            global_orient=gt_pose[:, :3],
            num_joints=gt_keypoints2d.shape[1])
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
                # True if '3dpw' in self.hparams.DATASET.DATASETS_AND_RATIOS else False,
            )

        new_cam = get_cameras(
            self.hparams.DATASET.FOCAL_LENGTH/self.hparams.MODEL.downsample_rate, 
            self.hparams.MODEL.RENDERER_GT.RENDER_RES, # self.hparams.DATASET.IMG_RES, 
            gt_cam_t)
        
        with torch.no_grad():
            kp2ds_gt = new_cam.transform_points_screen(gt_keypoints3d_smpl, 
                                    image_size=((self.hparams.DATASET.IMG_RES, self.hparams.DATASET.IMG_RES), )
                                    )[:, :, :2] # image_size is (W, H)
            
        return gt_pose, gt_betas, gt_cam_t, kp2ds_gt

    def compute_smpl_mask(self, gt_pose, gt_betas, gt_cam_t, gender=None):
        # gt_pose N, 72
        if self.body_model_train is not None:
            if gender is not None:
                gt_output = self.body_model_train(
                    betas=gt_betas,
                    body_pose=gt_pose[:, 3:],
                    global_orient=gt_pose[:, :3],
                    gender=gender)
            else:
                gt_output = self.body_model_train(
                    betas=gt_betas,
                    body_pose=gt_pose[:, 3:],
                    global_orient=gt_pose[:, :3],)
            gt_vertices = gt_output['vertices']
            gt_keypoints3d_smpl = gt_output['joints']
        gt_verts_down = gt_vertices[:, self.ds_indices]
        new_cam = get_cameras(
            self.hparams.DATASET.FOCAL_LENGTH/4, # (TODO: check if this is correct)
            self.hparams.MODEL.RENDERER_GT.RENDER_RES, # self.hparams.DATASET.IMG_RES, 
            gt_cam_t)
        self.neural_renderer_gt.cameras = new_cam
        # self.neural_renderer_gt1.cameras = new_cam
            
        mask, iskpvisible, vert_coke_features = get_mask_and_visibility_voge(
            vertices=gt_verts_down,
            neural_renderer=self.neural_renderer_gt.to(gt_vertices.device),
            image_size=self.hparams.DATASET.IMG_RES,
            downsample_rate=self.hparams.MODEL.downsample_rate,
        )
        
        return mask
            
    def compute_losses(self, predictions: dict, targets: dict):
        """Compute losses."""
        bs = targets['keypoints3d'].shape[0]

        gt_keypoints3d = targets['keypoints3d']
        gt_keypoints2d = targets['keypoints2d']

        # features = predictions['features']
        coke_features = predictions['coke_features']
        
        has_smpl = targets['has_smpl'].view(-1)
        gt_pose = targets['smpl_body_pose']
        global_orient = targets['smpl_global_orient'].view(-1, 1, 3)
        gt_pose = torch.cat((global_orient, gt_pose), dim=1).float().flatten(1)
        gt_betas = targets['smpl_betas'].float()

        # gt_pose N, 72
        if self.body_model_train is not None:
            gt_output = self.body_model_train(
                betas=gt_betas,
                body_pose=gt_pose[:, 3:],
                global_orient=gt_pose[:, :3],
                num_joints=gt_keypoints2d.shape[1])
            gt_vertices = gt_output['vertices']
            gt_keypoints3d_smpl = gt_output['joints']

        gt_verts_down = gt_vertices[:, self.ds_indices]

        losses = {}
        # The coordinate origin of gt_keypoints_2d is
        # the top left corner of the input image.
        with torch.no_grad():
            gt_cam_t = estimate_translation(
                gt_output['joints'],
                gt_keypoints2d, 
                focal_length=self.hparams.DATASET.FOCAL_LENGTH,
                img_size=self.hparams.DATASET.IMG_RES,
                use_all_joints=True if '3dpw' in self.hparams.DATASET.DATASETS_AND_RATIOS else False,
            )
        new_cam = get_cameras(
            self.hparams.DATASET.FOCAL_LENGTH/4, 
            self.hparams.MODEL.RENDERER_GT.RENDER_RES, # self.hparams.DATASET.IMG_RES, 
            gt_cam_t)
        self.neural_renderer_gt.cameras = new_cam
        # self.neural_renderer_gt1.cameras = new_cam

        with torch.no_grad():
            vertices2d_proj = new_cam.transform_points_screen(gt_verts_down, 
                                    image_size=(coke_features.shape[3:1:-1], )
                                    )[:, :, :2] # image_size is (W, H)
            
        mask, iskpvisible, vert_coke_features = get_mask_and_visibility_voge(
            vertices=gt_verts_down,
            neural_renderer=self.neural_renderer_gt.to(gt_vertices.device),
            image_size=self.hparams.DATASET.IMG_RES,
            coke_features=coke_features if self.hparams.MODEL.VOGE_SAMPLE else None,
            downsample_rate=self.hparams.MODEL.downsample_rate,
        )
        with torch.no_grad():
            vert_orients = get_vert_orients(gt_keypoints3d_smpl[:, :25], self.hparams.MODEL.N_ORIENT)

            bg_mask = 1. - mask

        # if (bg_mask.view(mask.shape[0], -1).sum(dim=1) < 1e-6).any():
        #     isInvalid = (bg_mask.view(mask.shape[0], -1).sum(dim=1) < 1e-6)
        #     for i in range(len(isInvalid)):
        #         if isInvalid[i]:
        #             logger.debug(targets['img_metas'][i]['image_path'])
        losses['loss_contrastive'], losses['loss_noise_reg'] = \
                self.compute_coke_loss(coke_features, 
                    torch.flip(vertices2d_proj, [2]),
                    targets['has_smpl'].view(-1).bool(),
                    iskpvisible,
                    self.feature_bank,
                    self.adj_mat.to(coke_features.device),
                    vert_orients=vert_orients,
                    bg_mask=bg_mask,
                    vert_coke_features=vert_coke_features if self.hparams.MODEL.VOGE_SAMPLE else None)

        return losses

    def run_registration_test(
            self,
            predictions: dict,
            targets: dict,
            threshold: Optional[float] = 10.0,
            focal_length: Optional[float] = 5000.0,
            img_res: Optional[Union[Tuple[int], int]] = 224,
            img: Optional[torch.Tensor]=None) -> dict:
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
        dataset_name = [meta['dataset_name'] for meta in img_metas
                        ]  # name of the dataset the image comes from

        indices = targets['sample_idx'].squeeze()
        if 'is_flipped' not in targets:
            is_flipped = torch.zeros_like(targets['sample_idx']).bool()
        else:
            is_flipped = targets['is_flipped'].squeeze().bool(
            )  # flag that indicates whether image was flipped
        # during data augmentation
        rot_angle = targets['rotation'].squeeze(
        )  # rotation angle used for data augmentation Q

        if self.hparams.REGISTRANT.use_other_init:
            pred_rotmat = targets['poses_init'].float()
            pred_betas = targets['betas_init'].float()
            pred_cam = targets['cameras_init'].float()
            keypoinst3d = targets['keypoints3d_init'].float()
        else:
            pred_rotmat = predictions['pred_pose'].detach().clone()
            pred_betas = predictions['pred_shape'].detach().clone()
            pred_cam = predictions['pred_cam'].detach().clone()

        if (hasattr(self.hparams.MODEL, 'NON_STANDARD_WEAK_CAM')) and self.hparams.MODEL.NON_STANDARD_WEAK_CAM:
            pred_cam[:,1:] = pred_cam[:, 1:]/pred_cam[:, 0:1]

        pred_cam_t = convert_weak_perspective_to_perspective(
            pred_cam,
            focal_length=focal_length,
            img_res=img_res,
        )
        if 'pred_segm_mask' in predictions:
            pred_segm_mask = predictions['pred_segm_mask'].detach().clone()
        elif self.hparams.REGISTRANT.use_saved_partseg:
            pred_segm_mask = targets['pred_segm_mask']
            if pred_segm_mask.shape[-1] != img_res//4:
                pred_segm_mask = F.interpolate(pred_segm_mask, (img_res//4, img_res//4), mode='bilinear', align_corners=True)
        else:
            pred_segm_mask = None

        if 'mask' in targets:
            gt_mask = targets['mask']
            if gt_mask.shape[-1] != img_res//4:
                gt_mask = F.interpolate(gt_mask.unsqueeze(1), 
                    (img_res//4, img_res//4), mode='nearest').squeeze(1).long()
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
        tl = bbox_xywh[:, :2] + 0.5*bbox_xywh[:, 2:4] - 0.5 * bbox_xywh[:, 2:4].max(dim=1, keepdim=True)[0]
        bbox_xyxy = torch.cat((bbox_xywh[:, :2]-tl, bbox_xywh[:, :2]+bbox_xywh[:, 2:4]-tl), dim=1)
        scale = bbox_xywh[:, 2:4].max(dim=1)[0] / self.hparams.DATASET.IMG_RES
        bbox_xyxy = bbox_xyxy / scale[:, None]
        vertices2d_det, vertices2d_det_conf = get_detected_2d_vertices(coke_features,
            bbox_xyxy, 
            self.registrant.neural_mesh_model_voge.features, 
            self.hparams.MODEL.downsample_rate, 
            n_orient=self.registrant.hparams.n_orient)
        vertices2d_det = torch.cat((vertices2d_det, vertices2d_det_conf.unsqueeze(2)), dim=2)

        gt_keypoints_2d = targets['keypoints2d'].float()
        num_keypoints = gt_keypoints_2d.shape[1]
        has_smpl = targets['has_smpl'].view(
            -1).bool()  # flag that indicates whether SMPL parameters are valid
        batch_size = has_smpl.shape[0]
        device = has_smpl.device

        # Get inital fits from the prediction
        opt_pose = matrix_to_axis_angle(pred_rotmat).flatten(1)
        opt_pose = opt_pose.to(device)
        opt_betas = pred_betas.to(device)
        opt_cam_t = pred_cam_t.clone().to(device)

        opt_output = self.body_model_train(
            betas=opt_betas,
            body_pose=opt_pose[:, 3:],
            global_orient=opt_pose[:, :3])

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
                predicted_map=coke_features)
        opt_loss = loss_dict['total_loss']

        init_pose = opt_pose.clone()
        init_betas = opt_betas.clone()
        init_cam_t = opt_cam_t.clone()

        def _optimization(gt_keypoints_2d_orig, vertices2d_det, pred_segm_mask, 
                          gt_mask, coke_features, opt_pose, opt_betas, opt_cam_t):
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
                        init_transl=opt_cam_t, # only correct when Rotation is None
                        init_body_pose=opt_pose[:, 3:],
                        init_betas=opt_betas,
                        img=img,
                        img_meta=img_metas,
                        return_joints=True,
                        return_verts=True,
                        return_losses=True)
            
            new_opt_vertices = registrant_output['vertices']
            new_opt_joints = registrant_output['joints']

            new_opt_global_orient = registrant_output['global_orient']
            new_opt_body_pose = registrant_output['body_pose']
            new_opt_pose = torch.cat([new_opt_global_orient, new_opt_body_pose],
                                    dim=1)

            new_opt_betas = registrant_output['betas']
            new_opt_cam_t = registrant_output['transl']
            new_opt_loss = registrant_output['total_loss']
            return new_opt_loss, new_opt_vertices, new_opt_joints, \
                    new_opt_pose, new_opt_betas, new_opt_cam_t

        def _update(opt_loss, new_opt_loss, 
                    opt_vertices, new_opt_vertices, 
                    opt_joints, new_opt_joints,
                    opt_pose, new_opt_pose, 
                    opt_betas, new_opt_betas, 
                    opt_cam_t, new_opt_cam_t):
            # Will update the dictionary for the examples where the new loss
            # is less than the current one
            update = (new_opt_loss < opt_loss)

            opt_loss[update] = new_opt_loss[update]
            opt_vertices[update, :] = new_opt_vertices[update, :]
            opt_joints[update, :] = new_opt_joints[update, :]
            opt_pose[update, :] = new_opt_pose[update, :]
            opt_betas[update, :] = new_opt_betas[update, :]
            opt_cam_t[update, :] = new_opt_cam_t[update, :]

            # Replace extreme betas with zero betas
            opt_betas[(opt_betas.abs() > 3).any(dim=-1)] = 0.

        global_orient_rot = rotate_aroundy(init_pose[:, :3], 180)
        init_pose_rotflip = _flip_smpl_pose_batch(torch.cat([global_orient_rot, init_pose[:, 3:]], dim=1))
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
                predicted_map=coke_features)
        opt_loss_rot = loss_dict_rot['total_loss'] # loss for initial prediction
        update = (opt_loss_rot < opt_loss)
        # opt_pose[update, :3] = global_orient_rot[update]
        opt_pose[update] = init_pose_rotflip[update]
        opt_loss[update] = opt_loss_rot[update]

        # Evaluate GT SMPL parameters
        if hasattr(self.hparams.REGISTRANT, 'use_gt_smpl') and self.hparams.REGISTRANT.use_gt_smpl:
            gt_pose, gt_betas, gt_cam_t, _ = self.get_gt_smpl(targets, self.body_model_train)
            with torch.no_grad():
                loss_dict_gt = self.registrant.evaluate(
                    global_orient=gt_pose[:, :3],
                    body_pose=gt_pose[:, 3:],
                    betas=gt_betas,
                    transl=gt_cam_t,
                    keypoints2d=gt_keypoints_2d_orig[:, :, :2],
                    keypoints2d_conf=gt_keypoints_2d_orig[:, :, 2],
                    vertices2d=vertices2d_det[:, :, :2],
                    vertices2d_conf=vertices2d_det[:, :, 2],
                    reduction_override='none',
                    pred_segm_mask=pred_segm_mask,
                    predicted_map=coke_features)
            opt_loss_gt = loss_dict_gt['total_loss'] # loss for initial prediction
            update = (opt_loss_gt < opt_loss)
            # opt_pose[update, :3] = global_orient_rot[update]
            opt_pose[update] = gt_pose[update]
            opt_loss[update] = opt_loss_gt[update]
            opt_cam_t[update] = gt_cam_t[update]
            opt_pose = gt_pose
            opt_cam_t = gt_cam_t
            opt_betas = gt_betas
            if True:
                targets['opt_cam_t'] = opt_cam_t
                targets['opt_pose'] = opt_pose
                targets['opt_betas'] = opt_betas
            # return targets
        
        self.registrant.set_summary_writer(self.writer)
        # TODO: support HMR+

        hypotheses = [
            (init_pose.clone(), init_betas.clone(), init_cam_t.clone()),
        ]
        if self.hparams.REGISTRANT.get('optimize_twoside', False):
            hypotheses.append(
                (init_pose_rotflip, init_betas.clone(), init_cam_t.clone())
            )
        for _init_pose, _init_betas, _init_cam_t in hypotheses:

            new_opt_loss, new_opt_vertices, new_opt_joints, \
                new_opt_pose, new_opt_betas, new_opt_cam_t \
                    = _optimization(gt_keypoints_2d_orig, vertices2d_det, pred_segm_mask, 
                        gt_mask, coke_features, 
                        _init_pose, 
                        _init_betas, 
                        _init_cam_t)
            
            _update(
                opt_loss, new_opt_loss,
                opt_vertices, new_opt_vertices,
                opt_joints, new_opt_joints,
                opt_pose, new_opt_pose,
                opt_betas, new_opt_betas,
                opt_cam_t, new_opt_cam_t
            )


        # Replace extreme betas with zero betas
        opt_betas[(opt_betas.abs() > 3).any(dim=-1)] = 0.

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


        if hasattr(self.hparams.DEBUG, 'eval_occ_seg') and self.hparams.DEBUG.eval_occ_seg:
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
                    predicted_map=coke_features,
                    return_occ_mask=True)
            targets['clutter_scores'] = loss_dict['clutter_scores']
            targets['object_scores'] = loss_dict['object_scores']

        if hasattr(self.hparams.REGISTRANT, 'debug_feature_map') and self.hparams.REGISTRANT.debug_feature_map:
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
                    predicted_map=coke_features,
                    return_occ_mask=True)
            targets['occ_masks'] = loss_dict['occ_masks']
            targets['clutter_scores'] = loss_dict['clutter_scores']
            targets['object_scores'] = loss_dict['object_scores']
            targets['coke_features'] = coke_features

        return targets

    def forward_train(self, **kwargs):
        """Forward function for general training.

        For mesh estimation, we do not use this interface.
        """
        raise NotImplementedError('This interface should not be used in '
                                  'current training schedule. Please use '
                                  '`train_step` for training.')

    def forward_test(self, img: torch.Tensor, img_metas: dict, **kwargs):
        """Defines the computation performed at every call when testing."""

        if self.backbone is not None:
            if isinstance(self.backbone, VisionTransformer) or \
            (hasattr(self.backbone, 'module') and isinstance(self.backbone.module, VisionTransformer)):
                features = self.backbone(img, get_vit_features=True, layer_idx=self.hparams.MODEL.layer_idx)
            else:
                features = self.backbone(img)
        else:
            features = kwargs['features']

        if self.neck is not None:
            features = self.neck(features)
        if self.head is not None:
            predictions = self.head(features)
        else:
            predictions = dict(coke_features=features)
        
        if self.registrant is not None:
            # img_nps = []
            img_norm_cfg = img_metas[0]['img_norm_cfg']
            mean = torch.from_numpy(np.array(img_norm_cfg['mean'])).view(1, 3, 1, 1).to(img.device)
            std = torch.from_numpy(np.array(img_norm_cfg['std'])).view(1, 3, 1, 1).to(img.device)
            img_tensor = img * std + mean
            img_tensor = F.interpolate(img_tensor, size=(self.hparams.VISUALIZER.IMG_RES, self.hparams.VISUALIZER.IMG_RES), mode='bilinear', align_corners=True)
            img_tensor = img_tensor.permute(0, 2, 3, 1)
            # for im in img.cpu().numpy():
            #     im = im.transpose((1, 2, 0))
            #     img_np = imdenormalize(im, mean=img_norm_cfg['mean'], 
            #                             std=img_norm_cfg['std'], 
            #                             to_bgr=img_norm_cfg['to_rgb'])
            #     img_np = img_np.astype(np.uint8)
            #     img_nps.append(img_np)
            # img_tensor = torch.from_numpy(np.stack(img_nps))

            if (not hasattr(self.hparams.REGISTRANT, 'use_otf_openpose')) or self.hparams.REGISTRANT.use_otf_openpose:
                gt_pose, gt_betas, gt_cam_t, kp2ds_gt = self.get_gt_smpl(kwargs, self.body_model_train)
                kwargs['kp2ds_gt'] = kp2ds_gt
                # kwargs['keypoints2d'][:, :25, :2] = kp2ds_gt[:, :25]
                # kwargs['keypoints2d'][:, :25, 2] = 1.0
                openpose_keypoints = self.run_openpose_estimator(img, img_metas, **kwargs)
                kwargs['keypoints2d'][:, :25] = openpose_keypoints
            kwargs['img_metas'] = img_metas

            assert isinstance(self.registrant, NeuralSMPLFittingVoGE)

            neural_features, clutter_bank = self.feature_bank.get_feature_banks()
            self.registrant.neural_mesh_model_voge.set_features(neural_features)

            # Mixture of VMF for clutter model
            if hasattr(self.hparams.REGISTRANT, 'mixture_path'):
                clutter_bank = F.normalize(
                            torch.from_numpy(np.load(self.hparams.REGISTRANT.mixture_path)).float().cuda(), 
                            p=2, dim=1)
                self.registrant.set_clutter(clutter_bank[:, :]) # 
            else:
                self.registrant.set_clutter(clutter_bank)
            
            ret = self.run_registration_test(predictions, kwargs,
                focal_length=self.hparams.DATASET.FOCAL_LENGTH, 
                img_res=self.hparams.DATASET.IMG_RES,
                img=img_tensor)
            # import ipdb; ipdb.set_trace()
            if hasattr(self.hparams.DEBUG, 'eval_occ_seg') and self.hparams.DEBUG.eval_occ_seg:
                gt_pose, gt_betas, gt_cam_t, _ = self.get_gt_smpl(kwargs, self.body_model_train)
                with torch.no_grad():
                    masks_smpl = self.compute_smpl_mask(gt_pose, gt_betas, gt_cam_t, gender=kwargs['gender'][:, 0])
                    bg_masks_gt = 1. - masks_smpl
                if 'have_occ' in kwargs:
                    have_occ = kwargs['have_occ'] # (B,) BoolTensor
                    # if have_occ:
                    occ_mask = kwargs['occ_mask']
                    bg_masks_gt[have_occ] = (bg_masks_gt[have_occ].bool() + F.interpolate(
                                occ_mask[have_occ].unsqueeze(1), 
                                (self.hparams.DATASET.IMG_RES//self.hparams.MODEL.downsample_rate, 
                                self.hparams.DATASET.IMG_RES//self.hparams.MODEL.downsample_rate)
                                ).squeeze(1).bool()
                        ).float()
                    
                distributed = torch.distributed.is_initialized()
                if distributed:
                    img_nps = []
                    img_norm_cfg = img_metas[0]['img_norm_cfg']
                    for im in allgather_tensor(img).cpu().numpy():
                        im = im.transpose((1, 2, 0))
                        img_np = imdenormalize(im, mean=img_norm_cfg['mean'], 
                                                std=img_norm_cfg['std'], 
                                                to_bgr=img_norm_cfg['to_rgb'])
                        img_np = img_np.astype(np.uint8)
                        img_nps.append(img_np)

                    clutter_scores = allgather_tensor(ret['clutter_scores'])
                    object_scores = allgather_tensor(ret['object_scores'])
                    bg_masks_gt = allgather_tensor(bg_masks_gt)
                    masks_smpl = allgather_tensor(masks_smpl)
                    self.dice_avg_meter.update((clutter_scores[masks_smpl>0].cpu().numpy() < object_scores[masks_smpl>0].cpu().numpy()).astype(np.float32), 
                                                bg_masks_gt[masks_smpl>0].int().cpu().numpy())

                    rank, _ = get_dist_info()
                    if rank == 0:
                        os.makedirs(self.hparams.DEBUG.save_dir, exist_ok=True)
                        np.savez(f'{self.hparams.DEBUG.save_dir}/out_{self.count}.npz', 
                                pred=clutter_scores.cpu().numpy(), 
                                obj_scores=object_scores.cpu().numpy(),
                                gt=bg_masks_gt.cpu().numpy(), 
                                fg_mask=masks_smpl.cpu().numpy(),
                                img=img_nps)
                        self.count += 1
                    # clutter_scores = ret['clutter_scores']
                    # self.dice_avg_meter.update(clutter_scores, bg_masks_gt.int()) # torchmetrics

            # debug
            if hasattr(self.hparams.REGISTRANT, 'debug_occ_mask') and self.hparams.REGISTRANT.debug_occ_mask:
                from mmhuman3d.utils.vis_utils import (visualize_activation, 
                                    get_fg_activations, get_bg_activations,
                                    get_bg_activations_mixture)
                import uuid
                import os.path as osp
                id_str = uuid.uuid4().hex
                img_nps = []
                for im, img_meta in zip(img.cpu().numpy(), img_metas):
                    img_norm_cfg = img_meta['img_norm_cfg']
                    im = im.transpose((1, 2, 0))
                    img_np = imdenormalize(im, mean=img_norm_cfg['mean'], 
                                            std=img_norm_cfg['std'], 
                                            to_bgr=img_norm_cfg['to_rgb'])
                    img_np = img_np.astype(np.uint8)
                    img_nps.append(img_np)
                debug_info = dict(imgs=img_nps, occ_masks=ret['occ_masks'].cpu().numpy(), 
                            clutter_scores=ret['clutter_scores'].cpu().numpy(), 
                            object_scores=ret['object_scores'].cpu().numpy(),
                            coke_features=ret['coke_features'].cpu().numpy(),)
                os.makedirs(self.hparams.DEBUG.save_dir_vis, exist_ok=True)
                np.savez(osp.join(self.hparams.DEBUG.save_dir_vis, id_str), {**debug_info, **img_meta})
                visualize_activation(osp.join(self.hparams.DEBUG.save_dir_vis, id_str+'.png'), debug_info)

                get_fg_activations(torch.from_numpy(debug_info['coke_features']), 
                    # self.feature_bank.memory[:self.feature_bank.num_pos*self.feature_bank.num_orient].cpu(), 
                    self.feature_bank.get_fg_feature_banks().detach().cpu(),
                    self.hparams.REGISTRANT.n_vert, 
                    osp.join(self.hparams.DEBUG.save_dir_vis, id_str+'_fg_acts.png'), 
                    n_orient=self.hparams.REGISTRANT.n_orient)
                # get_bg_activations(torch.from_numpy(debug_info['coke_features']), 
                #     self.feature_bank.memory[self.feature_bank.num_pos*self.feature_bank.num_orient:].cpu(), 
                #     osp.join(self.hparams.DEBUG.save_dir_vis, id_str+'_bg_acts.png')
                #     )
                # bg_feats_mixture = torch.from_numpy(np.load(osp.join(self.hparams.DEBUG.save_dir_vis, 'bg_feats_mixture.npy'))).float()
                # bg_feats_mixture = torch.from_numpy(np.load(osp.join(self.hparams.DEBUG.save_dir_vis, '..','bg_feats_mixture_vmfsoft_nc15_10x.npy'))).float()
                if hasattr(self.hparams.REGISTRANT, 'mixture_path'):
                    bg_feats_mixture = torch.from_numpy(np.load(self.hparams.REGISTRANT.mixture_path)).float()
                else:
                    bg_feats_mixture = clutter_bank.detach().cpu()
                # bg_feats_mixture = torch.from_numpy(np.load(osp.join(self.hparams.DEBUG.save_dir_vis, 'bg_feats_mixture_bank_means.npy'))).float()
                get_bg_activations_mixture(torch.from_numpy(debug_info['coke_features']), 
                    bg_feats_mixture, 
                    img_nps,
                    osp.join(self.hparams.DEBUG.save_dir_vis, id_str+'_bg_mixture_acts.png')
                    )
                import ipdb; ipdb.set_trace()

            predictions = dict(pred_pose=pytorch3d.transforms.axis_angle_to_matrix(
                ret['opt_pose'].view(-1, 24, 3)),
                pred_shape=ret['opt_betas'],
                pred_cam=convert_perspective_to_weak_perspective(ret['opt_cam_t'],
                            focal_length=self.hparams.DATASET.FOCAL_LENGTH,
                            img_res=self.hparams.DATASET.IMG_RES)
            )

        pred_pose = predictions['pred_pose']
        pred_betas = predictions['pred_shape']
        pred_cam = predictions['pred_cam']
        
        pred_output = self.body_model_test(
            betas=pred_betas,
            body_pose=pred_pose[:, 1:],
            global_orient=pred_pose[:, 0].unsqueeze(1),
            pose2rot=False)
        pred_vertices = pred_output['vertices']
        pred_keypoints_3d = pred_output['joints']
        all_preds = {}
        all_preds['keypoints_3d'] = pred_keypoints_3d.detach().cpu().numpy()
        all_preds['smpl_pose'] = pred_pose.detach().cpu().numpy()
        all_preds['smpl_beta'] = pred_betas.detach().cpu().numpy()
        all_preds['camera'] = pred_cam.detach().cpu().numpy()
        all_preds['vertices'] = pred_vertices.detach().cpu().numpy()
        image_path = []
        for img_meta in img_metas:
            image_path.append(img_meta['image_path'])
        all_preds['image_path'] = image_path
        all_preds['image_idx'] = kwargs['sample_idx']

        meta_info = {}
        meta_info['keypoints_2d'] = kwargs['keypoints2d'].cpu().numpy()
        for img_meta in img_metas:
            for occ_info in ['occ_size', 'occ_stride', 'occ_idx', 'texture_file', 'texture_crop_tl']:
                if occ_info in img_meta:
                    meta_info.setdefault(occ_info, []).append(img_meta[occ_info])
        all_preds['meta_info'] = meta_info
        return all_preds

    def prepare_targets(self, data_batch: dict):
        # Image Mesh Estimator does not need extra process for ground truth
        return data_batch

    def run_openpose_estimator_old(self, imgs, img_metas, **kwargs):
        openpose_dets = []
        gt_keypoints_batch, _ = convert_kps(kwargs['keypoints2d'].cpu().numpy(),
                                    src='smpl_49',
                                    dst='h36m',)
        
        for img, img_meta, gt_keypoints in zip(imgs.cpu().numpy(), img_metas, gt_keypoints_batch):
            img_norm_cfg = img_meta['img_norm_cfg']
            img = img.transpose((1, 2, 0))
            img_np = imdenormalize(img, mean=img_norm_cfg['mean'], 
                                    std=img_norm_cfg['std'], 
                                    to_bgr=img_norm_cfg['to_rgb'])
            # resize to higher resolution
            # resize_shape = int(scale[0])
            # img_resize = cv2.resize(img_np, (resize_shape, resize_shape))
            openpose_keypoints = np.array(self.openpose_body_estimator(img_np)) # N, 25, 3
            
            # match with groundtruth
            # use mean keypoint distance to decide center human
            if openpose_keypoints.any():
                # openpose_keypoints[:, :, :2] = openpose_keypoints[:, :, :2] * 224 / resize_shape
                gt_keypoints = gt_keypoints[list(constants.LSP_TO_OPENPOSE.keys())]
                joint_dists = np.nanmean(np.linalg.norm(openpose_keypoints[:, list(constants.LSP_TO_OPENPOSE.values()), :2] - gt_keypoints[None,:,:2], axis=2), axis=1)
                joint_dists[np.isnan(joint_dists)] = float('inf')
                eval_person_id = np.argmin(joint_dists)
                openpose_kp = openpose_keypoints[eval_person_id]
                
            else:
                openpose_kp = np.zeros((25, 3))
            openpose_dets.append(openpose_kp)
        openpose_dets = torch.from_numpy(np.array(openpose_dets)).type_as(imgs)
        return openpose_dets
    
    def run_openpose_estimator(self, imgs, img_metas, **kwargs):
        openpose_dets = []
        # gt_keypoints_batch, _ = convert_kps(kwargs['keypoints2d'].cpu().numpy(),
        #                             src='smpl_49',
        #                             dst='h36m',)
        # gt_keypoints_batch = kwargs['keypoints2d'].cpu().numpy()
        gt_keypoints_batch = kwargs['kp2ds_gt'].cpu().numpy()
        for img, img_meta, gt_keypoints in zip(imgs.cpu().numpy(), img_metas, gt_keypoints_batch):
            img_norm_cfg = img_meta['img_norm_cfg']
            img = img.transpose((1, 2, 0))
            img_np = imdenormalize(img, mean=img_norm_cfg['mean'], 
                                    std=img_norm_cfg['std'], 
                                    to_bgr=img_norm_cfg['to_rgb'])
            # resize to higher resolution
            # resize_shape = int(scale[0])
            # img_resize = cv2.resize(img_np, (resize_shape, resize_shape))
            openpose_keypoints = np.array(self.openpose_body_estimator(img_np)) # N, 25, 3
            
            # match with groundtruth
            # use mean keypoint distance to decide center human
            if openpose_keypoints.any():
                # openpose_keypoints[:, :, :2] = openpose_keypoints[:, :, :2] * 224 / resize_shape
                # gt_keypoints = gt_keypoints[list(constants.LSP_TO_OPENPOSE.keys())]
                gt_keypoints = gt_keypoints[:25]
                openpose_keypoints[openpose_keypoints[:, :, 2] == 0] = np.nan
                joint_dists = np.nanmean(np.linalg.norm(openpose_keypoints[:, :, :2] - gt_keypoints[None, :, :2], axis=2), axis=1)
                joint_dists[np.isnan(joint_dists)] = float('inf')
                eval_person_id = np.argmin(joint_dists)
                openpose_kp = openpose_keypoints[eval_person_id]
                openpose_kp[np.isnan(openpose_kp)] = 0.0
            else:
                openpose_kp = np.zeros((25, 3))
            openpose_dets.append(openpose_kp)
        openpose_dets = torch.from_numpy(np.array(openpose_dets)).type_as(imgs)
        return openpose_dets