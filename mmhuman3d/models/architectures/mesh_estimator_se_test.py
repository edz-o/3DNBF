from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple, Union
from yacs.config import CfgNode as CN
import torch
import torch.nn.functional as F
import pytorch3d
import numpy as np
from pytorch3d.transforms.rotation_conversions import matrix_to_axis_angle

from mmhuman3d.core.conventions.keypoints_mapping import get_keypoint_idx
from mmhuman3d.models.utils import FitsDict
from mmhuman3d.utils.geometry import (
    batch_rodrigues,
    estimate_translation,
    project_points,
    rotation_matrix_to_angle_axis,
    convert_weak_perspective_to_perspective,
    convert_perspective_to_weak_perspective
)
from mmhuman3d.utils.image_utils import (
    generate_part_labels, get_vert_to_part,
    get_mask_and_visibility,
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
    build_feature_bank
)
from .base_architecture import BaseArchitecture
from ..registrants import NeuralSMPLFitting

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


class BodyModelEstimatorSE(BaseArchitecture, metaclass=ABCMeta):
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
                 loss_keypoints2d: Optional[Union[dict, None]] = None,
                 loss_keypoints3d: Optional[Union[dict, None]] = None,
                 loss_vertex: Optional[Union[dict, None]] = None,
                 loss_smpl_pose: Optional[Union[dict, None]] = None,
                 loss_smpl_betas: Optional[Union[dict, None]] = None,
                 loss_camera: Optional[Union[dict, None]] = None,
                 loss_segm_mask: Optional[Union[dict, None]] = None,
                 loss_part_segm: Optional[Union[dict, None]] = None,
                 loss_contrastive: Optional[Union[dict, None]] = None,
                 #loss_noise_reg: Optional[Union[dict, None]] = None,
                 loss_adv: Optional[Union[dict, None]] = None,
                 init_cfg: Optional[Union[list, dict, None]] = None,
                 hparams: Optional[Union[dict, None]] = None):
        super(BodyModelEstimatorSE, self).__init__(init_cfg)
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
        # if self.loss_segm_mask is not None:
        self.neural_renderer_gt = build_neural_renderer(hparams.MODEL.RENDERER_GT)
        self.neural_renderer_pred = build_neural_renderer(hparams.MODEL.RENDERER_PRED)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_mesh_info()

        self.registrant = build_registrant(registrant)
        if registrant is not None:
            self.fits = 'registration'
            self.fits_dict = FitsDict(fits='static')
            if isinstance(self.registrant, NeuralSMPLFitting):
                self.registrant.set_mesh_info(ds_indices=self.ds_indices, 
                                                faces_down=self.faces_down)

    def load_mesh_info(self):
        mesh_sample_data = np.load(self.hparams.mesh_sample_param_path)
        mesh_sample_data = dict(mesh_sample_data)
        self.ds_indices = mesh_sample_data['indices']
        n = len(self.ds_indices)
        
        # self.ds_fun = lambda x: x[ds_indices]
        self.faces_down = torch.tensor(mesh_sample_data['faces_downsampled'], device=self.device)
        adj_mat = torch.zeros((n, n)).long()
        for face in self.faces_down:
            x1, x2, x3 = sorted(face)
            adj_mat[x1, x2] = 1
            adj_mat[x1, x3] = 1
            adj_mat[x2, x3] = 1
        self.adj_mat = (adj_mat + adj_mat.transpose(1,0)).unsqueeze(0)

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
            features = self.backbone(img)
        else:
            features = data_batch['features']

        if self.neck is not None:
            features = self.neck(features)

        predictions = self.head(features)
        targets = self.prepare_targets(data_batch)

        # optimize discriminator (if have)
        if self.disc is not None:
            self.optimize_discrinimator(predictions, data_batch, optimizer)

        if self.registrant is not None:
            if isinstance(self.registrant, NeuralSMPLFitting):
                neural_features, clutter_bank = self.feature_bank.get_feature_banks()
                self.registrant.neural_mesh_model.set_features(neural_features)
                self.registrant.set_clutter(clutter_bank)

            targets = self.run_registration(predictions, targets,
                                focal_length=self.hparams.DATASET.FOCAL_LENGTH, 
                                img_res=self.hparams.DATASET.IMG_RES)

        losses = self.compute_losses(predictions, targets)
        # optimizer generator part
        if self.disc is not None:
            adv_loss = self.optimize_generator(predictions)
            losses.update(adv_loss)

        loss, log_vars = self._parse_losses(losses)
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

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))
        return outputs

    def run_registration(
            self,
            predictions: dict,
            targets: dict,
            threshold: Optional[float] = 10.0,
            focal_length: Optional[float] = 5000.0,
            img_res: Optional[Union[Tuple[int], int]] = 224) -> dict:
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
        gt_betas = targets['smpl_betas'].float()
        gt_global_orient = targets['smpl_global_orient'].float()
        gt_pose = targets['smpl_body_pose'].float().view(-1, 69)

        pred_rotmat = predictions['pred_pose'].detach().clone()
        pred_betas = predictions['pred_shape'].detach().clone()
        pred_cam = predictions['pred_cam'].detach().clone()
        pred_cam_t = convert_weak_perspective_to_perspective(
                    pred_cam,
                    focal_length=focal_length,
                    img_res=img_res,
                )
        if 'pred_segm_mask' in predictions:
            pred_segm_mask = predictions['pred_segm_mask'].detach().clone()
        else:
            pred_segm_mask = None

        if 'coke_features' in predictions:
            coke_features = predictions['coke_features'].detach().clone()
        else:
            coke_features = None

        gt_keypoints_2d = targets['keypoints2d'].float()

        # try:
        #     gt_keypoints_2d = torch.cat(
        #         [keypoints2d, keypoints2d_mask.reshape(-1, 49, 1)], dim=-1)
        # except Exception:
        #     gt_keypoints_2d = torch.cat(
        #         [keypoints2d, keypoints2d_mask.reshape(-1, 24, 1)], dim=-1)
        num_keypoints = gt_keypoints_2d.shape[1]

        has_smpl = targets['has_smpl'].view(
            -1).bool()  # flag that indicates whether SMPL parameters are valid
        batch_size = has_smpl.shape[0]
        device = has_smpl.device

        # Get GT vertices and model joints
        # Note that gt_model_joints is different from gt_joints as
        # it comes from SMPL
        gt_out = self.body_model_train(
            betas=gt_betas, body_pose=gt_pose, global_orient=gt_global_orient)
        if num_keypoints == 49:
            gt_model_joints = gt_out['joints']
            gt_vertices = gt_out['vertices']
        else:
            gt_model_joints = gt_out['joints'][:, 25:, :]
            gt_vertices = gt_out['vertices']
        # TODO: add joint mask

        # Get current best fits from the dictionary
        try:
            opt_pose, opt_betas = self.fits_dict[(dataset_name, indices.cpu(),
                                                rot_angle.cpu(),
                                                is_flipped.cpu())]
        except:
            opt_pose = torch.cat((gt_global_orient, gt_pose), dim=1)
            opt_betas = gt_betas

        opt_pose = opt_pose.to(device)
        opt_betas = opt_betas.to(device)
        opt_output = self.body_model_train(
            betas=opt_betas,
            body_pose=opt_pose[:, 3:],
            global_orient=opt_pose[:, :3])
        if num_keypoints == 49:
            opt_joints = opt_output['joints']
            opt_vertices = opt_output['vertices']
        else:
            opt_joints = opt_output['joints'][:, 25:, :]
            opt_vertices = opt_output['vertices']

        # TODO: current pipeline, the keypoints are already in the pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()

        # Estimate camera translation given the model joints and 2D keypoints
        # by minimizing a weighted least squares loss
        gt_cam_t = estimate_translation(
            gt_model_joints,
            gt_keypoints_2d_orig,
            focal_length=focal_length,
            img_size=img_res)

        opt_cam_t = estimate_translation(
            opt_joints,
            gt_keypoints_2d_orig,
            focal_length=focal_length,
            img_size=img_res)

        with torch.no_grad():
            loss_dict = self.registrant.evaluate(
                global_orient=opt_pose[:, :3],
                body_pose=opt_pose[:, 3:],
                betas=opt_betas,
                transl=opt_cam_t,
                keypoints2d=gt_keypoints_2d_orig[:, :, :2],
                keypoints2d_conf=gt_keypoints_2d_orig[:, :, 2],
                reduction_override='none',
                pred_segm_mask=pred_segm_mask,
                predicted_map=coke_features)
        opt_joint_loss = loss_dict['keypoint2d_loss'].sum(dim=-1).sum(dim=-1)

        # Convert predicted rotation matrices to axis-angle
        # # Deprecated hack
        # pred_rotmat_hom = torch.cat([
        #     pred_rotmat.detach().view(-1, 3, 3).detach(),
        #     torch.tensor([0, 0, 1], dtype=torch.float32, device=device).view(
        #         1, 3, 1).expand(batch_size * 24, -1, -1)
        # ],
        #                             dim=-1)
        # pred_pose = rotation_matrix_to_angle_axis(
        #     pred_rotmat_hom).contiguous().view(batch_size, -1)
        # # tgm.rotation_matrix_to_angle_axis returns NaN for 0 rotation,
        # # so manually hack it
        # pred_pose[torch.isnan(pred_pose)] = 0.0
        pred_pose = matrix_to_axis_angle(pred_rotmat.detach()).flatten(1)

        # TODO: support HMR+
        registrant_output = self.registrant(
            keypoints2d=gt_keypoints_2d_orig[:, :, :2],
            keypoints2d_conf=gt_keypoints_2d_orig[:, :, 2],
            pred_segm_mask=pred_segm_mask,
            predicted_map=coke_features,
            init_global_orient=pred_pose[:, :3],
            init_transl=pred_cam_t, # only correct when Rotation is None
            init_body_pose=pred_pose[:, 3:],
            init_betas=pred_betas,
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
        new_opt_joint_loss = registrant_output['keypoint2d_loss'].sum(
            dim=-1).sum(dim=-1)

        # new_opt_joint_loss = new_opt_joint_loss.mean(dim=-1)

        # Will update the dictionary for the examples where the new loss
        # is less than the current one
        update = (new_opt_joint_loss < opt_joint_loss)

        opt_joint_loss[update] = new_opt_joint_loss[update]
        opt_vertices[update, :] = new_opt_vertices[update, :]
        opt_joints[update, :] = new_opt_joints[update, :]
        opt_pose[update, :] = new_opt_pose[update, :]
        opt_betas[update, :] = new_opt_betas[update, :]
        opt_cam_t[update, :] = new_opt_cam_t[update, :]

        self.fits_dict[(dataset_name, indices.cpu(), rot_angle.cpu(),
                        is_flipped.cpu(), update.cpu())] = (opt_pose.cpu(),
                                                            opt_betas.cpu())

        # Replace extreme betas with zero betas
        opt_betas[(opt_betas.abs() > 3).any(dim=-1)] = 0.

        # Replace the optimized parameters with the ground truth parameters,
        # if available
        opt_vertices[has_smpl, :, :] = gt_vertices[has_smpl, :, :]
        opt_cam_t[has_smpl, :] = gt_cam_t[has_smpl, :]
        opt_joints[has_smpl, :, :] = gt_model_joints[has_smpl, :, :]
        opt_pose[has_smpl, 3:] = gt_pose[has_smpl, :]
        opt_pose[has_smpl, :3] = gt_global_orient[has_smpl, :]
        opt_betas[has_smpl, :] = gt_betas[has_smpl, :]

        # Assert whether a fit is valid by comparing the joint loss with
        # the threshold
        valid_fit = (opt_joint_loss < threshold).to(device)
        valid_fit = valid_fit | has_smpl
        targets['valid_fit'] = valid_fit

        targets['opt_vertices'] = opt_vertices
        targets['opt_cam_t'] = opt_cam_t
        targets['opt_joints'] = opt_joints
        targets['opt_pose'] = opt_pose
        targets['opt_betas'] = opt_betas

        return targets

    def run_registration_test(
            self,
            predictions: dict,
            targets: dict,
            threshold: Optional[float] = 10.0,
            focal_length: Optional[float] = 5000.0,
            img_res: Optional[Union[Tuple[int], int]] = 224) -> dict:
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
        else:
            pred_rotmat = predictions['pred_pose'].detach().clone()
            pred_betas = predictions['pred_shape'].detach().clone()
            pred_cam = predictions['pred_cam'].detach().clone()

        pred_cam_t = convert_weak_perspective_to_perspective(
                    pred_cam,
                    focal_length=focal_length,
                    img_res=img_res,
                )
        if 'pred_segm_mask' in predictions:
            pred_segm_mask = predictions['pred_segm_mask'].detach().clone()
            if pred_segm_mask.shape[-1] != img_res//4:
                pred_segm_mask = F.interpolate(pred_segm_mask, (img_res//4, img_res//4), mode='bilinear', align_corners=True)
        else:
            pred_segm_mask = None

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
                                                    self.registrant.neural_mesh_model.features, 
                                                    self.hparams.MODEL.downsample_rate, 
                                                    n_orient=self.registrant.hparams.n_orient)
        vertices2d_det = torch.cat((vertices2d_det, vertices2d_det_conf.unsqueeze(2)), dim=2)

        gt_keypoints_2d = targets['keypoints2d'].float()

        # try:
        #     gt_keypoints_2d = torch.cat(
        #         [keypoints2d, keypoints2d_mask.reshape(-1, 49, 1)], dim=-1)
        # except Exception:
        #     gt_keypoints_2d = torch.cat(
        #         [keypoints2d, keypoints2d_mask.reshape(-1, 24, 1)], dim=-1)
        num_keypoints = gt_keypoints_2d.shape[1]

        has_smpl = targets['has_smpl'].view(
            -1).bool()  # flag that indicates whether SMPL parameters are valid
        batch_size = has_smpl.shape[0]
        device = has_smpl.device

        # Get inital fits from the prediction
    
        opt_pose = matrix_to_axis_angle(pred_rotmat).flatten(1)
        opt_betas = pred_betas
        # import pdb; pdb.set_trace()
        opt_pose = opt_pose.to(device)
        opt_betas = opt_betas.to(device)
        opt_output = self.body_model_train(
            betas=opt_betas,
            body_pose=opt_pose[:, 3:],
            global_orient=opt_pose[:, :3])
        if num_keypoints == 49:
            opt_joints = opt_output['joints']
            opt_vertices = opt_output['vertices']
        else:
            opt_joints = opt_output['joints'][:, 25:, :]
            opt_vertices = opt_output['vertices']

        # TODO: current pipeline, the keypoints are already in the pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        gt_keypoints_2d_orig[:, 25:, 2] = 0

        opt_cam_t = pred_cam_t.clone()

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
                predicted_map=coke_features)
        opt_loss = loss_dict['total_loss']

        # # Deprecated hack
        # # Convert predicted rotation matrices to axis-angle
        # pred_rotmat_hom = torch.cat([
        #     pred_rotmat.detach().view(-1, 3, 3).detach(),
        #     torch.tensor([0, 0, 1], dtype=torch.float32, device=device).view(
        #         1, 3, 1).expand(batch_size * 24, -1, -1)
        # ],
        #                             dim=-1)
        # pred_pose = rotation_matrix_to_angle_axis(
        #     pred_rotmat_hom).contiguous().view(batch_size, -1)
        # # tgm.rotation_matrix_to_angle_axis returns NaN for 0 rotation,
        # # so manually hack it
        # pred_pose[torch.isnan(pred_pose)] = 0.0

        pred_pose = matrix_to_axis_angle(pred_rotmat.detach()).flatten(1)
        
        # TODO: support HMR+
        registrant_output = self.registrant(
            keypoints2d=gt_keypoints_2d_orig[:, :, :2],
            keypoints2d_conf=gt_keypoints_2d_orig[:, :, 2],
            vertices2d=vertices2d_det[:, :, :2],
            vertices2d_conf=vertices2d_det[:, :, 2],
            pred_segm_mask=pred_segm_mask,
            predicted_map=coke_features,
            init_global_orient=pred_pose[:, :3],
            init_transl=pred_cam_t, # only correct when Rotation is None
            init_body_pose=pred_pose[:, 3:],
            init_betas=pred_betas,
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

        # new_opt_joint_loss = new_opt_joint_loss.mean(dim=-1)

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

        return targets

    def optimize_discrinimator(self, predictions: dict, data_batch: dict,
                               optimizer: dict):
        """Optimize discrinimator during adversarial training."""
        set_requires_grad(self.disc, True)
        fake_data = self.make_fake_data(predictions, requires_grad=False)
        real_data = self.make_real_data(data_batch)
        fake_score = self.disc(fake_data)
        real_score = self.disc(real_data)

        disc_losses = {}
        disc_losses['real_loss'] = self.loss_adv(
            real_score, target_is_real=True, is_disc=True)
        disc_losses['fake_loss'] = self.loss_adv(
            fake_score, target_is_real=False, is_disc=True)
        loss_disc, log_vars_d = self._parse_losses(disc_losses)

        optimizer['disc'].zero_grad()
        loss_disc.backward()
        optimizer['disc'].step()

    def optimize_generator(self, predictions: dict):
        """Optimize generator during adversarial training."""
        set_requires_grad(self.disc, False)
        fake_data = self.make_fake_data(predictions, requires_grad=True)
        pred_score = self.disc(fake_data)
        loss_adv = self.loss_adv(
            pred_score, target_is_real=True, is_disc=False)
        loss = dict(adv_loss=loss_adv)
        return loss

    def compute_keypoints3d_loss(self, pred_keypoints3d: torch.Tensor,
                                 gt_keypoints3d: torch.Tensor):
        """Compute loss for 3d keypoints."""
        keypoints3d_conf = gt_keypoints3d[:, :, 3].float().unsqueeze(-1)
        keypoints3d_conf = keypoints3d_conf.repeat(1, 1, 3)
        pred_keypoints3d = pred_keypoints3d.float()
        gt_keypoints3d = gt_keypoints3d[:, :, :3].float()

        # currently, only mpi_inf_3dhp and h36m have 3d keypoints
        # both datasets have right_hip_extra and left_hip_extra
        right_hip_idx = get_keypoint_idx('right_hip_extra', self.convention)
        left_hip_idx = get_keypoint_idx('left_hip_extra', self.convention)
        gt_pelvis = (gt_keypoints3d[:, right_hip_idx, :] +
                     gt_keypoints3d[:, left_hip_idx, :]) / 2
        pred_pelvis = (pred_keypoints3d[:, right_hip_idx, :] +
                       pred_keypoints3d[:, left_hip_idx, :]) / 2

        gt_keypoints3d = gt_keypoints3d - gt_pelvis[:, None, :]
        pred_keypoints3d = pred_keypoints3d - pred_pelvis[:, None, :]
        loss = self.loss_keypoints3d(
            pred_keypoints3d, gt_keypoints3d, reduction_override='none')

        # import numpy as np; np.savez('debug_normal_h36m.npz', pred=pred_keypoints3d.detach().cpu().numpy(), gt=gt_keypoints3d.detach().cpu().numpy(), conf=keypoints3d_conf.detach().cpu().numpy())
        valid_pos = keypoints3d_conf > 0
        if keypoints3d_conf[valid_pos].numel() == 0:
            return torch.Tensor([0]).type_as(gt_keypoints3d)
        loss = torch.sum(loss * keypoints3d_conf)
        loss /= keypoints3d_conf[valid_pos].numel()
        return loss

    def compute_keypoints2d_loss(self,
                                 pred_keypoints3d: torch.Tensor,
                                 pred_cam: torch.Tensor,
                                 gt_keypoints2d: torch.Tensor,
                                 img_res: Optional[int] = 224,
                                 focal_length: Optional[int] = 5000):
        """Compute loss for 2d keypoints."""
        keypoints2d_conf = gt_keypoints2d[:, :, 2].float().unsqueeze(-1)
        keypoints2d_conf = keypoints2d_conf.repeat(1, 1, 2)
        gt_keypoints2d = gt_keypoints2d[:, :, :2].float()
        pred_keypoints2d = project_points(
            pred_keypoints3d,
            pred_cam,
            focal_length=focal_length,
            img_res=img_res)
        # Normalize keypoints to [-1,1]
        # The coordinate origin of pred_keypoints_2d is
        # the center of the input image.
        pred_keypoints2d = 2 * pred_keypoints2d / (img_res - 1)
        # The coordinate origin of gt_keypoints_2d is
        # the top left corner of the input image.
        gt_keypoints2d = 2 * gt_keypoints2d / (img_res - 1) - 1
        loss = self.loss_keypoints2d(
            pred_keypoints2d, gt_keypoints2d, reduction_override='none')
        valid_pos = keypoints2d_conf > 0
        if keypoints2d_conf[valid_pos].numel() == 0:
            return torch.Tensor([0]).type_as(gt_keypoints2d)
        loss = torch.sum(loss * keypoints2d_conf)
        loss /= keypoints2d_conf[valid_pos].numel()
        return loss

    def compute_vertex_loss(self, pred_vertices: torch.Tensor,
                            gt_vertices: torch.Tensor, has_smpl: torch.Tensor):
        """Compute loss for vertices."""
        gt_vertices = gt_vertices.float()
        conf = has_smpl.float().view(-1, 1, 1)
        conf = conf.repeat(1, gt_vertices.shape[1], gt_vertices.shape[2])
        loss = self.loss_vertex(
            pred_vertices, gt_vertices, reduction_override='none')
        valid_pos = conf > 0
        if conf[valid_pos].numel() == 0:
            return torch.Tensor([0]).type_as(gt_vertices)
        loss = torch.sum(loss * conf) / conf[valid_pos].numel()
        return loss

    def compute_smpl_pose_loss(self, pred_rotmat: torch.Tensor,
                               gt_pose: torch.Tensor, has_smpl: torch.Tensor):
        """Compute loss for smpl pose."""
        conf = has_smpl.float().view(-1, 1, 1, 1).repeat(1, 24, 3, 3)
        gt_rotmat = batch_rodrigues(gt_pose.view(-1, 3)).view(-1, 24, 3, 3)
        loss = self.loss_smpl_pose(
            pred_rotmat, gt_rotmat, reduction_override='none')
        valid_pos = conf > 0
        if conf[valid_pos].numel() == 0:
            return torch.Tensor([0]).type_as(gt_pose)
        loss = torch.sum(loss * conf) / conf[valid_pos].numel()
        return loss

    def compute_smpl_betas_loss(self, pred_betas: torch.Tensor,
                                gt_betas: torch.Tensor,
                                has_smpl: torch.Tensor):
        """Compute loss for smpl betas."""
        conf = has_smpl.float().view(-1, 1).repeat(1, 10)
        loss = self.loss_smpl_betas(
            pred_betas, gt_betas, reduction_override='none')
        valid_pos = conf > 0
        if conf[valid_pos].numel() == 0:
            return torch.Tensor([0]).type_as(gt_betas)
        loss = torch.sum(loss * conf) / conf[valid_pos].numel()
        return loss

    def compute_camera_loss(self, cameras: torch.Tensor):
        """Compute loss for predicted camera parameters."""
        loss = self.loss_camera(cameras)
        return loss

    def compute_segm_mask_loss(self, pred_segm_mask: torch.Tensor, 
                                gt_segm_mask: torch.Tensor):
        loss = self.loss_segm_mask(pred_segm_mask, gt_segm_mask)
        return loss
    
    def compute_part_segm_loss(self, pred_segm_rgb, gt_segm_rgb):
        loss = self.loss_part_segm(pred_segm_rgb, gt_segm_rgb)
        return loss

    def compute_coke_loss(self, coke_features,
                keypoint_positions,
                has_smpl,
                iskpvisible,
                feature_bank,
                adj_mat,
                vert_orients=None,
                bg_mask=None):
        """Verbose"""

        loss_contrastive, loss_noise_reg = self.loss_contrastive(coke_features,
                keypoint_positions,
                has_smpl,
                iskpvisible,
                feature_bank,
                adj_mat,
                vert_orients=vert_orients,
                bg_mask=bg_mask)
        return loss_contrastive, loss_noise_reg

    def compute_losses(self, predictions: dict, targets: dict):
        """Compute losses."""
        bs = targets['keypoints3d'].shape[0]

        pred_betas = predictions['pred_shape'].view(-1, 10)
        pred_pose = predictions['pred_pose'].view(-1, 24, 3, 3)
        pred_cam = predictions['pred_cam'].view(-1, 3)

        gt_keypoints3d = targets['keypoints3d']
        gt_keypoints2d = targets['keypoints2d']

        coke_features = predictions['coke_features']

        
        # pred_pose N, 24, 3, 3
        if self.body_model_train is not None:
            pred_output = self.body_model_train(
                betas=pred_betas,
                body_pose=pred_pose[:, 1:],
                global_orient=pred_pose[:, 0].unsqueeze(1),
                pose2rot=False,
                num_joints=gt_keypoints2d.shape[1])
            pred_keypoints3d = pred_output['joints']
            pred_vertices = pred_output['vertices']

        # # TODO: temp. Should we multiply confs here?
        # pred_keypoints3d_mask = pred_output['joint_mask']
        # keypoints3d_mask = keypoints3d_mask * pred_keypoints3d_mask

        # TODO: temp solution
        if 'valid_fit' in targets:
            has_smpl = targets['valid_fit'].view(-1)
            # global_orient = targets['opt_pose'][:, :3].view(-1, 1, 3)
            gt_pose = targets['opt_pose']
            gt_betas = targets['opt_betas']
            gt_vertices = targets['opt_vertices']
            if self.body_model_train is not None:
                gt_output = self.body_model_train(
                    betas=gt_betas,
                    body_pose=gt_pose[:, 3:],
                    global_orient=gt_pose[:, :3],
                    num_joints=gt_keypoints2d.shape[1])
        else:
            has_smpl = targets['has_smpl'].view(-1)
            gt_pose = targets['smpl_body_pose']
            global_orient = targets['smpl_global_orient'].view(-1, 1, 3)
            gt_pose = torch.cat((global_orient, gt_pose), dim=1).float()
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
        if self.loss_keypoints3d is not None:
            losses['keypoints3d_loss'] = self.compute_keypoints3d_loss(
                pred_keypoints3d, gt_keypoints3d)
        if self.loss_keypoints2d is not None:
            losses['keypoints2d_loss'] = self.compute_keypoints2d_loss(
                pred_keypoints3d, pred_cam, gt_keypoints2d)
        if self.loss_vertex is not None:
            losses['vertex_loss'] = self.compute_vertex_loss(
                pred_vertices, gt_vertices, has_smpl)
        if self.loss_smpl_pose is not None:
            losses['smpl_pose_loss'] = self.compute_smpl_pose_loss(
                pred_pose, gt_pose, has_smpl)
        if self.loss_smpl_betas is not None:
            losses['smpl_betas_loss'] = self.compute_smpl_betas_loss(
                pred_betas, gt_betas, has_smpl)
        if self.loss_camera is not None:
            losses['camera_loss'] = self.compute_camera_loss(pred_cam)
        if self.loss_segm_mask is not None:
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
                self.hparams.DATASET.FOCAL_LENGTH, 
                self.hparams.DATASET.IMG_RES, 
                gt_cam_t)
            self.neural_renderer_gt.rasterizer.cameras = new_cam
            gt_segm_mask, gt_segm_rgb = generate_part_labels(
                vertices=gt_vertices,
                faces=self.body_model_train.faces_tensor[None].expand(bs, -1, -1),
                vert_to_part=self.vert_to_part.to(gt_vertices.device),
                neural_renderer=self.neural_renderer_gt.to(gt_vertices.device)
            )

            pred_segm_mask = predictions['pred_segm_mask'][has_smpl == 1]
            gt_segm_mask = gt_segm_mask[has_smpl == 1]
            losses['loss_segm_mask'] = self.compute_segm_mask_loss(pred_segm_mask, gt_segm_mask)
            
            with torch.no_grad():
                vertices2d_proj = new_cam.transform_points_screen(gt_verts_down, 
                                        image_size=(coke_features.shape[3:1:-1], )
                                        )[:, :, :2] # image_size is (W, H)
                
                mask, iskpvisible = get_mask_and_visibility(vertices=gt_verts_down,
                                        faces=self.faces_down, 
                                        rasterizer=self.neural_renderer_gt.rasterizer, )

                vert_orients = get_vert_orients(gt_keypoints3d_smpl[:, :25], self.hparams.MODEL.N_ORIENT)

                bg_mask = 1. - mask

            # if (bg_mask.view(mask.shape[0], -1).sum(dim=1) < 1e-6).any():
            #     isInvalid = (bg_mask.view(mask.shape[0], -1).sum(dim=1) < 1e-6)
            #     for i in range(len(isInvalid)):
            #         if isInvalid[i]:
            #             logger.debug(targets['img_metas'][i]['image_path'])
                 
            if self.loss_contrastive is not None:
                losses['loss_contrastive'], losses['loss_noise_reg'] = self.compute_coke_loss(coke_features, 
                                                        torch.flip(vertices2d_proj, [2]),
                                                        targets['has_smpl'].view(-1).bool(),
                                                        iskpvisible,
                                                        self.feature_bank,
                                                        self.adj_mat.to(coke_features.device),
                                                        vert_orients=vert_orients,
                                                        bg_mask=bg_mask)

            if self.loss_part_segm is not None:
                pred_cam_t = convert_weak_perspective_to_perspective(
                    pred_cam,
                    focal_length=self.hparams.DATASET.FOCAL_LENGTH,
                    img_res=self.hparams.DATASET.IMG_RES,
                )
                new_cam = get_cameras(
                    self.hparams.DATASET.FOCAL_LENGTH, 
                    self.hparams.DATASET.IMG_RES, 
                    pred_cam_t)
                self.neural_renderer_pred.rasterizer.cameras = new_cam
                _, pred_segm_rgb = generate_part_labels(
                    vertices=pred_vertices,
                    faces=self.body_model_train.faces_tensor[None].expand(bs, -1, -1),
                    vert_to_part=self.vert_to_part.to(gt_vertices.device),
                    neural_renderer=self.neural_renderer_pred.to(gt_vertices.device)
                )
                losses['loss_part_segm'] = self.compute_part_segm_loss(pred_segm_rgb[has_smpl == 1],
                                                                    gt_segm_rgb[has_smpl == 1])

        return losses

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
        raise NotImplementedError('This interface should not be used in '
                                  'current training schedule. Please use '
                                  '`train_step` for training.')

    @abstractmethod
    def forward_test(self, img, img_metas, **kwargs):
        """Defines the computation performed at every call when testing."""
        pass


@ARCHITECTURES.register_module()
class ImageBodyModelEstimatorSE(BodyModelEstimatorSE):

    def make_fake_data(self, predictions: dict, requires_grad: bool):
        pred_cam = predictions['pred_cam']
        pred_pose = predictions['pred_pose']
        pred_betas = predictions['pred_shape']
        if requires_grad:
            fake_data = (pred_cam, pred_pose, pred_betas)
        else:
            fake_data = (pred_cam.detach(), pred_pose.detach(),
                         pred_betas.detach())
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

        predictions = self.head(features)

        if self.registrant is not None:
            # targets = self.prepare_targets_test(img_metas=img_metas, **kwargs)
            kwargs['img_metas'] = img_metas
            if isinstance(self.registrant, NeuralSMPLFitting):
                neural_features, clutter_bank = self.feature_bank.get_feature_banks()
                self.registrant.neural_mesh_model.set_features(neural_features)
                self.registrant.set_clutter(clutter_bank)

            ret = self.run_registration_test(predictions, kwargs,
                                focal_length=self.hparams.DATASET.FOCAL_LENGTH, 
                                img_res=self.hparams.DATASET.IMG_RES)
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
        # pred_output = self.body_model_test(betas=pred_betas, body_pose=pred_pose[:, 1:], global_orient=pred_pose[:, 0].unsqueeze(1), pose2rot=False)
        
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
        return all_preds


@ARCHITECTURES.register_module()
class VideoBodyModelEstimatorSE(BodyModelEstimatorSE):

    def make_fake_data(self, predictions: dict, requires_grad: bool):
        B, T = predictions['pred_cam'].shape[:2]
        pred_cam_vec = predictions['pred_cam']
        pred_betas_vec = predictions['pred_shape']
        pred_pose = predictions['pred_pose']
        pred_pose_vec = rotation_matrix_to_angle_axis(pred_pose.view(-1, 3, 3))
        pred_pose_vec = pred_pose_vec.contiguous().view(B, T, -1)
        pred_theta_vec = (pred_cam_vec, pred_pose_vec, pred_betas_vec)
        pred_theta_vec = torch.cat(pred_theta_vec, dim=-1)

        if not requires_grad:
            pred_theta_vec = pred_theta_vec.detach()
        return pred_theta_vec[:, :, 6:75]

    def make_real_data(self, data_batch: dict):
        B, T = data_batch['adv_smpl_transl'].shape[:2]
        transl = data_batch['adv_smpl_transl'].view(B, T, -1)
        global_orient = \
            data_batch['adv_smpl_global_orient'].view(B, T, -1)
        body_pose = data_batch['adv_smpl_body_pose'].view(B, T, -1)
        betas = data_batch['adv_smpl_betas'].view(B, T, -1)
        real_data = (transl, global_orient, body_pose, betas)
        real_data = torch.cat(real_data, dim=-1).float()
        return real_data[:, :, 6:75]

    def prepare_targets(self, data_batch: dict):
        # Video Mesh Estimator needs squeeze first two dimensions
        B, T = data_batch['smpl_body_pose'].shape[:2]

        output = {
            'smpl_body_pose': data_batch['smpl_body_pose'].view(-1, 23, 3),
            'smpl_global_orient': data_batch['smpl_global_orient'].view(-1, 3),
            'smpl_betas': data_batch['smpl_betas'].view(-1, 10),
            'has_smpl': data_batch['has_smpl'].view(-1),
            'keypoints3d': data_batch['keypoints3d'].view(B * T, -1, 4),
            'keypoints2d': data_batch['keypoints2d'].view(B * T, -1, 3)
        }
        return output

    def forward_test(self, img_metas: dict, **kwargs):
        """Defines the computation performed at every call when testing."""
        if self.backbone is not None:
            features = self.backbone(kwargs['img'])
        else:
            features = kwargs['features']

        if self.neck is not None:
            features = self.neck(features)

        B, T = features.shape[:2]
        predictions = self.head(features)
        pred_pose = predictions['pred_pose'].view(-1, 24, 3, 3)
        pred_betas = predictions['pred_shape'].view(-1, 10)
        pred_cam = predictions['pred_cam'].view(-1, 3)

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
        all_preds['image_idx'] = \
            kwargs['sample_idx'].detach().cpu().numpy().reshape((-1))
        return 