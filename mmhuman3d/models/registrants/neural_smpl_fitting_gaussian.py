from typing import List, Tuple, Union
from yacs.config import CfgNode as CN
import numpy as np
import os
import os.path as osp
import torch
import torch.nn.functional as F

from mmcv.runner import build_optimizer
import ipdb
from mmhuman3d.core.cameras import build_cameras
from mmhuman3d.core.conventions import constants
from mmhuman3d.core.conventions.keypoints_mapping import (
    get_keypoint_idx,
    get_keypoint_idxs_by_part,
    convert_kps,
    get_mapping,
)
from mmhuman3d.models.builder import REGISTRANTS, build_body_model, build_loss
from mmhuman3d.utils.neural_renderer import (
    build_neural_mesh_model,
    get_blend_params,
    get_cameras,
)
from mmhuman3d.utils.neural_renderer_voge import build_neural_mesh_model_voge
from mmhuman3d.utils.neuralsmpl_utils import get_vert_orient_weights
from mmhuman3d.utils.neural_renderer_voge import build_neural_renderer_voge
from mmhuman3d.utils.image_utils import get_vert_to_part, sample_feature
from mmhuman3d.utils.vis_utils import visualize_prediction_pth
from mmhuman3d.utils.geometry import convert_perspective_to_weak_perspective
from loguru import logger
import sys


class OptimizableParameters:
    """Collects parameters for optimization."""

    def __init__(self):
        self.opt_params = []

    def set_param(self, fit_param: torch.Tensor, param: torch.Tensor) -> None:
        """Set requires_grad and collect parameters for optimization.

        Args:
            fit_param: whether to optimize this body model parameter
            param: body model parameter

        Returns:
            None
        """
        if fit_param:
            param.requires_grad = True
            self.opt_params.append(param)
        else:
            param.requires_grad = False

    def parameters(self) -> List[torch.Tensor]:
        """Returns parameters. Compatible with mmcv's build_parameters()

        Returns:
            opt_params: a list of body model parameters for optimization
        """
        return self.opt_params


@REGISTRANTS.register_module()
class NeuralSMPLFittingVoGE(object):
    """NeuralSMPL with extended features.

    - video input
    - 3D keypoints
    """

    def __init__(
        self,
        body_model: Union[dict, torch.nn.Module],
        num_epochs: int = 20,
        camera: Union[dict, torch.nn.Module] = None,
        img_res: Union[Tuple[int], int] = 224,
        stages: dict = None,
        optimizer: dict = None,
        likelihood_loss: dict = None,
        segm_mask_loss: dict = None,
        mask_loss: dict = None,
        keypoints2d_loss: dict = None,
        keypoints3d_loss: dict = None,
        vertices2d_loss: dict = None,
        shape_prior_loss: dict = None,
        joint_prior_loss: dict = None,
        silhouette_loss: dict = None,
        smooth_loss: dict = None,
        pose_prior_loss: dict = None,
        depth_loss: dict = None,
        use_one_betas_per_video: bool = False,
        ignore_keypoints: List[int] = None,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        verbose: bool = False,
        hparams: Union[dict, None] = None,
        visualizer=None,
    ) -> None:
        """
        Args:
            body_model: config or an object of body model.
            num_epochs: number of epochs of registration
            camera: config or an object of camera
            img_res: image resolution. If tuple, values are (width, height)
            stages: config of registration stages
            optimizer: config of optimizer
            keypoints2d_loss: config of keypoint 2D loss
            keypoints3d_loss: config of keypoint 3D loss
            vertices2d_loss: config of keypoint 2D loss
            shape_prior_loss: config of shape prior loss.
                Used to prevent extreme shapes.
            joint_prior_loss: config of joint prior loss.
                Used to prevent large joint rotations.
            smooth_loss: config of smooth loss.
                Used to prevent jittering by temporal smoothing.
            pose_prior_loss: config of pose prior loss.
                Used to prevent
            use_one_betas_per_video: whether to use the same beta parameters
                for all frames in a single video sequence.
            ignore_keypoints: list of keypoint names to ignore in keypoint
                loss computation
            device: torch device
            verbose: whether to print individual losses during registration

        Returns:
            None
        """

        self.use_one_betas_per_video = use_one_betas_per_video
        self.num_epochs = num_epochs
        self.img_res = img_res
        self.device = device
        self.stage_config = stages
        self.optimizer = optimizer
        self.visualizer = visualizer
        self.keypoints2d_mse_loss = build_loss(keypoints2d_loss)
        self.keypoints3d_mse_loss = build_loss(keypoints3d_loss)
        self.vertices2d_mse_loss = build_loss(vertices2d_loss)
        self.shape_prior_loss = build_loss(shape_prior_loss)
        self.joint_prior_loss = build_loss(joint_prior_loss)
        self.smooth_loss = build_loss(smooth_loss)
        self.pose_prior_loss = build_loss(pose_prior_loss)
        self.likelihood_loss = build_loss(likelihood_loss)
        self.segm_mask_loss = build_loss(segm_mask_loss)
        self.silhouette_loss = build_loss(silhouette_loss)
        self.depth_loss = build_loss(depth_loss)

        self.hparams = CN.load_cfg(str(hparams))
        self.neural_mesh_model_voge = build_neural_mesh_model_voge(hparams)
        self.clutter_bank = None

        if self.joint_prior_loss is not None:
            self.joint_prior_loss = self.joint_prior_loss.to(self.device)
        if self.smooth_loss is not None:
            self.smooth_loss = self.smooth_loss.to(self.device)
        if self.pose_prior_loss is not None:
            self.pose_prior_loss = self.pose_prior_loss.to(self.device)

        # initialize body model
        if isinstance(body_model, dict):
            self.body_model = build_body_model(body_model).to(self.device)
        elif isinstance(body_model, torch.nn.Module):
            self.body_model = body_model.to(self.device)
        else:
            raise TypeError(
                f'body_model should be either dict or '
                f'torch.nn.Module, but got {type(body_model)}'
            )

        # initialize camera
        if camera is not None:
            if isinstance(camera, dict):
                self.camera = build_cameras(camera).to(self.device)
            elif isinstance(camera, torch.nn.Module):
                self.camera = camera.to(device)
            else:
                raise TypeError(
                    f'camera should be either dict or '
                    f'torch.nn.Module, but got {type(camera)}'
                )

        self.v2p_onehot = get_vert_to_part(self.device)

        self.ignore_keypoints = ignore_keypoints
        self.verbose = verbose

        self.visualizer = None

        self._set_keypoint_idxs()

    def set_visualizer(self, visualizer):
        self.visualizer = visualizer

    def set_summary_writer(self, writer):
        self.writer = writer
        self.global_step = 0

    def set_clutter(self, clutter_bank):
        self.clutter_bank = clutter_bank

    def set_mesh_info(self, ds_indices, faces_down):
        self.ds_indices = ds_indices
        self.faces_down = faces_down

    def __call__(
        self,
        keypoints2d: torch.Tensor = None,
        keypoints2d_conf: torch.Tensor = None,
        keypoints3d: torch.Tensor = None,
        keypoints3d_conf: torch.Tensor = None,
        vertices2d: torch.Tensor = None,
        vertices2d_conf: torch.Tensor = None,
        pred_segm_mask: torch.Tensor = None,
        gt_mask: torch.Tensor = None,
        gt_depth: torch.Tensor = None,
        predicted_map: torch.Tensor = None,
        init_global_orient: torch.Tensor = None,
        init_transl: torch.Tensor = None,
        init_body_pose: torch.Tensor = None,
        init_betas: torch.Tensor = None,
        img: torch.Tensor = None,
        img_meta: dict = None,
        return_verts: bool = False,
        return_joints: bool = False,
        return_full_pose: bool = False,
        return_losses: bool = False,
    ) -> dict:
        """Run registration.

        Notes:
            B: batch size
            K: number of keypoints
            D: shape dimension
            Provide only keypoints2d or keypoints3d, not both.

        Args:
            keypoints2d: 2D keypoints of shape (B, K, 2)
            keypoints2d_conf: 2D keypoint confidence of shape (B, K)
            keypoints3d: 3D keypoints of shape (B, K, 3).
            keypoints3d_conf: 3D keypoint confidence of shape (B, K)
            init_global_orient: initial global_orient of shape (B, 3)
            init_transl: initial transl of shape (B, 3)
            init_body_pose: initial body_pose of shape (B, 69)
            init_betas: initial betas of shape (B, D)
            return_verts: whether to return vertices
            return_joints: whether to return joints
            return_full_pose: whether to return full pose
            return_losses: whether to return loss dict

        Returns:
            ret: a dictionary that includes body model parameters,
                and optional attributes such as vertices and joints
        """
        assert (
            keypoints2d is not None or keypoints3d is not None
        ), 'Neither of 2D nor 3D keypoints are provided.'
        # assert not (keypoints2d is not None and keypoints3d is not None), \
        #     'Do not provide both 2D and 3D keypoints.'
        batch_size = (
            keypoints2d.shape[0] if keypoints2d is not None else keypoints3d.shape[0]
        )

        global_orient = self._match_init_batch_size(
            init_global_orient, self.body_model.global_orient, batch_size
        )
        transl = self._match_init_batch_size(
            init_transl, self.body_model.transl, batch_size
        )
        body_pose = self._match_init_batch_size(
            init_body_pose, self.body_model.body_pose, batch_size
        )
        if init_betas is None and self.use_one_betas_per_video:
            betas = torch.zeros(1, self.body_model.betas.shape[-1]).to(self.device)
        else:
            betas = self._match_init_batch_size(
                init_betas, self.body_model.betas, batch_size
            )

        for i in range(self.num_epochs):
            for stage_idx, stage_config in enumerate(self.stage_config):
                # if img_meta is not None:
                #     img_meta['stage_idx'] = stage_idx

                self._optimize_stage(
                    global_orient=global_orient,
                    transl=transl,
                    body_pose=body_pose,
                    betas=betas,
                    keypoints2d=keypoints2d,
                    keypoints2d_conf=keypoints2d_conf,
                    keypoints3d=keypoints3d,
                    keypoints3d_conf=keypoints3d_conf,
                    vertices2d=vertices2d,
                    vertices2d_conf=vertices2d_conf,
                    pred_segm_mask=pred_segm_mask,
                    gt_mask=gt_mask,
                    gt_depth=gt_depth,
                    predicted_map=predicted_map,
                    img=img,
                    img_meta=img_meta,
                    **stage_config,
                )

        # collate results
        ret = {
            'global_orient': global_orient,
            'transl': transl,
            'body_pose': body_pose,
            'betas': betas,
        }

        if return_verts or return_joints or return_full_pose or return_losses:
            eval_ret = self.evaluate(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas,
                transl=transl,
                keypoints2d=keypoints2d,
                keypoints2d_conf=keypoints2d_conf,
                keypoints3d=keypoints3d,
                keypoints3d_conf=keypoints3d_conf,
                vertices2d=vertices2d,
                vertices2d_conf=vertices2d_conf,
                pred_segm_mask=pred_segm_mask,
                gt_mask=gt_mask,
                gt_depth=gt_depth,
                predicted_map=predicted_map,
                return_verts=return_verts,
                return_full_pose=return_full_pose,
                return_joints=return_joints,
                reduction_override='none',  # sample-wise loss
            )

            if return_verts:
                ret['vertices'] = eval_ret['vertices']
            if return_joints:
                ret['joints'] = eval_ret['joints']
            if return_full_pose:
                ret['full_pose'] = eval_ret['full_pose']
            if return_losses:
                for k in eval_ret.keys():
                    if 'loss' in k:
                        ret[k] = eval_ret[k]

        for k, v in ret.items():
            if isinstance(v, torch.Tensor):
                ret[k] = v.detach().clone()

        return ret

    def _optimize_stage(
        self,
        betas: torch.Tensor,
        body_pose: torch.Tensor,
        global_orient: torch.Tensor,
        transl: torch.Tensor,
        fit_global_orient: bool = True,
        fit_transl: bool = True,
        fit_body_pose: bool = True,
        fit_betas: bool = True,
        keypoints2d: torch.Tensor = None,
        keypoints2d_conf: torch.Tensor = None,
        keypoints2d_weight: float = None,
        keypoints3d: torch.Tensor = None,
        keypoints3d_conf: torch.Tensor = None,
        keypoints3d_weight: float = None,
        vertices2d: torch.Tensor = None,
        vertices2d_conf: torch.Tensor = None,
        pred_segm_mask: torch.Tensor = None,
        gt_mask: torch.Tensor = None,
        gt_depth: torch.Tensor = None,
        predicted_map: torch.Tensor = None,
        img: torch.Tensor = None,
        img_meta: dict = None,
        likelihood_weight: float = None,
        shape_prior_weight: float = None,
        joint_prior_weight: float = None,
        smooth_loss_weight: float = None,
        pose_prior_weight: float = None,
        segm_mask_loss_weight: float = None,
        silhouette_loss_weight: float = None,
        depth_loss_weight: float = None,
        vertices2d_weight: float = None,
        joint_weights: dict = {},
        num_iter: int = 1,
    ) -> None:
        """Optimize a stage of body model parameters according to
        configuration.

        Notes:
            B: batch size
            K: number of keypoints
            D: shape dimension

        Args:
            betas: shape (B, D)
            body_pose: shape (B, 69)
            global_orient: shape (B, 3)
            transl: shape (B, 3)
            fit_global_orient: whether to optimize global_orient
            fit_transl: whether to optimize transl
            fit_body_pose: whether to optimize body_pose
            fit_betas: whether to optimize betas
            keypoints2d: 2D keypoints of shape (B, K, 2)
            keypoints2d_conf: 2D keypoint confidence of shape (B, K)
            keypoints2d_weight: weight of 2D keypoint loss
            keypoints3d: 3D keypoints of shape (B, K, 3).
            keypoints3d_conf: 3D keypoint confidence of shape (B, K)
            keypoints3d_weight: weight of 3D keypoint loss
            shape_prior_weight: weight of shape prior loss
            joint_prior_weight: weight of joint prior loss
            smooth_loss_weight: weight of smooth loss
            pose_prior_weight: weight of pose prior loss
            joint_weights: per joint weight of shape (K, )
            num_iter: number of iterations

        Returns:
            None
        """

        parameters = OptimizableParameters()
        parameters.set_param(fit_global_orient, global_orient)
        parameters.set_param(fit_transl, transl)
        if self.hparams.get('fit_z', False):
            body_pose_org_rest = body_pose[:, -6:].detach().clone()
            with torch.no_grad():
                body_pose_z = self.pose_prior_loss.vposer.encode(
                    body_pose[:, :-6].float()
                ).mean[:, :32]
            parameters.set_param(fit_body_pose, body_pose_z)
        else:
            body_pose_z = None
            parameters.set_param(fit_body_pose, body_pose)
        parameters.set_param(fit_betas, betas)

        optimizer = build_optimizer(parameters, self.optimizer)

        opt_global_orient = global_orient.detach().clone()
        opt_transl = transl.detach().clone()
        opt_body_pose = body_pose.detach().clone()
        opt_betas = betas.detach().clone()
        opt_loss = torch.ones(body_pose.shape[0], device=body_pose.device) * float(
            'inf'
        )
        for iter_idx in range(num_iter):

            def closure():
                # nonlocal body_pose
                optimizer.zero_grad()
                if self.hparams.get('fit_z', False):
                    body_pose_closure = torch.cat(
                        (
                            self.pose_prior_loss.vposer.decode(
                                body_pose_z, output_type='aa'
                            )
                            .squeeze(1)
                            .flatten(1, 2),
                            body_pose_org_rest,
                        ),
                        dim=1,
                    )
                else:
                    body_pose_closure = body_pose
                betas_video = self._expand_betas(body_pose_closure.shape[0], betas)
                loss_dict = self.evaluate(
                    iter=iter_idx,
                    global_orient=global_orient,
                    body_pose=body_pose_closure,
                    body_pose_z=body_pose_z,
                    betas=betas_video,
                    transl=transl,
                    keypoints2d=keypoints2d,
                    keypoints2d_conf=keypoints2d_conf,
                    keypoints2d_weight=keypoints2d_weight,
                    keypoints3d=keypoints3d,
                    keypoints3d_conf=keypoints3d_conf,
                    keypoints3d_weight=keypoints3d_weight,
                    vertices2d=vertices2d,
                    vertices2d_conf=vertices2d_conf,
                    pred_segm_mask=pred_segm_mask,
                    gt_mask=gt_mask,
                    gt_depth=gt_depth,
                    predicted_map=predicted_map,
                    likelihood_weight=likelihood_weight,
                    joint_prior_weight=joint_prior_weight,
                    shape_prior_weight=shape_prior_weight,
                    smooth_loss_weight=smooth_loss_weight,
                    pose_prior_weight=pose_prior_weight,
                    segm_mask_loss_weight=segm_mask_loss_weight,
                    silhouette_loss_weight=silhouette_loss_weight,
                    depth_loss_weight=depth_loss_weight,
                    vertices2d_weight=vertices2d_weight,
                    joint_weights=joint_weights,
                    img=img,
                    img_meta=img_meta,
                    reduction_override='none',
                )
                self.global_step += 1

                loss = loss_dict['total_loss']
                update = loss < opt_loss
                opt_loss[update] = loss[update].detach().clone()
                opt_global_orient[update, :] = global_orient[update, :].detach().clone()
                # opt_global_orient = global_orient[update, :].detach().clone()
                opt_body_pose[update, :] = body_pose_closure[update, :].detach().clone()
                opt_betas[update, :] = betas[update, :].detach().clone()
                opt_transl[update, :] = transl[update, :].detach().clone()

                loss = loss.sum()
                if self.hparams.show_debug_info:
                    log_str = ''
                    log_items = []
                    log_items.append(f'iter: {iter_idx}')
                    for name, val in loss_dict.items():
                        # TODO: resolve this hack
                        # these items have been in log_str
                        if name in [
                            'mode',
                            'Epoch',
                            'iter',
                            'lr',
                            'time',
                            'data_time',
                            'memory',
                            'epoch',
                        ]:
                            continue
                        if isinstance(val, torch.Tensor):
                            val = val.mean().item()
                        if isinstance(val, float):
                            val = f'{val:.4f}'
                        log_items.append(f'{name}: {val}')
                    log_str += ', '.join(log_items)
                    logger.info(log_str)
                loss.backward()
                # print(betas)
                # print(betas.grad)
                return loss

            # import ipdb; ipdb.set_trace()
            optimizer.step(closure)

        with torch.no_grad():
            global_orient[:] = opt_global_orient
            transl[:] = opt_transl
            body_pose[:] = opt_body_pose
            betas[:] = opt_betas

    def detect_occlusion_mask(
        self,
        predicted_map: torch.Tensor,
        projected_map: torch.Tensor,
        clutter_scores: torch.Tensor,
        mask: torch.Tensor,
        pixel_orient_weights: torch.Tensor,
        n_orient: int,
        options: dict,
        pixel_var: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Finds occlusion regions by comparing object score and clutter score.

        Args:
            predicted_map: B, C, H, W
            projected_map: B, C, H, W
            mask: B, H, W
            pixel_orient_weights: B, N_orient, H, W
            options: hyperparamters
            pixel_var: pixel-wise variance
        Returns:
            rasterization_loss: (B,) or (B, H, W)
        """
        B, _, H, W = projected_map.shape
        projected_map = projected_map.view(
            B, n_orient, -1, H, W
        )  # B, N_orient, C, H, W
        projected_map = torch.nn.functional.normalize(projected_map, dim=2, p=2)

        object_score = torch.sum(
            projected_map * predicted_map.unsqueeze(1), dim=2
        )  # B, N_orient, H, W

        options.bUseVariance = False
        if pixel_var is not None:
            # pixel_var: B, N_orient, H, W
            pixel_var += (1 - mask) * 1e8

        if n_orient > 1:
            if options.bUseVariance:
                object_score = (pixel_orient_weights * object_score / (pixel_var)).sum(
                    1
                )
            else:
                object_score = (pixel_orient_weights * object_score).sum(1)
        else:
            if options.bUseVariance:
                object_score = object_score / pixel_var
            else:
                object_score = object_score

        # object_score: B, H, W
        # clutter_scores: B, H, W

        # Rasterization loss
        if options.bUseVariance:
            clutter_scores = clutter_scores / max(
                pixel_var.min(dim=1)[0][mask > 0].min(), 1e-8
            )  # approximate clutter variance
        else:
            clutter_scores = clutter_scores
        return clutter_scores > object_score

    def evaluate(
        self,
        iter=None,
        betas: torch.Tensor = None,
        body_pose: torch.Tensor = None,
        body_pose_z: torch.Tensor = None,
        global_orient: torch.Tensor = None,
        transl: torch.Tensor = None,
        keypoints2d: torch.Tensor = None,
        keypoints2d_conf: torch.Tensor = None,
        keypoints2d_weight: float = None,
        keypoints3d: torch.Tensor = None,
        keypoints3d_conf: torch.Tensor = None,
        keypoints3d_weight: float = None,
        vertices2d: torch.Tensor = None,
        vertices2d_conf: torch.Tensor = None,
        pred_segm_mask: torch.Tensor = None,
        gt_mask: torch.Tensor = None,
        predicted_map: torch.Tensor = None,
        gt_depth: torch.Tensor = None,
        img: torch.Tensor = None,
        img_meta: dict = None,
        likelihood_weight: float = None,
        shape_prior_weight: float = None,
        joint_prior_weight: float = None,
        smooth_loss_weight: float = None,
        pose_prior_weight: float = None,
        segm_mask_loss_weight: float = None,
        silhouette_loss_weight: float = None,
        depth_loss_weight: float = None,
        vertices2d_weight: float = None,
        joint_weights: dict = {},
        return_verts: bool = False,
        return_full_pose: bool = False,
        return_joints: bool = False,
        return_occ_mask: bool = False,
        reduction_override: str = None,
    ) -> dict:
        """Evaluate fitted parameters through loss computation. This function
        serves two purposes: 1) internally, for loss backpropagation 2)
        externally, for fitting quality evaluation.

        Notes:
            B: batch size
            K: number of keypoints
            D: shape dimension

        Args:
            betas: shape (B, D)
            body_pose: shape (B, 69)
            global_orient: shape (B, 3)
            transl: shape (B, 3)
            keypoints2d: 2D keypoints of shape (B, K, 2)
            keypoints2d_conf: 2D keypoint confidence of shape (B, K)
            keypoints2d_weight: weight of 2D keypoint loss
            keypoints3d: 3D keypoints of shape (B, K, 3).
            keypoints3d_conf: 3D keypoint confidence of shape (B, K)
            keypoints3d_weight: weight of 3D keypoint loss
            shape_prior_weight: weight of shape prior loss
            joint_prior_weight: weight of joint prior loss
            smooth_loss_weight: weight of smooth loss
            pose_prior_weight: weight of pose prior loss
            joint_weights: per joint weight of shape (K, )
            return_verts: whether to return vertices
            return_joints: whether to return joints
            return_full_pose: whether to return full pose
            reduction_override: reduction method, e.g., 'none', 'sum', 'mean'

        Returns:
            ret: a dictionary that includes body model parameters,
                and optional attributes such as vertices and joints
        """

        ret = {}

        # remove hand poses
        body_pose_nohands = body_pose.clone()
        body_pose_nohands[:, -12:] = 0
        body_model_output = self.body_model(
            global_orient=global_orient,
            body_pose=body_pose_nohands,
            betas=betas,
            transl=transl,
            return_verts=True,
            return_full_pose=return_full_pose,
        )

        model_joints = body_model_output['joints']
        model_joint_mask = body_model_output['joint_mask']
        model_vertices = body_model_output['vertices']
        vertices_ds = model_vertices[:, self.ds_indices]

        cameras = get_cameras(
            self.hparams.FOCAL_LENGTH / 4,
            self.hparams.RENDER_RES,
            torch.zeros_like(transl).to('cuda'),
        ).to('cuda')

        self.neural_mesh_model_voge.rasterizer.cameras = cameras
        vert_orient_weights = get_vert_orient_weights(
            model_joints[:, :25],
            cameras,
            self.hparams.n_orient,
            theta=self.hparams.theta,
        )
        ras_out = self.neural_mesh_model_voge(
            vertices_ds,
            self.faces_down.expand(vertices_ds.shape[0], -1, -1),
            vert_orient_weights=vert_orient_weights,
            vert_part=self.v2p_onehot[self.ds_indices],
        )

        mask = ras_out['mask'].squeeze(1)
        mask_bp = ras_out['mask_bp']
        proj_segm_mask = ras_out['pixel_part']
        proj_segm_mask = (proj_segm_mask.argmax(axis=1).detach() * mask).long()
        projected_map = ras_out['projected_map']
        vert_visibility = ras_out['vert_visibility']
        pixel_orient_weights = ras_out['pixel_orient_weights']

        # Normalize CoKe features
        predicted_map = F.normalize(predicted_map, p=2, dim=1)
        with torch.no_grad():
            # [B, C, H, W] * [N, C, 1, 1]
            clutter_scores = torch.nn.functional.conv2d(
                predicted_map, self.clutter_bank.unsqueeze(2).unsqueeze(3)
            ).max(dim=1)[
                0
            ]  # B, H, W

        if img_meta is not None and self.hparams.get('use_gt_occ', False):
            for i, meta in enumerate(img_meta):
                img_size = clutter_scores.shape[-1]
                img_size_org = clutter_scores.shape[-1] * self.hparams.downsample_rate
                ratio = img_size_org / constants.IMG_RES
                occ_size = (meta['occ_size'] * ratio) // self.hparams.downsample_rate

                occ_stride = meta['occ_stride']
                h, w = meta['occ_idx']
                occ_stride = int(round(occ_stride * ratio))
                h_start = h * occ_stride
                w_start = w * occ_stride

                h_end = int(max(0, min(h_start + occ_size, img_size)))
                w_end = int(max(0, min(w_start + occ_size, img_size)))
                clutter_scores[i, h_start:h_end, w_start:w_end] = 1.0
        # import ipdb; ipdb.set_trace()

        loss_dict = self._compute_loss(
            model_joints,
            model_joint_mask,
            keypoints2d=keypoints2d,
            keypoints2d_conf=keypoints2d_conf,
            keypoints2d_weight=keypoints2d_weight,
            keypoints3d=keypoints3d,
            keypoints3d_conf=keypoints3d_conf,
            keypoints3d_weight=keypoints3d_weight,
            model_vertices3d=vertices_ds,
            model_vertices=model_vertices,
            vertices2d=vertices2d,
            vertices2d_conf=vertices2d_conf,
            vert_visibility=vert_visibility,
            proj_segm_mask=proj_segm_mask,
            pred_segm_mask=pred_segm_mask,
            mask=mask,
            mask_bp=mask_bp,
            gt_mask=gt_mask,
            gt_depth=gt_depth,
            pixel_orient_weights=pixel_orient_weights,
            predicted_map=predicted_map,
            projected_map=projected_map,
            clutter_scores=clutter_scores,
            likelihood_weight=likelihood_weight,
            joint_prior_weight=joint_prior_weight,
            vertices2d_weight=vertices2d_weight,
            shape_prior_weight=shape_prior_weight,
            smooth_loss_weight=smooth_loss_weight,
            pose_prior_weight=pose_prior_weight,
            joint_weights=joint_weights,
            segm_mask_loss_weight=segm_mask_loss_weight,
            silhouette_loss_weight=silhouette_loss_weight,
            depth_loss_weight=depth_loss_weight,
            reduction_override=reduction_override,
            body_pose=body_pose,
            body_pose_z=body_pose_z,
            betas=betas,
        )
        ret.update(loss_dict)

        if return_verts:
            ret['vertices'] = body_model_output['vertices']
        if return_full_pose:
            ret['full_pose'] = body_model_output['full_pose']
        if return_joints:
            ret['joints'] = model_joints
        if return_occ_mask:
            ret['occ_masks'] = self.detect_occlusion_mask(
                predicted_map,
                projected_map,
                clutter_scores,
                mask,
                pixel_orient_weights,
                self.hparams.n_orient,
                self.hparams,
            )
            ret['clutter_scores'] = clutter_scores

            B, _, H, W = projected_map.shape
            projected_map = projected_map.view(
                B, self.hparams.n_orient, -1, H, W
            )  # B, N_orient, C, H, W
            projected_map = torch.nn.functional.normalize(projected_map, dim=2, p=2)

            object_scores = torch.sum(projected_map * predicted_map.unsqueeze(1), dim=2)
            object_scores = (pixel_orient_weights * object_scores).sum(1)
            object_scores = (
                -torch.ones_like(object_scores) * (1 - mask) + object_scores * mask
            )

            ret['object_scores'] = object_scores

            ret['mask_bp'] = mask_bp

            projected_vertices_xyd = (
                self.neural_mesh_model_voge.rasterizer.cameras.transform_points_screen(
                    vertices_ds,
                    image_size=((self.hparams.RENDER_RES, self.hparams.RENDER_RES),),
                )
            )
            projected_vertices = projected_vertices_xyd[..., :2]
            ret['projected_vertices'] = projected_vertices

        # TODO debug
        if img is not None and self.visualizer is not None:
            camera = convert_perspective_to_weak_perspective(
                transl,
                focal_length=self.hparams.FOCAL_LENGTH,
                img_res=self.hparams.IMG_RES,
            )

            img_r, img_r_rot = visualize_prediction_pth(
                img,
                torch.cat((global_orient, body_pose_nohands), dim=1),
                betas,
                camera,
                self.visualizer,
                vis_res=512,
                device=img.device,
            )
            visualization = (
                torch.cat((img.cpu(), img_r, img_r_rot), dim=2).clamp(0, 255).byte()
            )
            for i in range(len(img)):
                image_path = img_meta[i]['image_path'].replace('/', '_')
                self.writer.add_image(
                    image_path, visualization[i], self.global_step, dataformats='HWC'
                )

        return ret

    def _compute_loss(
        self,
        model_joints: torch.Tensor,
        model_joint_conf: torch.Tensor,
        keypoints2d: torch.Tensor = None,
        keypoints2d_conf: torch.Tensor = None,
        keypoints2d_weight: float = None,
        keypoints3d: torch.Tensor = None,
        keypoints3d_conf: torch.Tensor = None,
        keypoints3d_weight: float = None,
        model_vertices3d: torch.Tensor = None,
        model_vertices: torch.Tensor = None,
        vertices2d: torch.Tensor = None,
        vertices2d_conf: torch.Tensor = None,
        vert_visibility: torch.Tensor = None,
        proj_segm_mask: torch.Tensor = None,
        pred_segm_mask: torch.Tensor = None,
        mask: torch.Tensor = None,
        mask_bp: torch.Tensor = None,
        gt_mask: torch.Tensor = None,
        gt_depth: torch.Tensor = None,
        pixel_orient_weights: torch.Tensor = None,
        predicted_map: torch.Tensor = None,
        projected_map: torch.Tensor = None,
        clutter_scores: torch.Tensor = None,
        likelihood_weight: float = None,
        shape_prior_weight: float = None,
        joint_prior_weight: float = None,
        smooth_loss_weight: float = None,
        pose_prior_weight: float = None,
        segm_mask_loss_weight: float = None,
        silhouette_loss_weight: float = None,
        depth_loss_weight: float = None,
        vertices2d_weight: float = None,
        joint_weights: dict = {},
        reduction_override: str = None,
        body_pose: torch.Tensor = None,
        body_pose_z: torch.Tensor = None,
        betas: torch.Tensor = None,
    ):
        """Loss computation.

        Notes:
            B: batch size
            K: number of keypoints
            D: shape dimension

        Args:
            model_joints: 3D joints regressed from body model of shape (B, K)
            model_joint_conf: 3D joint confidence of shape (B, K). It is
                normally all 1, except for zero-pads due to convert_kps in
                the SMPL wrapper.
            keypoints2d: 2D keypoints of shape (B, K, 2)
            keypoints2d_conf: 2D keypoint confidence of shape (B, K)
            keypoints2d_weight: weight of 2D keypoint loss
            keypoints3d: 3D keypoints of shape (B, K, 3).
            keypoints3d_conf: 3D keypoint confidence of shape (B, K)
            keypoints3d_weight: weight of 3D keypoint loss
            shape_prior_weight: weight of shape prior loss
            joint_prior_weight: weight of joint prior loss
            smooth_loss_weight: weight of smooth loss
            pose_prior_weight: weight of pose prior loss
            joint_weights: per joint weight of shape (K, )
            reduction_override: reduction method, e.g., 'none', 'sum', 'mean'
            body_pose: shape (B, 69), for loss computation
            betas: shape (B, D), for loss computation

        Returns:
            losses: a dict that contains all losses
        """
        losses = {}
        weight = self._get_weight(**joint_weights)
        if self.segm_mask_loss is not None:
            seg_loss = self.segm_mask_loss(
                pred_segm_mask,
                proj_segm_mask,
                weight=torch.ones_like(proj_segm_mask),
                reduction_override=reduction_override,
                loss_weight_override=segm_mask_loss_weight,
            )
            losses['seg_loss'] = seg_loss

        if self.silhouette_loss is not None:
            if gt_mask is None:
                bg_threshold = 0.2
                gt_mask = clutter_scores < bg_threshold
            projected_vertices_xyd = (
                self.neural_mesh_model_voge.rasterizer.cameras.transform_points_screen(
                    model_vertices,
                    image_size=((self.hparams.RENDER_RES, self.hparams.RENDER_RES),),
                )
            )
            projected_vertices = projected_vertices_xyd[..., :2]
            silhouette_loss = self.silhouette_loss(
                projected_vertices,
                (gt_mask > 0),
                loss_weight_override=silhouette_loss_weight,
            )
            losses['silhouette_loss'] = silhouette_loss

        if self.likelihood_loss is not None:
            likelihood_loss = self.likelihood_loss(
                projected_map,
                predicted_map,
                clutter_scores=clutter_scores,
                mask=mask,
                pixel_orient_weights=pixel_orient_weights,
                n_orient=self.hparams.n_orient,
                pixel_var=None,
                options=self.hparams,
                reduction_override=reduction_override,
                loss_weight_override=likelihood_weight,
            )
            losses['likelihood_loss'] = likelihood_loss

        # 2D keypoint loss
        if keypoints2d is not None:
            head_ids = [0, 15, 16, 17, 18]
            invalid = torch.logical_or(
                (
                    (
                        keypoints2d_conf[:, head_ids] > self.hparams.thr_openpose_conf
                    ).sum(dim=1)
                    == 0
                ),
                (keypoints2d_conf > self.hparams.thr_openpose_conf).sum(dim=1)
                < self.hparams.thr_detected_joints,
            )
            keypoints2d_conf[invalid, :] = 0
            keypoints2d = keypoints2d / self.hparams.downsample_rate

            projected_joints_xyd = (
                self.neural_mesh_model_voge.rasterizer.cameras.transform_points_screen(
                    model_joints,
                    image_size=((self.hparams.RENDER_RES, self.hparams.RENDER_RES),),
                )
            )

            projected_joints = projected_joints_xyd[..., :2]

            keypoint2d_loss = self.keypoints2d_mse_loss(
                pred=projected_joints,
                pred_conf=model_joint_conf,
                target=keypoints2d,
                target_conf=keypoints2d_conf,
                keypoint_weight=weight,
                loss_weight_override=keypoints2d_weight,
                reduction_override=reduction_override,
            )
            losses['keypoint2d_loss'] = keypoint2d_loss

        projected_vertices_xyd = (
            self.neural_mesh_model_voge.rasterizer.cameras.transform_points_screen(
                model_vertices3d,
                image_size=((self.hparams.RENDER_RES, self.hparams.RENDER_RES),),
            )
        )
        projected_vertices = projected_vertices_xyd[..., :2]
        # 2D vertex loss
        if vertices2d is not None:
            model_vertex_conf = torch.ones(
                projected_vertices.shape[:2], device=projected_vertices.device
            )
            vert_visibility_ = torch.ones_like(vert_visibility)
            vertices2d_conf = (
                vertices2d_conf
                * vert_visibility_
                * (vertices2d_conf > self.hparams.thr_vert_det)
            )
            vertices2d_loss = self.vertices2d_mse_loss(
                pred=projected_vertices,
                pred_conf=model_vertex_conf,
                target=vertices2d,
                target_conf=vertices2d_conf,
                loss_weight_override=vertices2d_weight,
                reduction_override=reduction_override,
            )

            losses['vertices2d_loss'] = vertices2d_loss

        if self.depth_loss is not None:
            # gt_depth N, 1, H, W
            projected_vertices = (
                projected_vertices
                / torch.tensor(gt_depth.shape[-2:], device=gt_depth.device).float()
            )
            gt_vert_depth = sample_feature(255 - gt_depth, projected_vertices).squeeze(
                1
            )  # convert to depth
            # import ipdb; ipdb.set_trace()
            vert_depth = model_vertices3d[:, :, 2]  # N, V
            depth_loss = self.depth_loss(
                pred=vert_depth * vert_visibility,
                target=gt_vert_depth * vert_visibility,
                loss_weight_override=depth_loss_weight,
            )

            losses['depth_loss'] = depth_loss

        # 3D keypoint loss
        if keypoints3d is not None:
            H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
            H36M_TO_J14 = H36M_TO_J17[:14]

            if keypoints3d.shape[1] == 17:
                keypoints3d = keypoints3d[:, H36M_TO_J14]
                keypoints3d_conf = keypoints3d_conf[:, H36M_TO_J14]

            model_joints_cvt, model_joints_mask = convert_kps(
                model_joints,
                src='smpl_49',
                dst='h36m',
                mask=torch.ones_like(model_joint_conf[0]),
            )
            dst_idxs, src_idxs, _ = get_mapping('smpl_49', 'h36m')

            weight_cvt = weight[src_idxs][H36M_TO_J14]
            model_joints_cvt = model_joints_cvt[:, H36M_TO_J14]
            model_joint_conf_cvt = model_joint_conf[:, src_idxs][:, H36M_TO_J14]
            keypoints3d_loss = self.keypoints3d_mse_loss(
                pred=model_joints_cvt - model_joints_cvt[:, 0:1],
                pred_conf=model_joint_conf_cvt,
                target=keypoints3d - keypoints3d[:, 0:1],
                target_conf=keypoints3d_conf,
                keypoint_weight=weight_cvt,
                loss_weight_override=keypoints3d_weight,
                reduction_override=reduction_override,
            )
            losses['keypoints3d_loss'] = keypoints3d_loss

        # regularizer to prevent betas from taking large values
        if self.shape_prior_loss is not None:
            shape_prior_loss = self.shape_prior_loss(
                betas=betas,
                loss_weight_override=shape_prior_weight,
                reduction_override=reduction_override,
            )
            losses['shape_prior_loss'] = shape_prior_loss

        # joint prior loss
        if self.joint_prior_loss is not None:
            joint_prior_loss = self.joint_prior_loss(
                body_pose=body_pose,
                loss_weight_override=joint_prior_weight,
                reduction_override=reduction_override,
            )
            losses['joint_prior_loss'] = joint_prior_loss

        # smooth body loss
        if self.smooth_loss is not None:
            smooth_loss = self.smooth_loss(
                body_pose=body_pose,
                loss_weight_override=smooth_loss_weight,
                reduction_override=reduction_override,
            )
            losses['smooth_loss'] = smooth_loss

        # pose prior loss
        if self.pose_prior_loss is not None:
            if self.hparams.get('fit_z', False) and body_pose_z is None:
                with torch.no_grad():
                    body_pose_z = self.pose_prior_loss.vposer.encode(
                        body_pose[:, :-6].float()
                    ).mean[:, :32]
            pose_prior_loss = self.pose_prior_loss(
                body_pose=body_pose_z
                if self.hparams.get('fit_z', False)
                else body_pose,
                loss_weight_override=pose_prior_weight,
                reduction_override=reduction_override,
            )
            losses['pose_prior_loss'] = pose_prior_loss

        if self.verbose:
            msg = ''
            for loss_name, loss in losses.items():
                msg += f'{loss_name}={loss.mean().item():.6f}'
            print(msg)

        total_loss = 0
        for loss_name, loss in losses.items():
            if loss.ndim == 3:
                total_loss = total_loss + loss.sum(dim=(2, 1))
            elif loss.ndim == 2:
                total_loss = total_loss + loss.sum(dim=-1)
            else:
                total_loss = total_loss + loss
        losses['total_loss'] = total_loss
        return losses

    def _match_init_batch_size(
        self,
        init_param: torch.Tensor,
        init_param_body_model: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """A helper function to ensure body model parameters have the same
        batch size as the input keypoints.

        Args:
            init_param: input initial body model parameters, may be None
            init_param_body_model: initial body model parameters from the
                body model
            batch_size: batch size of keypoints

        Returns:
            param: body model parameters with batch size aligned
        """

        # param takes init values
        param = (
            init_param.detach().clone()
            if init_param is not None
            else init_param_body_model.detach().clone()
        )

        # expand batch dimension to match batch size
        param_batch_size = param.shape[0]
        if param_batch_size != batch_size:
            if param_batch_size == 1:
                param = param.repeat(batch_size, *[1] * (param.ndim - 1))
            else:
                raise ValueError(
                    'Init param does not match the batch size of '
                    'keypoints, and is not 1.'
                )

        # shape check
        assert param.shape[0] == batch_size
        assert (
            param.shape[1:] == init_param_body_model.shape[1:]
        ), f'Shape mismatch: {param.shape} vs {init_param_body_model.shape}'

        return param

    def _set_keypoint_idxs(self) -> None:
        """Set keypoint indices to 1) body parts to be assigned different
        weights 2) be ignored for keypoint loss computation.

        Returns:
            None
        """
        convention = self.body_model.keypoint_dst

        # obtain ignore keypoint indices
        if self.ignore_keypoints is not None:
            self.ignore_keypoint_idxs = []
            for keypoint_name in self.ignore_keypoints:
                keypoint_idx = get_keypoint_idx(keypoint_name, convention=convention)
                if keypoint_idx != -1:
                    self.ignore_keypoint_idxs.append(keypoint_idx)

        # obtain body part keypoint indices
        shoulder_keypoint_idxs = get_keypoint_idxs_by_part(
            'shoulder', convention=convention
        )
        hip_keypoint_idxs = get_keypoint_idxs_by_part('hip', convention=convention)
        self.shoulder_hip_keypoint_idxs = [*shoulder_keypoint_idxs, *hip_keypoint_idxs]

    def _get_weight(
        self, use_shoulder_hip_only: bool = False, body_weight: float = 1.0
    ) -> torch.Tensor:
        """Get per keypoint weight.

        Notes:
            K: number of keypoints

        Args:
            use_shoulder_hip_only: whether to use only shoulder and hip
                keypoints for loss computation. This is useful in the
                warming-up stage to find a reasonably good initialization.
            body_weight: weight of body keypoints. Body part segmentation
                definition is included in the HumanData convention.

        Returns:
            weight: per keypoint weight tensor of shape (K)
        """

        num_keypoint = self.body_model.num_joints

        if use_shoulder_hip_only:
            weight = torch.zeros([num_keypoint]).to(self.device)
            weight[self.shoulder_hip_keypoint_idxs] = 1.0
            weight = weight * body_weight
        else:
            weight = torch.ones([num_keypoint]).to(self.device)
            weight = weight * body_weight

        if hasattr(self, 'ignore_keypoint_idxs'):
            weight[self.ignore_keypoint_idxs] = 0.0

        return weight

    def _expand_betas(self, batch_size, betas):
        """A helper function to expand the betas's first dim to match batch
        size such that the same beta parameters can be used for all frames in a
        video sequence.

        Notes:
            B: batch size
            K: number of keypoints
            D: shape dimension

        Args:
            batch_size: batch size
            betas: shape (B, D)

        Returns:
            betas_video: expanded betas
        """
        # no expansion needed
        if batch_size == betas.shape[0]:
            return betas

        # first dim is 1
        else:
            feat_dim = betas.shape[-1]
            betas_video = betas.view(1, feat_dim).expand(batch_size, feat_dim)

        return betas_video
