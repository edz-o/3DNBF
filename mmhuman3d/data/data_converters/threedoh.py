import os
import pickle
import json
from typing import List
from yacs.config import CfgNode as CN

import cv2
import numpy as np
from tqdm import tqdm

import torch

import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (look_at_view_transform, FoVPerspectiveCameras,
                                PerspectiveCameras, PointLights,
                                DirectionalLights, Materials,
                                RasterizationSettings, MeshRenderer,
                                MeshRasterizer, SoftPhongShader, TexturesUV,
                                TexturesVertex, HardPhongShader)

from mmhuman3d.core.cameras.camera_parameters import CameraParameter
from mmhuman3d.data.data_converters.builder import DATA_CONVERTERS
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.core.conventions.keypoints_mapping import convert_kps

from mmhuman3d.utils.geometry import (
    # batch_rodrigues,
    estimate_translation,
    # project_points,
    # rotation_matrix_to_angle_axis,
    # convert_weak_perspective_to_perspective
)
from mmhuman3d.utils.neural_renderer import (
    build_neural_renderer,
    get_blend_params,
    get_cameras)

from mmhuman3d.utils.image_utils import get_mask_and_visibility
from .base_converter import BaseModeConverter

from mmhuman3d.models.builder import build_body_model

from .base_converter import BaseConverter
from .builder import DATA_CONVERTERS

def set_up_renderer(R, T, f, c, image_size, device='cuda'):
    """
    (Lagacy) image_size is Tuple (w, h) 
    This is (w, h) in pytorch3d==0.4.0 but should be (h, w) for pytorch3d>=0.5.0
    """
    
    Rx = torch.diag(torch.tensor([-1, -1, 1])).unsqueeze(0).type_as(R)
    R = torch.bmm(Rx, R)
    T = torch.bmm(Rx, T.unsqueeze(2)).squeeze(2)
    
    cameras = PerspectiveCameras(device=device,
                                R=R.transpose(1, 2),
                                T=T, 
                                focal_length=f,
                                principal_point=c,
                                image_size=(image_size,)) 

    raster_settings = RasterizationSettings(
        image_size=image_size[::-1],
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    
    rasterizer = MeshRasterizer(cameras=cameras,
                                raster_settings=raster_settings)
    lights = PointLights(device=device, location=[[0.0, 0.0, -2.0]])
    renderer = MeshRenderer(rasterizer=rasterizer,
                            shader=HardPhongShader(device=device,
                                                cameras=cameras,
                                                lights=lights))

    return cameras, rasterizer, renderer

@DATA_CONVERTERS.register_module()
class ThreeDOHConverter(BaseModeConverter):
    """3D Poses in the Wild dataset `Object-Occluded Human Shape and Pose Estimation
     from a Single Color Image' CVPR'2020 More details can be found in
    the `paper.

    <https://https://www.yangangwang.com/papers/ZHANG-OOH-2020-03.html>`__ .

    Args:
        modes (list): 'test' and/or 'train' for accepted modes
    """

    ACCEPTED_MODES = ['train', 'test']

    def __init__(self, modes: List = [], device='cpu') -> None:
        super(ThreeDOHConverter, self).__init__(modes)
        self.device = device
        self.body_model = build_body_model(dict(
            type='SMPL',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='data/body_models/smpl',
            joints_regressor='data/body_models/J_regressor_h36m.npy'),).to(self.device)
        self.neural_renderer_gt = build_neural_renderer(CN.load_cfg(str(
                                                dict(SIGMA=0,
                                                    GAMMA=1e-2,
                                                    FACES_PER_PIXEL=1,
                                                    RENDER_RES=(1920, 1080))
                                                )))

    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:
        """
        Args:
            dataset_path (str): Path to directory where raw images and
            annotations are stored.
            out_path (str): Path to directory to save preprocessed npz file
            mode (str): Mode in accepted modes

        Returns:
            dict:
                A dict containing keys image_path, bbox_xywh, smpl
                stored in HumanData() format
        """
        # use HumanData to store all data
        human_data = HumanData()

        # structs we use
        image_path_, bbox_xywh_, cam_param_ = [], [], []
        keypoints2d_ = []
        smpl = {}
        smpl['body_pose'] = []
        smpl['global_orient'] = []
        smpl['betas'] = []

        root_path = dataset_path
        #os.path.join(, )

        annots = json.load(open(os.path.join(root_path, mode+'set', 'annots.json')))

        # go through all the .pkl files
        for imgname, annot in tqdm(annots.items()):
            
            K = np.array(annot['intri'])

            bbox_xyxy = np.array(annot['bbox']).reshape(-1)
            bbox_xyxy = self._bbox_expand(
                bbox_xyxy, scale_factor=1.2)
            bbox_xywh = self._xyxy2xywh(bbox_xyxy)
            
            scale = annot['scale'][0]
            image_path = os.path.join(mode+'set', annot['img_path'].replace('\\', '/'))
            image_abs_path = os.path.join(root_path, image_path)
            h, w, _ = cv2.imread(image_abs_path).shape

            # transform global pose
            pose = np.array(annot['pose'])[0]
            extrinsic_param = np.array(annot['extri'])
            R = extrinsic_param[:3, :3]
            T = extrinsic_param[:3, 3]
            
            camera = CameraParameter(H=h, W=w)
            camera.set_KRT(K, R, T)
            parameter_dict = camera.to_dict()
            
            # Compute keypoints2d
            betas = np.array(annot['betas'])[0]
            trans = np.array(annot['trans'])[0]

            
            # keypoints2d = np.array(annot['lsp_joints_2d'])
            # keypoints2d = np.concatenate((keypoints2d, np.ones([keypoints2d.shape[0], 1])), axis=1)
            # keypoints2d_.append(keypoints2d)
            
            # Has BUG
            array2tensor = lambda x: torch.from_numpy(x).float().unsqueeze(0).to(self.device)
            tensor2array = lambda x: x[0].detach().cpu().numpy()
            R, T, f, c, betas, pose, trans = [array2tensor(x) for x in [R, T, 
                                                                np.array([K[0, 0], K[1, 1]]), 
                                                                np.array([K[0, 2], K[1, 2]]), 
                                                                betas, pose, trans]]

            cameras, rasterizer, renderer = set_up_renderer(R, T, f, c, (w, h), self.device)

            with torch.no_grad():
                gt_output = self.body_model(
                        betas=betas, 
                        body_pose=pose[:, 3:], 
                        global_orient=pose[:, :3], 
                        )
                verts = gt_output['vertices'] + trans
                gt_keypoints3d = gt_output['joints'] + trans
                gt_keypoints2d = cameras.transform_points_screen(gt_keypoints3d, image_size=((w, h), ))[:, :, :2].squeeze(0)
                keypoints2d = np.ones([gt_keypoints2d.shape[0], 3])
                keypoints2d[:, :2] = gt_keypoints2d.cpu().numpy()

                keypoints2d_.append(keypoints2d)

            R = tensor2array(R) # recover R
            pose = tensor2array(pose)
            betas = tensor2array(betas)

            pose[:3] = cv2.Rodrigues(
                np.dot(R,
                        cv2.Rodrigues(pose[:3])[0]))[0].T[0]

            image_path_.append(image_path)
            bbox_xywh_.append(bbox_xywh)
            smpl['body_pose'].append(pose[3:].reshape((23, 3)))
            smpl['global_orient'].append(pose[:3])
            smpl['betas'].append(betas)
            cam_param_.append(parameter_dict)
            # break

        # change list to np array
        # import pdb; pdb.set_trace()
        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        smpl['body_pose'] = np.array(smpl['body_pose']).reshape((-1, 23, 3))
        smpl['global_orient'] = np.array(smpl['global_orient']).reshape(
            (-1, 3))
        smpl['betas'] = np.array(smpl['betas']).reshape((-1, 10))

        keypoints2d_ = np.array(keypoints2d_)
        # keypoints2d_, keypoints2d_mask = convert_kps(keypoints2d_, 'lsp', 'human_data')
        keypoints2d_, keypoints2d_mask = convert_kps(keypoints2d_, 'h36m', 'human_data')
        human_data['keypoints2d_mask'] = keypoints2d_mask
        human_data['keypoints2d'] = keypoints2d_
        human_data['image_path'] = image_path_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['smpl'] = smpl
        human_data['cam_param'] = cam_param_
        human_data['config'] = 'pw3d'

        # store data
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        file_name = '3doh_{}_w_kp2d_rendered.npz'.format(mode)
        out_file = os.path.join(out_path, file_name)
        human_data.dump(out_file)