import os
import pickle
from typing import List
from yacs.config import CfgNode as CN

import cv2
import numpy as np
from tqdm import tqdm

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

occ_seq_list = [
    'courtyard_backpack', 
    'courtyard_basketball', 
    'courtyard_bodyScannerMotions', 
    'courtyard_box', 
    'courtyard_golf', 
    'courtyard_jacket', 
    'courtyard_laceShoe', 
    'downtown_stairs', 
    'flat_guitar', 
    'flat_packBags', 
    'outdoors_climbing', 
    'outdoors_crosscountry', 
    'outdoors_fencing', 
    'outdoors_freestyle', 
    'outdoors_golf', 
    'outdoors_parcours', 
    'outdoors_slalom', 
]

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
                                image_size=(image_size,),
                                # in_ndc=False
                                ) 

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
class Pw3dOccConverter(BaseModeConverter):
    """3D Poses in the Wild dataset `Recovering Accurate 3D Human Pose in The
    Wild Using IMUs and a Moving Camera' ECCV'2018 More details can be found in
    the `paper.

    <https://virtualhumans.mpi-inf.mpg.de/papers/vonmarcardECCV18/
    vonmarcardECCV18.pdf>`__ .

    Args:
        modes (list): 'test' and/or 'train' for accepted modes
    """

    ACCEPTED_MODES = ['train', 'test']
    
    def __init__(self, modes: List = [], device='cpu') -> None:
        super(Pw3dOccConverter, self).__init__(modes)
        self.device = device
        self.body_model = build_body_model(dict(
            type='GenderedSMPL',
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
                A dict containing keys image_path, bbox_xywh, smpl, meta
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
        meta = {}
        meta['gender'] = []

        root_path = dataset_path
        # get a list of .pkl files in the directory
        files = []
        for mode in ['train', 'validation', 'test']:
            dataset_path = os.path.join(root_path, 'sequenceFiles', mode)
            for f in os.listdir(dataset_path):
                if f.endswith('.pkl') and f:
                    seq_name = '_'.join(f.split('_')[:-1])
                    if seq_name in occ_seq_list:
                        files.append(os.path.join(dataset_path, f))

        files = sorted(files) # New
        # go through all the .pkl files
        for filename in tqdm(files):
            # if not 'downtown_enterShop_00' in filename:
            #     continue
            # import ipdb; ipdb.set_trace()
            with open(filename, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
                smpl_pose = data['poses']
                smpl_betas = data['betas']
                smpl_trans = data['trans']
                poses2d = data['poses2d']
                global_poses = data['cam_poses']
                genders = data['genders']
                valid = np.array(data['campose_valid']).astype(np.bool)
                K = np.array(data['cam_intrinsics'])
                num_people = len(smpl_pose)
                num_frames = len(smpl_pose[0])
                seq_name = str(data['sequence'])
                img_names = np.array([
                    'imageFiles/' + seq_name + f'/image_{str(i).zfill(5)}.jpg'
                    for i in range(num_frames)
                ])
                # get through all the people in the sequence
                for i in range(num_people):
                    valid_pose = smpl_pose[i][valid[i]]
                    valid_betas = np.tile(smpl_betas[i][:10].reshape(1, -1),
                                          (num_frames, 1))
                    valid_betas = valid_betas[valid[i]]
                    valid_trans = smpl_trans[i][valid[i]]
                    valid_keypoints_2d = poses2d[i][valid[i]]
                    valid_img_names = img_names[valid[i]]
                    valid_global_poses = global_poses[valid[i]]
                    gender = genders[i]

                    # consider only valid frames
                    for valid_i in range(valid_pose.shape[0]):
                        keypoints2d = valid_keypoints_2d[valid_i, :, :].T
                        keypoints2d = keypoints2d[keypoints2d[:, 2] > 0, :]
                        bbox_xyxy = [
                            min(keypoints2d[:, 0]),
                            min(keypoints2d[:, 1]),
                            max(keypoints2d[:, 0]),
                            max(keypoints2d[:, 1])
                        ]

                        bbox_xyxy = self._bbox_expand(
                            bbox_xyxy, scale_factor=1.2)
                        bbox_xywh = self._xyxy2xywh(bbox_xyxy)

                        image_path = valid_img_names[valid_i]
                        image_abs_path = os.path.join(root_path, image_path)
                        h, w, _ = cv2.imread(image_abs_path).shape

                        # transform global pose
                        pose = valid_pose[valid_i]
                        extrinsic_param = valid_global_poses[valid_i]
                        R = extrinsic_param[:3, :3]
                        T = extrinsic_param[:3, 3]
                        
                        camera = CameraParameter(H=h, W=w)
                        camera.set_KRT(K, R, T)
                        parameter_dict = camera.to_dict()
                        
                        # Compute keypoints2d
                        betas = valid_betas[valid_i]
                        trans = valid_trans[valid_i]
                        if gender == 'm':
                            gender_ = 0
                        else:
                            gender_ = 1
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
                                    gender=torch.tensor([gender_], device=self.device))
                            verts = gt_output['vertices'] + trans
                            gt_keypoints3d = gt_output['joints'] + trans
                            gt_keypoints2d = cameras.transform_points_screen(gt_keypoints3d, image_size=((w, h), ))[:, :, :2].squeeze(0)
                            
                            keypoints2d = np.ones([gt_keypoints2d.shape[0], 3])
                            keypoints2d[:, :2] = gt_keypoints2d.cpu().numpy()

                            # # Compute visibility
                            # if gender_ == 0:
                            #     faces = self.body_model.smpl_male.faces_tensor.to(self.device)
                            # else:
                            #     faces = self.body_model.smpl_female.faces_tensor.to(self.device)
                            
                            # # verts_rgb = torch.ones_like(verts)  # (1, V, 3)
                            # # textures = TexturesVertex(verts_features=verts_rgb.to(self.device))
                            # # mesh = Meshes(verts=[verts[0].to(self.device)],
                            # #             faces=[faces],
                            # #             textures=textures)
                            
                            # mask, iskpvisible = get_mask_and_visibility(vertices=verts,
                            #             faces=faces,
                            #             rasterizer=rasterizer, )
                            # j_regressor = self.body_model.smpl_neutral.joints_regressor
                            # # TODO use the neighboring vertices of (j_regressor>0), currently too local
                            # kp_visibility = (((j_regressor>0).float() @ iskpvisible[0].float().unsqueeze(-1)).squeeze(-1)>0).cpu().numpy()
                            # keypoints2d[:, 2] = keypoints2d[:, 2] * kp_visibility

                            keypoints2d_.append(keypoints2d)

                        # import ipdb; ipdb.set_trace()
                        R = tensor2array(R) # recover R
                        pose = tensor2array(pose)

                        pose[:3] = cv2.Rodrigues(
                            np.dot(R,
                                   cv2.Rodrigues(pose[:3])[0]))[0].T[0]

                        image_path_.append(image_path)
                        bbox_xywh_.append(bbox_xywh)
                        smpl['body_pose'].append(pose[3:].reshape((23, 3)))
                        smpl['global_orient'].append(pose[:3])
                        smpl['betas'].append(valid_betas[valid_i])
                        meta['gender'].append(gender)
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
        meta['gender'] = np.array(meta['gender'])

        keypoints2d_ = np.array(keypoints2d_)
        keypoints2d_, keypoints2d_mask = convert_kps(keypoints2d_, 'h36m', 'human_data')
        human_data['keypoints2d_mask'] = keypoints2d_mask
        human_data['keypoints2d'] = keypoints2d_
        human_data['image_path'] = image_path_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['smpl'] = smpl
        human_data['meta'] = meta
        human_data['cam_param'] = cam_param_
        human_data['config'] = 'pw3d'

        # store data
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        file_name = 'pw3d_occ_w_kp2d_correct.npz'
        out_file = os.path.join(out_path, file_name)
        human_data.dump(out_file)
