from glob import glob
from tqdm import tqdm
import os
import os.path as osp
import random
import numpy as np
import cv2
from scipy.io import loadmat
from PIL import Image
import json
import torch.nn.functional as F
import pickle as pkl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mmhuman3d.utils.geometry import (
    batch_rodrigues,
    estimate_translation,
    project_points,
    rotation_matrix_to_angle_axis,
    convert_weak_perspective_to_perspective,
    convert_perspective_to_weak_perspective
)
from mmhuman3d.utils.neural_renderer import (
    get_cameras)
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from torchvision.utils import make_grid
import trimesh
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation
import ipdb
from pytorch3d.structures import Meshes
# Util function for loading meshes
import pytorch3d.transforms
from pytorch3d.transforms.rotation_conversions import matrix_to_axis_angle, rotation_6d_to_matrix
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments
from pytorch3d.renderer.mesh.utils import interpolate_face_attributes
# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PerspectiveCameras,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    HardFlatShader,
    HardPhongShader,
    TexturesUV,
    TexturesVertex
)

def get_extras(data_root):
    mesh_downsample_ratio = 8 
    mesh_downsample_method = 'uniform' 
    mesh_downsample_info_postfix = '2021-04-05'
    mesh_sample_data = np.load(
                        os.path.expanduser(
                            os.path.join(
                                data_root, 'sample_params', mesh_downsample_method,
                                'sample_data_{}-{}.npz'.format(
                                    mesh_downsample_ratio,
                                    mesh_downsample_info_postfix))))
    mesh_sample_data = dict(mesh_sample_data)
    extras = {}
    extras['faces_downsampled'] = mesh_sample_data['faces_downsampled']
    extras['ds_fun'] = lambda x: x[mesh_sample_data['indices']]
    return extras

class SMPLVisualizerQ:
    def __init__(self, smpl_model, device, image_size, light_dir=((0, 1, 0),), pred_cam=None, neural_renderer=None):
        self.smpl_model = smpl_model.to(device)
        self.device = device
        cameras = PerspectiveCameras(device=device,
                                 R=neural_renderer.cameras.R[0].unsqueeze(0),
                                 T=pred_cam.unsqueeze(0), #neural_renderer.cameras.T[0].unsqueeze(0)/100000,
                                 focal_length=neural_renderer.cameras.focal_length[0].unsqueeze(0),
                                 principal_point=neural_renderer.cameras.principal_point[0].unsqueeze(0),
                                 image_size=neural_renderer.cameras.image_size[0].unsqueeze(0),
                                 in_ndc=False)
        self.cameras = cameras

        # create renderer
        self.raster_settings = RasterizationSettings(image_size=image_size,
                                            blur_radius=0.0,
                                            faces_per_pixel=1,
                                            bin_size=None,
                                            perspective_correct=False)
        self.lights = DirectionalLights(device=device, direction=light_dir)
        
        # create face
        extras = get_extras('./data')
        faces_down = extras['faces_downsampled']       
        valid = []
        for i, face in enumerate(faces_down):
            if len(set(face)) == 3:
                valid.append(i)
        faces_down = faces_down[valid]
        self.faces_down = torch.from_numpy(faces_down).to(device)
        
    def visualize_gt(self, background=None, saveSMPL2d=False, saveSMPL3d=False):
        with open("/home/pengliang/human3d/exp/neuralsmpl_test_stage1_w_pare/gt_pose_beta.json",'r',encoding='UTF-8') as f:
            gt_betas = torch.Tensor(json.loads(f.read())['betas']).to(self.device)
        with open("/home/pengliang/human3d/exp/neuralsmpl_test_stage1_w_pare/gt_pose_beta.json",'r',encoding='UTF-8') as f:
            gt_pose = torch.Tensor(json.loads(f.read())['body_pose']).to(self.device)
        with open("/home/pengliang/human3d/exp/neuralsmpl_test_stage1_w_pare/gt_pose_beta.json",'r',encoding='UTF-8') as f:
            gt_global_orient = torch.Tensor(json.loads(f.read())['global_orient']).to(self.device)
        with open("/home/pengliang/human3d/exp/neuralsmpl_test_stage1_w_pare/gt_kp2d.json",'r',encoding='UTF-8') as f:
            gt_kp2d = torch.Tensor(json.loads(f.read())['keypoints2d']).to(self.device)
        gt_output = self.smpl_model(
            betas=gt_betas[:1,:],
            body_pose=gt_pose[:1,:],
            global_orient=gt_global_orient[:1,:].unsqueeze(1),
            num_joints= 49,
        )
        gt_cam_t = estimate_translation(
            gt_output['joints'], 
            gt_kp2d[0].unsqueeze(0),
            focal_length=5000.0,
            img_size=224,
            use_all_joints=True,
        )/1e5
        gt_cam_t[0][2] = gt_cam_t[0][2] * 1e5 + 40
        new_cam = get_cameras(
            5000.0, 
            224, 
            gt_cam_t)
        self.renderer = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(
                cameras=new_cam, 
                raster_settings=self.raster_settings
            ),
            shader=HardPhongShader(
                device=self.device, 
                cameras=new_cam,
                lights=self.lights
            )
        )
        verts_full = gt_output['vertices'].squeeze(0)
        faces = self.smpl_model.faces_tensor
        verts_rgb = torch.ones_like(verts_full)[None] 
        textures = TexturesVertex(verts_features=verts_rgb.to(self.device))
        gt_mesh = Meshes(verts=[verts_full.to(self.device)], faces=[faces.to(self.device)], textures=textures)
        with torch.no_grad():
            images, fragments = self.renderer(gt_mesh)
        pix_to_face = fragments.pix_to_face
        mask = (pix_to_face != -1).to(torch.float)

        background = background[0]
        background -= background.min()
        background /= background.max()
        _background = torch.cat((background.permute(1,2,0), images[0,:,:,3].unsqueeze(2)*255),dim=2)
        images = _background * (1-mask) + images[:,:,:,:] * mask

        img = images[0, ..., :].cpu().detach().numpy()
        if saveSMPL2d:
            np.save('./exp/vis_output/rendererSMPL.npy', img, allow_pickle=True)
            np.save('./exp/vis_output/rendererSMPLbackground.npy', _background.cpu().detach().numpy(), allow_pickle=True)
        if saveSMPL3d:
            mesh_trimesh = trimesh.Trimesh(vertices=verts_full.detach().cpu().numpy(), faces=faces.detach().cpu().numpy(), texture=textures.verts_features_packed().detach().cpu().numpy())
            mesh_trimesh.export('./exp/vis_output/SMPL.ply')
        ipdb.set_trace()

    def visualize_ours(self, saveSMPL2d=False, saveSMPL3d=False):
        with open("/home/pengliang/human3d/exp/neuralsmpl_test_stage1_w_pare/gt_pose_beta.json",'r',encoding='UTF-8') as f:
            gt_betas = torch.Tensor(json.loads(f.read())['betas']).to(self.device)
        with open("/home/pengliang/human3d/exp/neuralsmpl_test_stage1_w_pare/gt_pose_beta.json",'r',encoding='UTF-8') as f:
            gt_pose = torch.Tensor(json.loads(f.read())['body_pose']).to(self.device)
        with open("/home/pengliang/human3d/exp/neuralsmpl_test_stage1_w_pare/gt_pose_beta.json",'r',encoding='UTF-8') as f:
            gt_global_orient = torch.Tensor(json.loads(f.read())['global_orient']).to(self.device)
        with open("/home/pengliang/human3d/exp/neuralsmpl_test_stage1_w_pare/gt_kp2d.json",'r',encoding='UTF-8') as f:
            gt_kp2d = torch.Tensor(json.loads(f.read())['keypoints2d']).to(self.device)
        gt_output = self.smpl_model(
            betas=gt_betas[:1,:],
            body_pose=gt_pose[:1,:],
            global_orient=gt_global_orient[:1,:].unsqueeze(1),
            num_joints= 49,
        )
        gt_cam_t = estimate_translation(
            gt_output['joints'], 
            gt_kp2d[0].unsqueeze(0),
            focal_length=5000.0,
            img_size=224,
            use_all_joints=True,
        )/1e5
        gt_cam_t[0][2] = gt_cam_t[0][2] * 1e5 + 40
        new_cam = get_cameras(
            5000.0, 
            224, 
            gt_cam_t)
        self.renderer = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(
                cameras=new_cam, 
                raster_settings=self.raster_settings
            ),
            shader=HardPhongShader(
                device=self.device, 
                cameras=new_cam,
                lights=self.lights
            )
        )
        verts_full = gt_output['vertices'].squeeze(0)
        faces = self.smpl_model.faces_tensor
        verts_rgb = torch.ones_like(verts_full)[None] 
        textures = TexturesVertex(verts_features=verts_rgb.to(self.device))
        gt_mesh = Meshes(verts=[verts_full.to(self.device)], faces=[faces.to(self.device)], textures=textures)
        with torch.no_grad():
            images, fragments = self.renderer(gt_mesh)
        pix_to_face = fragments.pix_to_face
        mask = (pix_to_face != -1).to(torch.float)

        background = background[0]
        background -= background.min()
        background /= background.max()
        _background = torch.cat((background.permute(1,2,0), images[0,:,:,3].unsqueeze(2)*255),dim=2)
        images = _background * (1-mask) + images[:,:,:,:] * mask

        img = images[0, ..., :].cpu().detach().numpy()
        if saveSMPL2d:
            np.save('./exp/vis_output/rendererSMPL.npy', img, allow_pickle=True)
            np.save('./exp/vis_output/rendererSMPLbackground.npy', _background.cpu().detach().numpy(), allow_pickle=True)
        if saveSMPL3d:
            mesh_trimesh = trimesh.Trimesh(vertices=verts_full.detach().cpu().numpy(), faces=faces.detach().cpu().numpy(), texture=textures.verts_features_packed().detach().cpu().numpy())
            mesh_trimesh.export('./exp/vis_output/SMPL.ply')
        ipdb.set_trace()

    def visualize_nerualSMPL(self, saveSMPL2d=False, saveSMPL3d=False):
        ipdb.set_trace()
        with open("/home/pengliang/human3d/exp/neuralsmpl_test_stage1_w_pare/gt_pose_beta.json",'r',encoding='UTF-8') as f:
            gt_betas = torch.Tensor(json.loads(f.read())['betas']).to(self.device)
        with open("/home/pengliang/human3d/exp/neuralsmpl_test_stage1_w_pare/gt_pose_beta.json",'r',encoding='UTF-8') as f:
            gt_pose = torch.Tensor(json.loads(f.read())['body_pose']).to(self.device)
        with open("/home/pengliang/human3d/exp/neuralsmpl_test_stage1_w_pare/gt_pose_beta.json",'r',encoding='UTF-8') as f:
            gt_global_orient = torch.Tensor(json.loads(f.read())['global_orient']).to(self.device)
        verts_full = self.smpl_model(
            betas=gt_betas[:1,:],
            body_pose=gt_pose[:1,:],
            global_orient=gt_global_orient[:1,:].unsqueeze(1),
            num_joints= 49,
        )['vertices'].squeeze(0)
        faces = self.smpl_model.faces_tensor
        verts_rgb = torch.ones_like(verts_full)[None] 
        textures = TexturesVertex(verts_features=verts_rgb.to(self.device))
        gt_mesh = Meshes(verts=[verts_full.to(self.device)], faces=[faces.to(self.device)], textures=textures)
        with torch.no_grad():
            images, fragments = self.renderer(gt_mesh)
        pix_to_face = fragments.pix_to_face
        mask = (pix_to_face != -1).to(torch.float)
        if background is not None:
            background -= background.min()
            background /= background.max()
            _background = torch.cat((background.permute(1,2,0), images[0,:,:,3].unsqueeze(2)*255),dim=2)
            images = _background * (1-mask) + images[:,:,:,:] * mask
        img = images[0, ..., :].cpu().detach().numpy()
        if saveSMPL2d:
            np.save('./exp/vis_output/rendererSMPL.npy', img, allow_pickle=True)
            np.save('./exp/vis_output/rendererSMPLbackground.npy', _background.cpu().detach().numpy(), allow_pickle=True)
        if saveSMPL3d:
            mesh_trimesh = trimesh.Trimesh(vertices=verts_full.detach().cpu().numpy(), faces=faces.detach().cpu().numpy(), texture=textures.verts_features_packed().detach().cpu().numpy())
            mesh_trimesh.export('SMPL.ply')