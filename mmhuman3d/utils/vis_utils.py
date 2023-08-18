import os
import os.path as osp
import numpy as np
import cv2
from PIL import Image
import json
import torch.nn.functional as F
import pickle as pkl
import matplotlib.pyplot as plt
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import trimesh
import torch
import torch.nn as nn
from pytorch3d.structures import Meshes
# Util function for loading meshes
import pytorch3d.transforms
from pytorch3d.transforms import (
    axis_angle_to_matrix)
from pytorch3d.transforms.rotation_conversions import matrix_to_axis_angle, rotation_6d_to_matrix
from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments
# Data structures and functions for rendering
from pytorch3d.structures import Meshes
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

from mmhuman3d.utils.neural_renderer import get_cameras
from mmhuman3d.utils.geometry import convert_weak_perspective_to_perspective
from mmhuman3d.models.losses.likelihood_loss import soft_loss_fun

from VoGE.Renderer import to_white_background, get_silhouette, interpolate_attr, GaussianRenderSettings



rotation_6d_to_axis_angle = lambda x: matrix_to_axis_angle(
                                            pytorch3d.transforms.rotation_6d_to_matrix(x))
axis_angle_to_rotation_6d = lambda x: pytorch3d.transforms.matrix_to_rotation_6d(
                                            pytorch3d.transforms.axis_angle_to_matrix(x))

from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams

class SimpleShader(nn.Module):
    def __init__(self, device="cpu", blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, blend_params)
        return images  # (N, H, W, 3) RGBA image
    
def get_post_fun(map_shape):
    def post_fun(images):
        # [N, C, H, W]
        if isinstance(images, tuple):
            output = []
            for ims in images:
                if ims is None:
                    output.append(None)
                    continue
                ims = torch.nn.functional.interpolate(ims, map_shape)
                output.append(ims)
            return output
        else:
            images = torch.nn.functional.interpolate(images, map_shape)
            return images
    return post_fun


def convert_RT_pth3d_to_screen(R, T):
    R = R.transpose(1,2)
    Rx = torch.diag(torch.tensor([-1, -1, 1])).unsqueeze(0).repeat(R.shape[0], 1, 1).type_as(R)
    R = torch.bmm(Rx, R)
    T = torch.bmm(Rx, T.unsqueeze(2)).squeeze(2)
    return R, T

def rotate_vertices_aroundy(vertices, camera_rotation, angle=90., reverse=False):
    """
    (TODO) Support batching
    Args:
        vertices: Nx3 np.ndarray
        camera_rotation: 3x3
    Returns: 
        rot_vertices: Nx3
    """
    if reverse:
        angle *= -1
    aroundy = cv2.Rodrigues(np.array([0, np.radians(angle), 0]))[0]
    vertices = np.dot(vertices, camera_rotation.T)
    center = vertices.mean(axis=0)
    rot_vertices = np.dot((vertices - center), aroundy) + center
    rot_vertices = np.dot(rot_vertices, camera_rotation)
    return rot_vertices

def rotate_vertices_aroundy_torch(vertices: torch.Tensor, 
                            camera_rotation: torch.Tensor, 
                            angle=90., reverse=False):
    """
    Args:
        vertices (torch.Tensor): BxNx3 
        camera_rotation (torch.Tensor): Bx3x3
    Returns: 
        rot_vertices: BxNx3
    """
    if reverse:
        angle *= -1
    aroundy = axis_angle_to_matrix(
                torch.tensor(
                np.array([[0, np.radians(angle), 0]])
                ).type_as(camera_rotation)).repeat(vertices.shape[0], 1, 1) # 1x3x3
    vertices = torch.bmm(vertices, camera_rotation.transpose(1,2))
    center = vertices.mean(dim=1, keepdim=True)
    rot_vertices = torch.bmm((vertices - center), aroundy) + center
    rot_vertices = torch.bmm(rot_vertices, camera_rotation)
    return rot_vertices

def rotate_vertices_aroundx(vertices, camera_rotation, angle=90., reverse=False):
    if reverse:
        angle *= -1
    aroundy = cv2.Rodrigues(np.array([np.radians(angle), 0, 0]))[0]
    vertices = np.dot(vertices, camera_rotation.T)
    center = vertices.mean(axis=0)
    rot_vertices = np.dot((vertices - center), aroundy) + center
    rot_vertices = np.dot(rot_vertices, camera_rotation)
    return rot_vertices

def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        #pdb.set_trace()
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1,0], kpt_3d[i2,0]])
        y = np.array([kpt_3d[i1,1], kpt_3d[i2,1]])
        z = np.array([kpt_3d[i1,2], kpt_3d[i2,2]])
        
        if kpt_3d_vis[i1,0] > 0 and kpt_3d_vis[i2,0] > 0:
            ax.plot(x, z, -y, c=colors[l], linewidth=2)
        if kpt_3d_vis[i1,0] > 0:
            ax.scatter(kpt_3d[i1,0], kpt_3d[i1,2], -kpt_3d[i1,1], c=colors[l].reshape(1, -1), marker='o')
        if kpt_3d_vis[i2,0] > 0:
            ax.scatter(kpt_3d[i2,0], kpt_3d[i2,2], -kpt_3d[i2,1], c=colors[l].reshape(1, -1), marker='o')

    # x_r = np.array([0, cfg.input_shape[1]], dtype=np.float32)
    # y_r = np.array([0, cfg.input_shape[0]], dtype=np.float32)
    # z_r = np.array([0, 1], dtype=np.float32)
    
    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    #ax.legend()

    plt.show()
    cv2.waitKey(0)

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
    extras['ds_fun'] = lambda x: x[:, mesh_sample_data['indices']]
    return extras

def get_part_seg_vis(part_segm, part_segm_argmax=None, background=None):
    cm = plt.get_cmap('rainbow')
    if part_segm_argmax is not None:
        colors = np.concatenate((np.array([[0.,0.,0.,1.]]), cm(np.linspace(0, 1, 25-1))), axis=0)  # #class hard coded
    else:
        colors = np.concatenate((np.array([[0.,0.,0.,1.]]), cm(np.linspace(0, 1, part_segm.shape[2]-1))), axis=0)
        part_segm_argmax = part_segm.argmax(axis=2)
    part_segm_vis = colors[part_segm_argmax]
    part_segm_vis[:, :, 3] = ((part_segm_argmax>0)).astype(part_segm_vis.dtype)
    part_segm_vis = part_segm_vis * 255
    if background is not None:
        img_pl = Image.fromarray(background).convert('RGBA')
        img_pl.putalpha(127)
        part_segm_vis = np.array(Image.alpha_composite(img_pl, Image.fromarray((part_segm_vis).astype(np.uint8)).resize(img_pl.size)))
    else:
        part_segm_vis = part_segm_vis[:, :, :2].astype('uint8')
    return part_segm_vis
    
class SMPLVisualizer:
    def __init__(self, smpl_layer, device, cameras=None, image_size=None, light_dir=((0, 1, 0),), point_light_location=None, shader_type='hard_phong'):
        self.device = device
        
        if cameras is None:
            R, T = look_at_view_transform(2.3, 0, 0)
            # R, T = look_at_view_transform(2.7, 0, 0)
            cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        self.cameras = cameras
        #cameras = PerspectiveCameras(device=device, R=R.transpose(1,2), T=T,
        #                             focal_length=torch.tensor([[K[0,0], K[1,1]]]).type_as(R), 
        #                             principal_point=torch.tensor([[K[0,2], K[1,2]]]).type_as(R),
        #                             image_size=[gt['image_size']])
        
        if image_size is None:
            image_size = (512, 512) 
        self.image_size = image_size

        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
        # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
        # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
        # the difference between naive and coarse-to-fine rasterization. 
        raster_settings = RasterizationSettings(
            image_size=image_size, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )

        # Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
        # -z direction. 
        if point_light_location is not None:
            # [[0.0, 0.0, 3.0]]
            lights = PointLights(device=device, location=point_light_location, )
        else:
            lights = DirectionalLights(device=device, direction=light_dir,)
        # Create a phong renderer by composing a rasterizer and a shader. The textured phong shader will 
        # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
        # apply the Phong lighting model

        if shader_type == 'simple':
            shader = SimpleShader(device=device)
        elif shader_type == 'hard_phong':
            shader = HardPhongShader(
                        device=device, 
                        cameras=cameras,
                        lights=lights
                    )
        self.renderer = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=shader,
        )
        
        self.smpl_layer = smpl_layer.to(device)
        self.faces = torch.from_numpy(self.smpl_layer.faces.astype(int)).to(device)

        self.post_fun = get_post_fun(image_size)
        extras = get_extras('./data')
        self.ds_fun = extras['ds_fun']
        faces_down = extras['faces_downsampled']       
        valid = []
        for i, face in enumerate(faces_down):
            if len(set(face)) == 3:
                valid.append(i)
        faces_down = faces_down[valid]
        
        self.faces_down = torch.from_numpy(faces_down).to(device)
        
    
    def set_cameras(self, cameras):
        self.cameras = cameras
        self.renderer.rasterizer.cameras = cameras
        self.renderer.shader.cameras = cameras

    def visualize(self, 
                    pose: torch.Tensor = None, 
                    betas: torch.Tensor = None, 
                    vert_textures: torch.Tensor = None, 
                    downsample_vert: bool = False, 
                    show: bool = False, 
                    trans: torch.Tensor = None, 
                    color: torch.Tensor = None, 
                    background: torch.Tensor = None, 
                    rotate_around_y: torch.Tensor = 0, 
                    return_mask: bool = False, 
                    return_verts: bool = False, 
                    return_tensor: bool = False,
                    png_path: torch.Tensor = None,):
        """
        Args:
            pose (torch.Tensor): Bx72
            betas (torch.Tensor): Bx10
            vert_textures (torch.Tensor): BxNx3
            downsample_vert (bool):
            show (bool):
            trans (torch.Tensor): Bx3
            color (torch.Tensor): Bx3
            background (torch.Tensor): BxHxWx3
            rotate_around_y (torch.Tensor): B
            return_mask (bool):
            return_verts (bool):
            return_tensor (bool):
            png_path (str):
        Returns:
            res (dict): 
        """
        with torch.no_grad():
            if pose is not None:
                batch_size = pose.shape[0]
                if len(pose.shape) == 4 and (pose.shape[-3:] == (getattr(self.smpl_layer, 'NUM_BODY_JOINTS', 23)+1, 3, 3)):
                    smpl_output = self.smpl_layer(betas=betas, body_pose=pose[:, 1:], global_orient=pose[:, :1], pose2rot=False)
                else:
                    smpl_output = self.smpl_layer(betas=betas, body_pose=pose[:, 3:], global_orient=pose[:, :3], pose2rot=True)
            else:
                batch_size = 1
                smpl_output = self.smpl_layer(betas=None, body_pose=None, global_orient=None, pose2rot=True)

        if isinstance(smpl_output, dict):
            openpose_joints = smpl_output['joints']
            vertices = smpl_output['vertices']
        else:
            openpose_joints = smpl_output.joints
            vertices = smpl_output.vertices
        
        if trans is None:
            trans = 0
        verts_full = vertices + trans

        if rotate_around_y != 0:
            R, T = convert_RT_pth3d_to_screen(self.cameras.R, self.cameras.T)
            verts_full = rotate_vertices_aroundy_torch(
                verts_full, R, angle=rotate_around_y, reverse=False)
        
        if downsample_vert:
            verts_ds = self.ds_fun(verts_full)

        if vert_textures is not None:
            textures = TexturesVertex(verts_features=vert_textures.to(self.device))
        else:
            verts_rgb = torch.ones_like(verts_full) * 128. / 255.
            if color:
                verts_rgb[:, :, :] = torch.tensor(color).type_as(verts_rgb)
            textures = TexturesVertex(verts_features=verts_rgb.to(self.device))

        if downsample_vert:
            faces_down_ = self.faces_down[None].expand((batch_size,) + self.faces_down.shape).to(self.device)
            mesh = Meshes(verts=verts_ds, faces=faces_down_, textures=textures)
        else:
            faces_ = self.faces.expand((batch_size,) + self.faces.shape)
            mesh = Meshes(verts=verts_full, faces=faces_, textures=textures)
        self.mesh = mesh

        with torch.no_grad():
            images, fragments = self.renderer(mesh)
        # get mask
        pix_to_face = fragments.pix_to_face
        mask = (pix_to_face != -1).to(torch.float)

        if background is not None:
            images = background / 255.0 * (1-mask) + images[:,:,:,:3] * mask
        
        if return_tensor:
            im = (images[:, ..., :3].cpu().detach() * 255).long()
        else:
            im = (images[:, ..., :3].cpu().detach().numpy() * 255).astype('uint8')
        if show:
            plt.figure(figsize=(10, 10))
            plt.imshow(im[0])
            plt.savefig(png_path)
            # plt.imshow(images[0, ..., :3].cpu().detach().numpy())

        res = dict(image=im)
        if return_verts:
            res.update(dict(verts=verts_full if not downsample_vert else verts_ds))

        if return_mask:
            res.update(dict(mask=mask))
    
        return res
            
    def save_mesh_ply(self, mesh_path):
        mesh_trimesh = trimesh.Trimesh(vertices=self.mesh.verts_packed().detach().cpu().numpy(), faces=self.mesh.faces_packed().detach().cpu().numpy(), vertex_colors=self.mesh.textures.verts_features_packed().detach().cpu().numpy())
        mesh_trimesh.export(mesh_path)
        print('mesh ply save success')

def visualize_texture_pth(visualizer, vert_textures, pose, betas, 
                            img=None, cam_wp=None, cam_t=None, focal_length=5000.,
                            downsample_vert=True, vis_res=512, device='cpu'):
    """
    Args:
        img: torch.Tensor, B, H, W, 3, [0, 255]
        pose: torch.Tensor, B, 72
        betas: torch.Tensor, B, 10
        camera: torch.Tensor, camera translation, weak perspective B, 3
        visualizer: SMPLVisualizer
    Returns:
        img: np.ndarray, B, H, W, 3, [0, 255]
        img_rot: np.ndarray, B, H, W, 3, [0, 255]
    """
    pred_pose = pose.to(device).float()
    pred_betas = betas.to(device).float()

    if cam_t is not None:
        pred_cam_t = cam_t.to(device).float()
    else:
        pred_cam = cam_wp.to(device).float()
        pred_cam_t = convert_weak_perspective_to_perspective(
                    pred_cam,
                    focal_length=focal_length,
                    img_res=vis_res,
                )
    # pred_cam_t = pred_cam # prohmr
    pred_cam = get_cameras(
        focal_length,
        vis_res,
        pred_cam_t)

    visualizer.set_cameras(pred_cam)
    if img is None:
        img = 255*torch.ones((pred_cam_t.size(0), vis_res, vis_res, 3)).to(device)
    res = visualizer.visualize(pose=pred_pose, betas=pred_betas, 
                            vert_textures=vert_textures, 
                            downsample_vert=downsample_vert,
                            background=img, return_tensor=False)
    # res_rot = visualizer.visualize(pose=pred_pose, betas=pred_betas, 
    #                         color=(0.6, 0.6, 1), background=None,
    #                         rotate_around_y=90,
    #                         return_tensor=True)
    return res['image'] #, res_rot['image']

def visualize_prediction_pth(img, pose, betas, camera, visualizer, vis_res=512, device='cpu',
                             color=(0.6, 0.6, 1.0)):
    """
    Args:
        img: torch.Tensor, B, H, W, 3, [0, 255]
        pose: torch.Tensor, B, 72
        betas: torch.Tensor, B, 10
        camera: torch.Tensor, B, 3
        visualizer: SMPLVisualizer
    Returns:
        img: np.ndarray, B, H, W, 3, [0, 255]
        img_rot: np.ndarray, B, H, W, 3, [0, 255]
    """
    pred_pose = pose.to(device).float()
    pred_betas = betas.to(device).float()
    pred_cam = camera.to(device).float()

    pred_cam_t = convert_weak_perspective_to_perspective(
                pred_cam,
                focal_length=5000.,
                img_res=vis_res,
            )
    # pred_cam_t = pred_cam # prohmr
    pred_cam = get_cameras(
        5000.,
        vis_res,
        pred_cam_t)

    visualizer.set_cameras(pred_cam)
    res = visualizer.visualize(pose=pred_pose, betas=pred_betas, 
                            color=color, background=img, return_tensor=True)
    res_rot = visualizer.visualize(pose=pred_pose, betas=pred_betas, 
                            color=color, background=None,
                            rotate_around_y=90,
                            return_tensor=True)
    return res['image'], res_rot['image']

def visualize_activation(save_name, debug_info):
    imgs = debug_info['imgs']
    occ_masks = debug_info['occ_masks']
    clutter_scores = debug_info['clutter_scores']
    object_scores = debug_info['object_scores']

    total_scores = 1 - soft_loss_fun(
                torch.from_numpy(object_scores), 
                torch.from_numpy(clutter_scores), 
                reduce=None, normalize=True).numpy()

    coke_feats = torch.from_numpy(debug_info['coke_features'])
    coke_feats = F.normalize(coke_feats, p=2, dim=1)
    fig, axes = plt.subplots(len(imgs), 5, figsize=(10,3*len(imgs)))
    if len(imgs) == 1:
        axes[0].imshow(imgs[0][:,:,::-1])
        axes[1].imshow(clutter_scores[0], vmin=-1, vmax=1)
        # axes[2].imshow(object_scores[0].max(axis=0))
        axes[2].imshow(object_scores[0], vmin=-1, vmax=1)
        axes[3].imshow(occ_masks[0])
        axes[4].imshow(total_scores[0], vmin=0, vmax=1)
        axes[0].set_title('image')
        axes[1].set_title('clutter score')
        axes[2].set_title('object score')
        axes[3].set_title('occlusion mask')
        axes[4].set_title('total score')
    else:
        for i in range(len(imgs)):
            axes[i][0].imshow(imgs[i][:,:,::-1])
            axes[i][1].imshow(clutter_scores[i], vmin=-1, vmax=1)
            # axes[i][2].imshow(object_scores[i].max(axis=0))
            axes[i][2].imshow(object_scores[i], vmin=-1, vmax=1)
            axes[i][3].imshow(occ_masks[i])
            axes[i][4].imshow(total_scores[i], vmin=0, vmax=1)
            axes[i][0].set_title('image')
            axes[i][1].set_title('clutter score')
            axes[i][2].set_title('object score')
            axes[i][3].set_title('occlusion mask')
            axes[i][4].set_title('total score')
    
    os.makedirs(osp.dirname(osp.abspath(save_name)), exist_ok=True)
    fig.savefig(save_name, bbox_inches='tight')

    fig, axes = plt.subplots(2*len(imgs), 6, figsize=(10,3*len(imgs)))
    for ib in range(len(imgs)):
        axes[2*ib][0].imshow(imgs[ib][:,:,::-1])
        axes[2*ib+1][0].imshow(object_scores[ib])
        for i, thresh in enumerate(np.arange(0.0, 1, 0.1)):
            object_scores = debug_info['object_scores'].copy()
            obj_threshed = object_scores[ib]
            obj_threshed[obj_threshed<thresh] = 0
            row = i // 5
            col = i % 5 + 1
            axes[2*ib+row][col].imshow(obj_threshed)
    save_name_base, ext = osp.splitext(save_name)
    fig.savefig(save_name_base+'_fg_thresh' + ext, bbox_inches='tight')

def color_coding(xs, n_class, cmap='gist_rainbow'):
    cm = plt.get_cmap(cmap)
    colors = np.concatenate((np.array([[0.,0.,0.,1.]]), cm(np.linspace(0, 1, n_class))), axis=0) # n_class+1, 4
    return colors[xs]

@torch.no_grad()
def get_vert_detection_cc(coke_feats, fg_feats, n_vert, n_orient=3, 
                          vis_argmax_loc=False, clutter_scores=None):
    """
    Args:
        coke_feats: torch.Tensor (B, C, H, W)
        fg_feats: torch.Tensor (V, n_orient*C)
        clutter_scores: torch.Tensor (B, H, W)
    Returns:
        det_cc: np.ndarray, color coded keypoint map (B, H, W, 3)
    """
    coke_feats = F.normalize(coke_feats, p=2, dim=1)
    fg_feats = fg_feats.view(n_vert*n_orient, -1) # V*O, C
    hmap = F.conv2d(coke_feats, fg_feats[..., None, None]) # B, V*O, H, W
    hmap = hmap.view(-1, n_vert, n_orient, *hmap.shape[2:]).max(dim=2)[0] # B, V, H, W

    if vis_argmax_loc:
        # [n, k, h, w]
        w = hmap.size(3)
        hmap = hmap.view(*hmap.shape[0:2], -1)

        _, max_ = torch.max(hmap, dim=2) # max_: B, V

        pred_map = torch.zeros_like(coke_feats[:, 0, :, :]).view(coke_feats.size(0), -1).long() # B, H*W
        # pred_map[i, max_[i, j]] = np.arange(n_vert)[i, j]
        pred_map = pred_map.scatter_(1, max_, torch.arange(1, n_vert+1).to(
                    pred_map.device)[None, :].expand(coke_feats.size(0), -1))
        pred_map = pred_map.view(coke_feats.size(0), *coke_feats.shape[2:]) # B, H, W
    else:
        pred_conf = hmap.max(dim=1)[0] # B, H, W
        pred_map = hmap.argmax(dim=1)+1 # B, H, W
        if clutter_scores is not None:
            pred_map = pred_map * (pred_conf > clutter_scores) * (pred_conf > 0.8)

    vert_textures = np.concatenate((np.array([[1, 1, 1]]), np.load("data/sphere_color.npy")), axis=0)
    det_cc = (vert_textures[pred_map.cpu().numpy()] * 255).astype("uint8")

    return det_cc

def vis_vert_detection(vertices2d, out_size, n_vert):
    """
    Args:
        vertices2d: torch.Tensor (B, V, 2)
    """
    B = vertices2d.shape[0]
    
    vertices2d = vertices2d.long()
    indices = vertices2d[:, :, 1] * out_size[-1] + vertices2d[:, :, 0]
    # pred_map = torch.zeros((B,)+out_size).long() # B, H, W
    # vertices2d = vertices2d.long()
    # for i in range(B):
    #     pred_map[i, vertices2d[i, :, 1], vertices2d[i, :, 0]] = torch.arange(n_vert).to(pred_map.device)
    
    pred_map = torch.zeros((B,)+out_size).view(B, -1).long() # B, H*W
    pred_map = pred_map.scatter_(1, indices.long(), torch.arange(1, n_vert+1).to(pred_map.device)[None, :].expand(B, -1))
    pred_map = pred_map.view(B, *out_size) # B, H, W

    vert_textures = np.concatenate((np.array([[1, 1, 1]]), np.load("sphere_color.npy")), axis=0)
    det_cc = (vert_textures[pred_map.numpy()] * 255).astype("uint8")
    # det_cc = (color_coding(pred_map.numpy(), n_vert) * 255).astype("uint8")
    return det_cc

def get_fg_activations(coke_feats, fg_feats, n_vert, save_name, n_orient=3):
    """
    coke_feats: torch.Tensor (B, C, H, W)
    fg_feats: torch.Tensor (V, n_orient*C)
    """
    coke_feats = F.normalize(coke_feats, p=2, dim=1)
    fg_feats = fg_feats.view(n_vert, n_orient, -1)
    activations = []
    for v_idx in range(0, n_vert, 10):
        o_idx = 0
        feat = fg_feats[v_idx][o_idx]
        activation = F.conv2d(coke_feats, feat[None, :, None, None])
        activations.append(activation[0,0])

    ncol = 5
    nrow = int(np.ceil(len(activations) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10,3*nrow))
    for i in range(len(activations)):
        r = i // ncol
        c = i % ncol
        axes[r][c].imshow(activations[i], vmin=0.0, vmax=1)
        axes[r][c].set_title(f'{i}')

    os.makedirs(osp.dirname(osp.abspath(save_name)), exist_ok=True)
    fig.savefig(save_name, bbox_inches='tight')

def get_bg_activations(coke_feats, bg_feat, save_name):
    """
    coke_feats: torch.Tensor
    bg_feat: torch.Tensor
    """
    coke_feats = F.normalize(coke_feats, p=2, dim=1)

    max_group = 216
    num_noise = 10
    bg_activations = []
    for g_idx in range(5, max_group, 20):
        for n_idx in range(0, num_noise):
            feat = bg_feat[n_idx+g_idx*num_noise]
            bg_activation = F.conv2d(coke_feats, feat[None, :, None, None])
            bg_activations.append(bg_activation[0,0])

    ncol = 5
    nrow = int(np.ceil(len(bg_activations) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10,3*nrow))
    for i in range(len(bg_activations)):
        r = i // ncol
        c = i % ncol
        axes[r][c].imshow(bg_activations[i], vmin=0.0, vmax=1)
        axes[r][c].set_title(f'{i}')
    os.makedirs(osp.dirname(osp.abspath(save_name)), exist_ok=True)
    fig.savefig(save_name, bbox_inches='tight')


def get_bg_activations_mixture(coke_feats, bg_feats_mixture, imgs, save_name):
    """
    coke_feats: torch.Tensor
    bg_feat: torch.Tensor
    """
    coke_feats = F.normalize(coke_feats, p=2, dim=1)
    bg_feats_mixture = F.normalize(bg_feats_mixture, p=2, dim=1)

    n_mixtures = bg_feats_mixture.shape[0]
    bg_activations = []

    # B, n_mixtures, H, W
    bg_activations = F.conv2d(coke_feats, bg_feats_mixture[:, :, None, None])

    # for n in range(n_mixtures):
    #     feat = bg_feats_mixture[n]
    #     bg_activation = F.conv2d(coke_feats, feat[None, :, None, None])
    #     for bg_act_i in bg_activation:
    #         bg_activations.append(bg_act_i[0])

    ncol = 5
    nrow_cell = int(np.ceil(n_mixtures / ncol))
    nrow = nrow_cell * len(bg_activations)
    fig, axes = plt.subplots(nrow, ncol+1, figsize=(10,3*nrow))
    if len(bg_activations) == 1:
        axes = [axes]
    for j in range(len(bg_activations)):
        for i in range(nrow_cell):
            axes[j * nrow_cell+i][0].imshow(imgs[j][:,:,::-1])
        for i in range(n_mixtures):
            r = i // ncol + j * nrow_cell
            c = i % ncol
            axes[r][c+1].imshow(bg_activations[j][i], vmin=0.0, vmax=1)
            axes[r][c+1].set_title(f'{i}')
    os.makedirs(osp.dirname(osp.abspath(save_name)), exist_ok=True)
    fig.savefig(save_name, bbox_inches='tight')

