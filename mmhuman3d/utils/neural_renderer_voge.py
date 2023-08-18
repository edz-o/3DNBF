import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import ipdb
from pytorch3d.renderer.mesh.rasterizer import Fragments
import pytorch3d.renderer.mesh.utils as utils
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras,
    look_at_view_transform,
    look_at_rotation,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    BlendParams,
    camera_position_from_spherical_angles,
    HardPhongShader,
    PointLights,
    PerspectiveCameras,
)
import sys
from VoGE.Meshes import GaussianMeshes, GaussianMeshesNaive
from VoGE.Renderer import (
    GaussianRenderer,
    GaussianRenderSettings,
    to_white_background,
    interpolate_attr,
    get_silhouette,
)

try:
    from pytorch3d.structures import Meshes, Textures

    use_textures = True
except:
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import TexturesVertex
    from pytorch3d.renderer import TexturesVertex as Textures

    use_textures = False
from typing import NamedTuple, Sequence, Union
from mmhuman3d.utils.image_utils import get_visibility


def rasterize(R, T, meshes, rasterizer, blur_radius=0):
    # It will automatically update the camera settings -> R, T in rasterizer.camera
    fragments = rasterizer(meshes, R=R, T=T)

    # Copy from pytorch3D source code, try if this is necessary to do gradient decent
    if blur_radius > 0.0:
        clipped_bary_coords = utils._clip_barycentric_coordinates(fragments.bary_coords)
        clipped_zbuf = utils._interpolate_zbuf(
            fragments.pix_to_face, clipped_bary_coords, meshes
        )
        fragments = Fragments(
            bary_coords=clipped_bary_coords,
            zbuf=clipped_zbuf,
            dists=fragments.dists,
            pix_to_face=fragments.pix_to_face,
        )
    return fragments


def get_cameras(focal_length, img_size, T, device=None):
    if device is None:
        device = T.device
    bs = T.shape[0]
    R = torch.eye(3).expand(bs, -1, -1).to(device)  # [None, :, :]
    Rx = torch.diag(torch.tensor([-1, -1, 1])).expand(bs, -1, -1).type_as(R)
    R = torch.bmm(Rx, R)
    T = torch.bmm(Rx, T.unsqueeze(2)).squeeze(2)
    f = torch.tensor([[focal_length, focal_length]]).expand(bs, -1)
    c = torch.tensor([[img_size / 2, img_size / 2]]).expand(bs, -1)
    R, T, f, c = [val.float().to(device) for val in [R, T, f, c]]
    cameras = PerspectiveCameras(
        device=device,
        R=R.transpose(1, 2),
        T=T,
        focal_length=f,
        principal_point=c,
        image_size=((img_size, img_size),),
        #  in_ndc=False # pytorch3d>=0.5.0
    )
    return cameras


# Calculate interpolated maps -> [n, c, h, w]
# memory.shape: [n_vert, 3, c]
def forward_interpolate(gmesh, features, rasterizer):
    fragments = rasterizer(gmesh)
    out_map = interpolate_attr(fragments, features)

    # N, H, W, C
    return out_map


class NeuralMeshModelVoGE(nn.Module):
    def __init__(
        self,
        features=None,
        rasterizer=None,
        post_process=None,
        off_set_mesh=False,
    ):
        """
        features: features for vertices, [V, D]
        """
        super(NeuralMeshModelVoGE, self).__init__()
        self.rasterizer = rasterizer
        self.post_process = post_process
        self.off_set_mesh = off_set_mesh
        if features is not None:
            self.set_features(features)
        self.render_size = self.rasterizer.render_settings.image_size

    def set_features(self, features):
        self.register_buffer('features', features)

    def forward(
        self, verts, faces, vert_orient_weights=None, vert_part=None, deform_verts=None
    ):
        sigma = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]
        scale_base = 2000
        sigmas = (
            torch.Tensor(sigma)
            .unsqueeze(0)
            .repeat(verts.shape[0], verts.shape[1], 1, 1)
            .float()
            .to(verts.device)
            * scale_base
        )
        self.gmesh = GaussianMeshesNaive(verts, sigmas)

        if self.off_set_mesh:
            gmesh = self.gmesh.offset_verts(deform_verts)
        else:
            gmesh = self.gmesh

        fragments = self.rasterizer(gmesh)

        # mask: N, H, W
        pix_to_kernel = fragments.vert_index
        mask = (pix_to_kernel != -1).to(torch.float)[:, :, :, 0]
        mask = F.interpolate(mask.unsqueeze(0), self.render_size).squeeze(0)
        mask_bp = get_silhouette(fragments)
        # vert_visibility: N, P
        N = verts.shape[0]
        P = verts.shape[1]
        visibility_map = torch.zeros(N, P + 1).long()
        pix_to_kernel = pix_to_kernel[:, :, :, 0].view(N, -1)
        for i in range(N):
            visibility_map[i][
                (torch.clamp(torch.unique(pix_to_kernel[i]) - i * P, min=-1) + 1).long()
            ] = 1
        vert_visibility = visibility_map[:, 1:].long().to(verts.device)

        # pixel_part: N, C, H, W ← P, C (bs, 25, 224, 224)
        pixel_part = interpolate_attr(fragments, vert_part).permute(0, 3, 1, 2)
        pixel_part = F.interpolate(pixel_part, self.render_size)

        # pixel_orient_weights: N, C, H, W ← N, P, C (bs, 3, 224, 224)
        pixel_orient_weights = interpolate_attr(fragments, vert_orient_weights).permute(
            0, 3, 1, 2
        )
        pixel_orient_weights = F.interpolate(pixel_orient_weights, self.render_size)
        # (BUG?) ADD NORMALIZATION
        pixel_orient_weights = torch.softmax(pixel_orient_weights, dim=1)
        # projected_map: N, C, H, W  ← P, C (bs, 384, 224, 224) for computing loss with coke_features(prediccted_map)
        projected_map = interpolate_attr(fragments, self.features).permute(0, 3, 1, 2)
        projected_map = F.interpolate(projected_map, self.render_size)
        projected_map = torch.nn.functional.normalize(projected_map, dim=1, p=2)

        ras_out = dict(
            mask=mask,
            mask_bp=mask_bp,
            pixel_part=pixel_part,
            projected_map=projected_map,
            vert_visibility=vert_visibility,
            pixel_orient_weights=pixel_orient_weights,
        )

        return ras_out

    # def detect_parts(self, feature_map=None):


def build_neural_renderer_voge(hparams):
    render_image_size = hparams.RENDER_RES
    # render_image_size = 224

    render_settings = GaussianRenderSettings(
        image_size=(render_image_size, render_image_size),
        max_assign=hparams.FACES_PER_PIXEL,  # ellipsoid kernel number
        principal=(render_image_size / 2, render_image_size / 2),
        max_point_per_bin=-1,
        thr_activation=0,
    )

    # temporary camera
    camera = get_cameras(5000.0, render_image_size, torch.zeros((4, 3)))

    neural_renderer = GaussianRenderer(cameras=camera, render_settings=render_settings)

    return neural_renderer


def build_neural_mesh_model_voge(hparams):
    render_image_size = hparams.RENDER_RES

    render_settings = GaussianRenderSettings(
        image_size=(render_image_size, render_image_size),
        max_assign=hparams.FACES_PER_PIXEL,  # ellipsoid kernel number
        principal=(render_image_size / 2, render_image_size / 2),
        max_point_per_bin=-1,
        thr_activation=0,
    )

    # temporary camera
    camera = get_cameras(5000.0, 224, torch.zeros((4, 3)).to('cuda'))

    neural_renderer = GaussianRenderer(cameras=camera, render_settings=render_settings)

    neural_mesh_model_voge = NeuralMeshModelVoGE(
        features=None,
        rasterizer=neural_renderer,
        post_process=None,
    )

    return neural_mesh_model_voge
