import torch
import torch.nn as nn
import numpy as np
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
import pdb


def load_off(off_file_name, to_torch=False):
    file_handle = open(off_file_name)

    file_list = file_handle.readlines()
    n_points = int(file_list[1].split(' ')[0])
    all_strings = ''.join(file_list[2 : 2 + n_points])
    array_ = np.fromstring(all_strings, dtype=np.float32, sep='\n')

    all_strings = ''.join(file_list[2 + n_points :])
    array_int = np.fromstring(all_strings, dtype=np.int32, sep='\n')

    array_ = array_.reshape((-1, 3))

    if not to_torch:
        return array_, array_int.reshape((-1, 4))[:, 1::]
    else:
        return torch.from_numpy(array_), torch.from_numpy(
            array_int.reshape((-1, 4))[:, 1::]
        )


def save_off(off_file_name, vertices, faces):
    out_string = 'OFF\n'
    out_string += '%d %d 0\n' % (vertices.shape[0], faces.shape[0])
    for v in vertices:
        out_string += '%.16f %.16f %.16f\n' % (v[0], v[1], v[2])
    for f in faces:
        out_string += '3 %d %d %d\n' % (f[0], f[1], f[2])
    with open(off_file_name, 'w') as fl:
        fl.write(out_string)
    return


def rotation_theta(theta, device_=None):
    # cos -sin  0
    # sin  cos  0
    # 0    0    1
    if type(theta) == float:
        if device_ is None:
            device_ = 'cpu'
        theta = torch.ones((1, 1, 1)).to(device_) * theta
    else:
        if device_ is None:
            device_ = theta.device
        theta = theta.view(-1, 1, 1)

    mul_ = (
        torch.Tensor([[1, 0, 0, 0, 1, 0, 0, 0, 0], [0, -1, 0, 1, 0, 0, 0, 0, 0]])
        .view(1, 2, 9)
        .to(device_)
    )
    bia_ = torch.Tensor([0] * 8 + [1]).view(1, 1, 9).to(device_)

    # [n, 1, 2]
    cos_sin = torch.cat((torch.cos(theta), torch.sin(theta)), dim=2).to(device_)

    # [n, 1, 2] @ [1, 2, 9] + [1, 1, 9] => [n, 1, 9] => [n, 3, 3]
    trans = torch.matmul(cos_sin, mul_) + bia_
    trans = trans.view(-1, 3, 3)

    return trans


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


def campos_to_R_T(campos, theta, device='cpu', at=((0, 0, 0),), up=((0, 1, 0),)):
    R = look_at_rotation(campos, at=at, device=device, up=up)  # (n, 3, 3)
    R = torch.bmm(R, rotation_theta(theta, device_=device))
    T = -torch.bmm(R.transpose(1, 2), campos.unsqueeze(2))[:, :, 0]  # (1, 3)
    return R, T


# For meshes in PASCAL3D+
def pre_process_mesh_pascal(verts):
    verts = torch.cat((verts[:, 0:1], verts[:, 2:3], -verts[:, 1:2]), dim=1)
    return verts


def softmax_feature_blend(
    features: torch.Tensor,
    fragments,
    blend_params: BlendParams,
    znear: Union[float, torch.Tensor] = 1.0,
    zfar: Union[float, torch.Tensor] = 100,
) -> torch.Tensor:
    """
    RGB and alpha channel blending to return an RGBA image based on the method
    proposed in [1]
      - **RGB** - blend the colors based on the 2D distance based probability map and
        relative z distances.
      - **A** - blend based on the 2D distance based probability map.

    Args:
        features: (N, H, W, K, D) RGB color for each of the top K faces per pixel.
        fragments: namedtuple with outputs of rasterization. We use properties
            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image.
            - dists: FloatTensor of shape (N, H, W, K) specifying
              the 2D euclidean distance from the center of each pixel
              to each of the top K overlapping faces.
            - zbuf: FloatTensor of shape (N, H, W, K) specifying
              the interpolated depth from each pixel to to each of the
              top K overlapping faces.
        blend_params: instance of BlendParams dataclass containing properties
            - sigma: float, parameter which controls the width of the sigmoid
              function used to calculate the 2D distance based probability.
              Sigma controls the sharpness of the edges of the shape.
            - gamma: float, parameter which controls the scaling of the
              exponential function used to control the opacity of the color.
            - background_color: (3) element list/tuple/torch.Tensor specifying
              the RGB values for the background color.
        znear: float, near clipping plane in the z direction
        zfar: float, far clipping plane in the z direction

    Returns:
        pixel_features with alpha channel: (N, H, W, D+1)

    [0] Shichen Liu et al, 'Soft Rasterizer: A Differentiable Renderer for
    Image-based 3D Reasoning'
    """

    D = features.shape[4]
    N, H, W, K = fragments.pix_to_face.shape
    device = fragments.pix_to_face.device
    pixel_features = torch.ones(
        (N, H, W, D + 1), dtype=features.dtype, device=features.device
    )
    if (not hasattr(blend_params, 'background_color')) or (
        blend_params.background_color.shape[0] != D
    ):
        background_ = torch.zeros(D, dtype=features.dtype, device=features.device)
    else:
        background_ = blend_params.background_color
    if not isinstance(background_, torch.Tensor):
        background = torch.tensor(background_, dtype=torch.float32, device=device)
    else:
        background = background_.to(device)

    # Weight for background color
    eps = 1e-10

    # Mask for padded pixels.
    mask = fragments.pix_to_face >= 0

    # Sigmoid probability map based on the distance of the pixel to the face.
    prob_map = torch.sigmoid(-fragments.dists / (blend_params.sigma + 1e-8)) * mask

    # The cumulative product ensures that alpha will be 0.0 if at least 1
    # face fully covers the pixel as for that face, prob will be 1.0.
    # This results in a multiplication by 0.0 because of the (1.0 - prob)
    # term. Therefore 1.0 - alpha will be 1.0.
    alpha = torch.prod((1.0 - prob_map), dim=-1)

    # Weights for each face. Adjust the exponential by the max z to prevent
    # overflow. zbuf shape (N, H, W, K), find max over K.
    # TODO: there may still be some instability in the exponent calculation.

    # Reshape to be compatible with (N, H, W, K) values in fragments
    if torch.is_tensor(zfar):
        # pyre-fixme[16]
        zfar = zfar[:, None, None, None]
    if torch.is_tensor(znear):
        znear = znear[:, None, None, None]

    z_inv = (zfar - fragments.zbuf) / (zfar - znear) * mask
    # pyre-fixme[16]: `Tuple` has no attribute `values`.
    z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=eps)
    weights_num = prob_map * torch.exp((z_inv - z_inv_max) / blend_params.gamma)

    # Also apply exp normalize trick for the background color weight.
    # Clamp to ensure delta is never 0.
    # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
    delta = torch.exp((eps - z_inv_max) / blend_params.gamma).clamp(min=eps)

    # Normalize weights.
    # weights_num shape: (N, H, W, K). Sum over K and divide through by the sum.
    denom = weights_num.sum(dim=-1)[..., None] + delta

    # Sum: weights * textures + background color
    weighted_features = (weights_num[..., None] * features).sum(dim=-2)
    weighted_background = delta * background
    pixel_features[..., :D] = (weighted_features + weighted_background) / denom
    if torch.isnan(pixel_features).any():
        import ipdb

        ipdb.set_trace()
    pixel_features[..., D] = 1.0 - alpha

    return pixel_features


# Calculate interpolated maps -> [n, c, h, w]
def forward_interpolate(
    R,
    T,
    meshes,
    rasterizer,
    blur_radius=0,
    with_normal=False,
    blend_params=None,
    vert_part=None,
    features=None,
    vert_orient_weights=None,
    vert_mask=None,
    vert_loss_weights=None,
    vert_var=None,
):
    ret = dict()
    fragments = rasterize(R, T, meshes, rasterizer, blur_radius=blur_radius)
    pix_to_face = fragments.pix_to_face
    mask = (pix_to_face != -1).to(torch.float).unsqueeze(-1)
    mask = softmax_feature_blend(mask, fragments, blend_params)[..., :-1]
    mask = mask.permute((0, 3, 1, 2))
    ret['mask'] = mask
    faces = meshes.faces_packed()  # (B*F, 3)
    ret['vert_visibility'] = get_visibility(meshes, fragments)

    if with_normal:
        vertex_normals = meshes.verts_normals_packed()  # (B*V, 3)
        faces_normals = vertex_normals[faces]
        pixel_normals = utils.interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_normals
        )
        pixel_normals = softmax_feature_blend(pixel_normals, fragments, blend_params)
        pixel_normals = pixel_normals.permute((0, 3, 1, 2))
        ret['pixel_normals'] = pixel_normals

    if vert_part is not None:
        # vert_part: [V, P]
        # Works for batch where meshes have the same number of vertex and face
        faces_part = vert_part[faces % vert_part.shape[0]]
        pixel_part = utils.interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_part
        )
        pixel_part = softmax_feature_blend(pixel_part, fragments, blend_params)[
            ..., :-1
        ]
        pixel_part = pixel_part.permute((0, 3, 1, 2))
        ret['pixel_part'] = pixel_part

    if features is not None:
        # features: [V, C] or [B, V, C]
        if len(features.shape) == 3:
            face_features = features.view(-1, features.shape[-1])[faces]  # B*F, C
        else:
            face_features = features[faces % features.shape[0]]  # B*F, C
        projected_map = utils.interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, face_features
        )
        projected_map = softmax_feature_blend(
            projected_map, fragments, blend_params, zfar=100
        )[..., :-1]
        projected_map = projected_map.permute((0, 3, 1, 2))
        ret['projected_map'] = projected_map

    if vert_orient_weights is not None:
        # vert_orient_weights: [B, V, n_orient]
        faces_orient_weights = vert_orient_weights.view(
            -1, vert_orient_weights.shape[-1]
        )[
            faces
        ]  # B, F, n_orient
        pixel_orient_weights = utils.interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_orient_weights
        )
        pixel_orient_weights = softmax_feature_blend(
            pixel_orient_weights, fragments, blend_params
        )[..., :-1]
        pixel_orient_weights = pixel_orient_weights.permute((0, 3, 1, 2))
        ret['pixel_orient_weights'] = pixel_orient_weights
    else:
        ret['pixel_orient_weights'] = torch.ones(mask.shape[0], 1, 56, 56).to(
            mask.device
        )

    if vert_var is not None:
        # vert_var: [V, n_orient]
        faces_var = vert_var.view(-1, vert_var.shape[-1])[faces]
        pixel_var = utils.interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_var
        )
        pixel_var = softmax_feature_blend(pixel_var, fragments, blend_params)[..., :-1]
        pixel_var = pixel_var.permute((0, 3, 1, 2))

        ret['pixel_var'] = pixel_var

    if vert_mask is not None:
        # vert_mask: [V, 1]
        faces_mask = vert_mask.view(-1, vert_mask.shape[-1])[faces % features.shape[0]]
        pixel_mask = utils.interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_mask
        )
        pixel_mask = softmax_feature_blend(pixel_mask, fragments, blend_params)[
            ..., :-1
        ]
        pixel_mask = pixel_mask.permute((0, 3, 1, 2))

        ret['pixel_mask'] = pixel_mask

    if vert_loss_weights is not None:
        # vert_loss_weights: [V, 1]
        faces_loss_weights = vert_loss_weights.view(-1, vert_loss_weights.shape[-1])[
            faces
        ]
        pixel_loss_weights = utils.interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_loss_weights
        )
        pixel_loss_weights = softmax_feature_blend(
            pixel_loss_weights, fragments, blend_params
        )[..., :-1]
        pixel_loss_weights = pixel_loss_weights.permute((0, 3, 1, 2))
        ret['pixel_loss_weights'] = pixel_loss_weights
    return ret


class NeuralRenderer(nn.Module):
    def __init__(
        self,
        rasterizer,
        post_process=None,
        off_set_mesh=False,
        blend_params=None,
    ):
        super().__init__()

        # Device is used during theta to R
        self.rasterizer = rasterizer
        self.post_process = post_process
        self.off_set_mesh = off_set_mesh
        self.meshes = None
        self.blend_params = blend_params

    def to(self, *args, **kwargs):
        if 'device' in kwargs.keys():
            device = kwargs['device']
        else:
            device = args[0]
        super().to(device)
        self.rasterizer.cameras = self.rasterizer.cameras.to(device)
        if self.meshes is not None:
            self.meshes = self.meshes.to(device)
        return self

    def cuda(self, device=None):
        return self.to(torch.device("cuda"))

    def forward(
        self,
        verts,
        faces,
        blur_radius=0,
        deform_verts=None,
        with_normal=False,
        vert_part=None,
        vert_mask=None,
    ):
        self.meshes = Meshes(verts=verts, faces=faces, textures=None)
        if self.off_set_mesh:
            meshes = self.meshes.offset_verts(deform_verts)
        else:
            meshes = self.meshes

        cameras = self.rasterizer.cameras
        get = forward_interpolate(
            cameras.R,
            cameras.T,
            meshes,
            rasterizer=self.rasterizer,
            blur_radius=blur_radius,
            with_normal=with_normal,
            blend_params=self.blend_params,
            vert_part=vert_part,
            vert_mask=vert_mask,
        )

        if self.post_process is not None:
            get = self.post_process(get)

        return get


class NeuralMeshModel(NeuralRenderer):
    def __init__(
        self,
        features=None,
        rasterizer=None,
        post_process=None,
        off_set_mesh=False,
        blend_params=None,
    ):
        """
        features: features for vertices, [V, D]
        """
        super(NeuralMeshModel, self).__init__(
            rasterizer, post_process, off_set_mesh, blend_params
        )
        if features is not None:
            self.set_features(features)

    def set_features(self, features):
        self.register_buffer('features', features)

    def forward(
        self,
        verts,
        faces,
        blur_radius=0,
        deform_verts=None,
        with_normal=False,
        vert_orient_weights=None,
        vert_loss_weights=None,
        vert_part=None,
        vert_var=None,
        normalize_feature=True,
        vert_mask=None,
    ):
        self.meshes = Meshes(verts=verts, faces=faces, textures=None)

        if self.off_set_mesh:
            meshes = self.meshes.offset_verts(deform_verts)
        else:
            meshes = self.meshes

        cameras = self.rasterizer.cameras
        get = forward_interpolate(
            cameras.R,
            cameras.T,
            meshes,
            rasterizer=self.rasterizer,
            blur_radius=blur_radius,
            with_normal=with_normal,
            blend_params=self.blend_params,
            vert_part=vert_part,
            features=self.features,
            vert_orient_weights=vert_orient_weights,
            vert_loss_weights=vert_loss_weights,
            vert_var=vert_var,
            vert_mask=vert_mask,
        )
        feat = get['projected_map']
        if normalize_feature:
            feat = torch.nn.functional.normalize(feat + 1e-8, dim=1, p=2)
        get['projected_map'] = feat

        if self.post_process is not None:
            get = self.post_process(get)

        return get


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
        #  in_ndc=False
    )
    return cameras


def get_blend_params(sigma=0, gamma=1e-2):
    blend_params = BlendParams(
        sigma=sigma, gamma=gamma, background_color=torch.zeros(3)
    )
    return blend_params


def build_neural_renderer(hparams):
    sigma = hparams.SIGMA
    gamma = hparams.GAMMA
    faces_per_pixel = hparams.FACES_PER_PIXEL
    blur_radius = np.log(1.0 / 1e-4 - 1.0) * sigma

    render_image_size = hparams.RENDER_RES

    blend_params = BlendParams(
        sigma=sigma, gamma=gamma, background_color=torch.zeros(3)
    )
    raster_settings = RasterizationSettings(
        image_size=render_image_size,
        blur_radius=blur_radius,
        faces_per_pixel=faces_per_pixel,
        bin_size=None,
        perspective_correct=False,
    )

    cameras = get_cameras(5000.0, 224, torch.zeros((4, 3)))

    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    neural_renderer = NeuralRenderer(
        rasterizer,
        post_process=None,
        blend_params=blend_params,
    )
    return neural_renderer


def build_neural_mesh_model(hparams):
    sigma = hparams.SIGMA
    gamma = hparams.GAMMA
    faces_per_pixel = hparams.FACES_PER_PIXEL
    blur_radius = np.log(1.0 / 1e-4 - 1.0) * sigma

    render_image_size = hparams.RENDER_RES

    blend_params = BlendParams(
        sigma=sigma, gamma=gamma, background_color=torch.zeros(3)
    )
    raster_settings = RasterizationSettings(
        image_size=render_image_size,
        blur_radius=blur_radius,
        faces_per_pixel=faces_per_pixel,
        bin_size=None,
        perspective_correct=False,
    )

    cameras = get_cameras(5000.0, 224, torch.zeros((4, 3)))

    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    neural_mesh_model = NeuralMeshModel(
        None,
        rasterizer=rasterizer,
        post_process=None,
        blend_params=blend_params,
    )
    return neural_mesh_model
