import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform

# import open3d as o3d
import joblib
import ipdb
from mmhuman3d.core.conventions import constants
from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform
from VoGE.Meshes import GaussianMeshesNaive
from VoGE.Renderer import interpolate_attr, GaussianRenderer, GaussianRenderSettings
from VoGE.Converter.Converters import naive_vertices_converter
from VoGE.Renderer import to_white_background, to_colored_background
from loguru import logger
from VoGE.Converter.IO import to_torch
from VoGE.Sampler import sample_features, sample_features_py

import sys

# sys.path.append('./smplx-master')
# from HumanPose_voge import human_pose_guassian_mesh_nr

device = 'cuda'
color_list = [
    [255, 215, 0],  # 0: hips
    [218, 165, 105],  # 1: left hip
    [227, 168, 105],  # 2: right hip
    [255, 97, 0],  # 3: spine
    [237, 145, 33],  # 4: left knee
    [255, 128, 0],  # 5: right knee
    [245, 222, 179],  # 6: spine-1
    [0, 0, 255],  # 7: left ankle
    [61, 89, 171],  # 8: right ankle
    [30, 144, 255],  # 9: spine-2
    [11, 23, 70],  # 10: left toe
    [3, 168, 158],  # 11: right toe
    [227, 23, 13],  # 12: neck
    [255, 127, 80],  # 13: left shoudler
    [178, 34, 34],  # 14: right shoulder
    [178, 48, 96],  # 15: head
    [255, 192, 203],  # 16: left arm
    [135, 38, 87],  # 17: right arm
    [255, 99, 71],  # 18: left elbow
    [255, 69, 0],  # 19: right elbow
    [138, 54, 15],  # 20: left hand
    [240, 230, 140],  # 21: right hand
    [94, 38, 18],  # 22: left thumb
    [0, 255, 127],  # 23: right thumb
]


def generate_part_labels(vertices, faces, vert_to_part, neural_renderer):
    nparts = vert_to_part.max()  # TODO: bug? nparts = vert_to_part.shape[1]-1
    batch_size = vertices.shape[0]
    # with torch.no_grad():
    ret = neural_renderer(vertices, faces, vert_part=vert_to_part)

    body_parts, mask = ret['pixel_part'], ret['mask']

    body_parts = body_parts.argmax(axis=1).detach() * mask.squeeze(1)

    body_parts_rgb = body_parts / nparts

    return body_parts.long(), body_parts_rgb


# def naive_vertices_converter(vertices, faces, percentage=0.5, max_sig_rate=-1):
#     if torch.is_tensor(vertices):
#         vertices = vertices.numpy()
#         faces = faces.numpy()
#         is_torch = True
#     else:
#         is_torch = False

#     default_l = 10 * np.sum((vertices.max(axis=0) - vertices.min(axis=0)) ** 2) ** 0.5 / vertices.shape[0]
#     average_len = get_vert_edge_length(vertices, faces, default_l)

#     sigma = (average_len ** 2) / (2 * np.log(1 / percentage)) + 1e-10
#     isigma = 1 / sigma

#     if max_sig_rate > 0:
#         thr = np.mean(isigma) * max_sig_rate
#         isigma[isigma > thr] = thr

#     if is_torch:
#         return torch.from_numpy(vertices).type(torch.float32), torch.from_numpy(isigma).type(torch.float32), None
#     else:
#         return vertices, isigma, None


def generate_part_labels_voge(keypoints3d, vert_to_part, neural_renderer, img):
    nparts = vert_to_part.max()  # TODO: bug? nparts = vert_to_part.shape[1]-1

    gmesh = human_pose_guassian_mesh(keypoints3d.detach().cpu().numpy())

    fragments = neural_renderer(gmesh)

    pix_to_kernel = fragments.vert_index

    mask = (pix_to_face != -1).to(torch.float)[:, :, :, 0]

    body_parts = interpolate_attr(fragments, pix_to_kernel).permute((0, 3, 1, 2))

    body_parts = body_parts.argmax(axis=1).detach() * mask.squeeze(1)

    body_parts_rgb = body_parts / nparts

    return body_parts.long(), body_parts_rgb


def one_hot(y, max_size=None):
    if not max_size:
        max_size = int(torch.max(y).item() + 1)
    y = y.view(-1, 1)
    y_onehot = torch.zeros((y.shape[0], max_size), dtype=torch.float32, device=y.device)
    y_onehot.scatter_(1, y.type(torch.long), 1)
    return y_onehot


def get_vert_to_part(device='cpu'):
    smpl_segmentation = joblib.load('./data/smpl_partSegmentation_mapping.pkl')
    v2p = torch.from_numpy(smpl_segmentation['smpl_index'])
    nparts = 24
    v2p_onehot = one_hot(v2p + 1, nparts + 1)
    return v2p_onehot.to(device)


# def save_partseg(vertices_all, kernel, visual=False):
#     K = 10
#     smpl_segmentation = joblib.load('./data/smpl_partSegmentation_mapping.pkl')
#     kernel_idx = torch.norm(kernel.unsqueeze(1).repeat(1,6890,1)[:,:,:2] - vertices_all[:,:2], dim=2)#.min(dim=1)[1]
#     kernel_idx = torch.sort(kernel_idx, dim=1)[1][:,:K]
#     v2p = torch.from_numpy(smpl_segmentation['smpl_index'])
#     v2p_kernel, v2p_idx = torch.mode(v2p[kernel_idx], dim=1)
#     np.save('partseg.npy', v2p_kernel.numpy(), allow_pickle=True)
#     if visual:
#         vertices = vertices_all[kernel_idx][np.arange(len(v2p_idx)).tolist(),v2p_idx]
#         kernel_pcl = o3d.geometry.PointCloud()
#         colors = torch.Tensor(color_list) / 255
#         kernel_pcl.points = o3d.utility.Vector3dVector(kernel.cpu().detach().numpy())
#         kernel_pcl.colors = o3d.utility.Vector3dVector(colors[v2p_kernel].cpu().detach().numpy())
#         o3d.io.write_point_cloud("nr_kernel_location.ply", kernel_pcl)
#         kernel_pcl.points = o3d.utility.Vector3dVector(vertices.cpu().detach().numpy())
#         kernel_pcl.colors = o3d.utility.Vector3dVector(colors[v2p_kernel].cpu().detach().numpy())
#         o3d.io.write_point_cloud("nr_verts_location.ply", kernel_pcl)
#         kernel_pcl.points = o3d.utility.Vector3dVector(vertices_all.cpu().detach().numpy())
#         kernel_pcl.colors = o3d.utility.Vector3dVector(colors[v2p].cpu().detach().numpy())
#         o3d.io.write_point_cloud("verts_location.ply", kernel_pcl)
#     print('part seg have saved successfully')


def get_vert_to_part_nr(device='cpu'):
    v2p = torch.Tensor(np.load('partseg.npy', allow_pickle=True)).long()
    nparts = 24
    v2p_onehot = one_hot(v2p + 1, nparts + 1)
    return v2p_onehot.to(device)


# def generate_part_labels(vertices, faces, cam_t, neural_renderer, body_part_texture, K, R, part_bins):
#     batch_size = vertices.shape[0]

#     body_parts, depth, mask = neural_renderer(
#         vertices,
#         faces.expand(batch_size, -1, -1),
#         textures=body_part_texture.expand(batch_size, -1, -1, -1, -1, -1),
#         K=K.expand(batch_size, -1, -1),
#         R=R.expand(batch_size, -1, -1),
#         t=cam_t.unsqueeze(1),
#     )

#     render_rgb = body_parts.clone()

#     body_parts = body_parts.permute(0, 2, 3, 1)
#     body_parts *= 255. # multiply it with 255 to make labels distant
#     body_parts, _ = body_parts.max(-1) # reduce to single channel

#     body_parts = torch.bucketize(body_parts.detach(), part_bins, right=True) # np.digitize(body_parts, bins, right=True)

#     # add 1 to make background label 0
#     body_parts = body_parts.long() + 1
#     body_parts = body_parts * mask.detach()

#     return body_parts.long(), render_rgb


def get_visibility(meshes, fragments):
    # pix_to_face is of shape (N, H, W, 1)
    pix_to_face = fragments.pix_to_face
    # print(torch.unique(pix_to_face, return_counts=True))
    mask = (pix_to_face != -1).to(torch.float)[:, :, :, 0]
    # ipdb.set_trace()
    # (F, 3) where F is the total number of faces across all the meshes in the batch
    packed_faces = meshes.faces_packed()
    # (V, 3) where V is the total number of verts across all the meshes in the batch
    packed_verts = meshes.verts_packed()
    visibility_map = torch.zeros(
        packed_verts.shape[0], device=pix_to_face.device
    )  # (V,)

    # Indices of unique visible faces
    visible_faces = pix_to_face.unique()  # (num_visible_faces )
    visible_faces = visible_faces[visible_faces >= 0]
    # print(visible_faces, visible_faces.shape, packed_faces.shape)

    # Get Indices of unique visible verts using the vertex indices in the faces
    visible_verts_idx = packed_faces[visible_faces]  # (num_visible_faces,  3)
    unique_visible_verts_idx = torch.unique(visible_verts_idx)  # (num_visible_verts, )

    # Update visibility indicator to 1 for all visible vertices
    visibility_map[unique_visible_verts_idx] = 1.0
    return visibility_map.view(len(meshes), -1).bool()


# def get_mask_and_visibility_voge(vertices, neural_renderer, image_size=224, coke_features=None, downsample_rate=4):
#     """
#         vertices: N, 49, 3
#         mask: N, res/4, res/4
#         pix_to_kernel: N, res/4, res/4, M, refers to which kernels contributes this pixel
#     """
#     N = vertices.shape[0]

#     sigma = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]
#     scale_base = 7500 #60000 #7500 #15000 #7500
#     sigmas = torch.Tensor(sigma).unsqueeze(0).repeat(vertices.shape[0],vertices.shape[1],1,1).float().to(device) * scale_base

#     # _, T = look_at_view_transform(dist=175, elev=0, azim=0, device=device)

#     gmesh = GaussianMeshesNaive(vertices.to(device), sigmas)

#     fragments = neural_renderer(gmesh)

#     # fragments = neural_renderer(gmesh, T=T.repeat(N,1))

#     # img = to_white_background(fragments, (gmesh.verts + 1) / 3).clamp(0, 1)
#     # np.save('baseline_visualRender.npy',img[5].detach().cpu().numpy())
#     pix_to_kernel = fragments.vert_index # b, h, w, K
#     mask = (pix_to_kernel != -1).to(torch.float)[:,:,:,0]
#     mask = F.interpolate(mask.unsqueeze(0), (image_size//downsample_rate, image_size//downsample_rate)).squeeze(0)

#     P = gmesh.verts.shape[1]
#     if coke_features is not None:
#         def unshift_vert_index(vert_index, n_vert):
#             N = vert_index.shape[0]
#             return vert_index - (vert_index != -1).to(torch.int)[:,:,:,:] * n_vert * torch.arange(0, N)[:, None, None, None].type_as(vert_index)
#
#         # old_vi = pix_to_kernel
#         # fragments.vert_index = unshift_vert_index(old_vi, P)
#         # vert_feat, vert_sum_w = sample_features(fragments, coke_features.permute((0,2,3,1)), n_vert=P)
#         vert_feat, vert_sum_w, weights = sample_features_py(fragments, coke_features.permute((0,2,3,1)), n_vert=P*N)
#         vert_feat = vert_feat / (vert_sum_w[:,:,None]+1e-8)

#
#     visibility_map = torch.zeros(N, P+1).long()
#     pix_to_kernel = pix_to_kernel[:, :, :, 0].view(N,-1)
#     for i in range(N):
#         visibility_map[i][(torch.clamp(torch.unique(pix_to_kernel[i]) - i*P, min=-1) + 1).long()] = 1

#     # # np.savetxt('mask.txt',mask[5].cpu().detach().numpy(),fmt='%d',delimiter=' ')
#     # ipdb.set_trace()

#     if coke_features is not None:
#         return mask, visibility_map[:,1:].long().to(vertices.device).bool(), vert_feat
#     else:
#         return mask, visibility_map[:,1:].long().to(vertices.device).bool(), None


def get_occ_visibility(
    N,
    P,
    pix_to_kernel,
    img_size,
    have_occ=None,
    occ_stride=None,
    occ_size=None,
    occ_idx=None,
):
    visibility_map = torch.zeros(N, P + 1).long()

    if (have_occ > 0).any():
        # h, w = occ_idx
        h, w = occ_idx[:, 0], occ_idx[:, 1]
        # calculate relative occ_size and occ_stride
        ratio = img_size / constants.IMG_RES
        occ_size = (occ_size.float() * ratio).long()
        occ_stride = (occ_stride.float() * ratio).long()
        h_start = h * occ_stride
        w_start = w * occ_stride
        h_end = torch.clamp(
            h_start + occ_size, max=img_size
        )  # .unsqueeze(1).repeat(1,P)
        w_end = torch.clamp(
            w_start + occ_size, max=img_size
        )  # .unsqueeze(1).repeat(1,P)
        # h_start = h_start.unsqueeze(1).repeat(1,P)
        # w_start = w_start.unsqueeze(1).repeat(1,P)

    for i in range(N):
        if have_occ[i]:
            pix_to_kernel[i].reshape(img_size, img_size)[
                w_start[i] : w_end[i], h_start[i] : h_end[i]
            ] = -1
            # [torch.logical_and(
            #         torch.logical_and(
            #         torch.logical_and( keypoints[:, 1] < h_end[i], keypoints[:, 1] > h_start[i]),
            #         keypoints[:, 0] < w_end[i]),
            #         keypoints[:, 0] > w_start[i])] = -1
        visibility_map[i][
            (torch.clamp(torch.unique(pix_to_kernel[i]) - i * P, min=-1) + 1).long()
        ] = 1

    return visibility_map


def get_mask_and_visibility_voge(
    vertices,
    neural_renderer,
    img_size,
    have_occ,
    occ_stride=None,
    occ_size=None,
    occ_idx=None,
    coke_features=None,
):
    """
    vertices: N, 49, 3
    mask: N, res/4, res/4
    pix_to_kernel: N, res/4, res/4, M, refers to which kernels contributes this pixel
    """
    N = vertices.shape[0]

    sigma = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]
    scale_base = 2000  # 7500
    sigmas = (
        torch.Tensor(sigma)
        .unsqueeze(0)
        .repeat(vertices.shape[0], vertices.shape[1], 1, 1)
        .float()
        .to(device)
        * scale_base
    )

    # _, T = look_at_view_transform(dist=175, elev=0, azim=0, device=device)

    gmesh = GaussianMeshesNaive(vertices.to(device), sigmas)

    fragments = neural_renderer(gmesh)
    # fragments = neural_renderer(gmesh, T=T.repeat(N,1))

    # img = to_white_background(fragments, (gmesh.verts + 1) / 3).clamp(0, 1)
    # np.save('baseline_visualRender.npy',img[5].detach().cpu().numpy())
    # ipdb.set_trace()
    pix_to_kernel = fragments.vert_index
    mask = (pix_to_kernel != -1).to(torch.float)[:, :, :, 0]
    mask = F.interpolate(mask.unsqueeze(0), (img_size, img_size)).squeeze(0)
    # np.savetxt('mask.txt',mask[5].cpu().detach().numpy(),fmt='%d',delimiter=' ')

    P = gmesh.verts.shape[1]
    if coke_features is not None:
        n_coke_features = 1
        if isinstance(coke_features, (list, tuple)):
            n_coke_features = len(coke_features)
            coke_features = torch.cat(coke_features, dim=1)  # concat along channel dim

        def unshift_vert_index(vert_index, n_vert):
            N = vert_index.shape[0]
            return vert_index - (vert_index != -1).to(torch.int)[
                :, :, :, :
            ] * n_vert * torch.arange(0, N)[:, None, None, None].type_as(vert_index)

        ## TODO: debug C++ code
        # old_vi = pix_to_kernel
        # fragments.vert_index = unshift_vert_index(old_vi, P)
        # vert_feat, vert_sum_w = sample_features(fragments, coke_features.permute((0,2,3,1)), n_vert=P)
        if torch.isnan(coke_features).any():
            logger.debug('coke_features nan')
            coke_features[torch.where(torch.isnan(coke_features))] = 0.0

        # vert_feat:  B, P, C
        vert_feat, vert_sum_w, weights = sample_features_py(
            fragments, coke_features.permute((0, 2, 3, 1)), n_vert=P * N
        )
        vert_feat = vert_feat / (vert_sum_w[:, :, None] + 1e-8)
        if torch.isnan(vert_feat).any():
            logger.debug('vert_feat nan')
            vert_feat[torch.where(torch.isnan(vert_feat))] = 0.1
        if n_coke_features > 1:
            vert_feat = vert_feat.chunk(n_coke_features, dim=2)
    pix_to_kernel = pix_to_kernel[:, :, :, 0].view(N, -1)

    visibility_map = get_occ_visibility(
        N, P, pix_to_kernel, img_size, have_occ, occ_stride, occ_size, occ_idx
    )

    if coke_features is not None:
        return mask, visibility_map[:, 1:].long().to(vertices.device).bool(), vert_feat
    else:
        return mask, visibility_map[:, 1:].long().to(vertices.device).bool(), None


def get_mask_and_visibility(
    vertices,
    faces,
    rasterizer,
):
    verts_rgb = torch.ones_like(vertices)  # (N, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb)
    meshes = Meshes(
        verts=vertices,
        faces=faces.expand((vertices.shape[0],) + faces.shape),
        textures=textures,
    )
    # meshes = Meshes(verts=vertices, faces=faces[None], textures=textures) # faces do not broadcast!
    # https://github.com/facebookresearch/pytorch3d/issues/126
    # Get the output from rasterization
    fragments = rasterizer(meshes)

    # pix_to_face is of shape (N, H, W, 1)
    pix_to_face = fragments.pix_to_face
    # print(torch.unique(pix_to_face, return_counts=True))
    mask = (pix_to_face != -1).to(torch.float)[:, :, :, 0]

    # (F, 3) where F is the total number of faces across all the meshes in the batch
    packed_faces = meshes.faces_packed()
    # (V, 3) where V is the total number of verts across all the meshes in the batch
    packed_verts = meshes.verts_packed()
    visibility_map = torch.zeros(
        packed_verts.shape[0], device=pix_to_face.device
    )  # (V,)

    # Indices of unique visible faces
    visible_faces = pix_to_face.unique()  # (num_visible_faces )
    visible_faces = visible_faces[visible_faces >= 0]
    # print(visible_faces, visible_faces.shape, packed_faces.shape)

    # Get Indices of unique visible verts using the vertex indices in the faces
    visible_verts_idx = packed_faces[visible_faces]  # (num_visible_faces,  3)
    unique_visible_verts_idx = torch.unique(visible_verts_idx)  # (num_visible_verts, )

    # Update visibility indicator to 1 for all visible vertices
    visibility_map[unique_visible_verts_idx] = 1.0

    return mask, visibility_map.view(vertices.shape[0], -1).bool()


def get_orient_label(cosx, N):
    x = torch.arccos(cosx)
    return (torch.clip(x, 0, np.pi - 1e-3) / (np.pi / N)).long()


def get_part_orients(openpose_joints):
    part_orients = []
    for part in constants.nemo_part_list:
        lnk = constants.openpose_links[part]
        part_vec = openpose_joints[:, lnk[1]] - openpose_joints[:, lnk[0]]
        part_orients.append(part_vec.unsqueeze(1))

    part_orients = F.normalize(
        torch.cat(tuple(part_orients), dim=1), p=2, dim=2
    )  # N, P, 3
    return part_orients


def get_vert_orients(
    openpose_joints,
    n_orient,
    jointsKernel_included=None,
    K_kernel_skeleton_ellipsoid=None,
    K_kernel_skeleton_sphere=None,
    K_num=None,
):
    """
    Args:
        openpose_joints: N, 25, 3
        n_orient: int
    Returns:
        vert_orients: N, V
    """
    part_orients = get_part_orients(openpose_joints)  # N, P, 3
    z_vec = torch.tensor(
        [[0, 0, 1]], dtype=torch.float32, device=openpose_joints.device
    )  # 1, 3
    part_proj = torch.bmm(
        part_orients, z_vec.unsqueeze(2).expand(part_orients.shape[0], -1, -1)
    ).squeeze(
        -1
    )  # N, P

    if n_orient == 3:
        thr = 0.3  # 0.01
        part_orients_label = torch.zeros_like(part_proj)
        part_orients_label[part_proj > thr] = 1
        part_orients_label[part_proj < -thr] = 2
    elif n_orient == 2:
        part_orients_label = torch.zeros_like(part_proj)
        part_orients_label[part_proj > 0] = 0
        part_orients_label[part_proj < 0] = 1
    else:
        part_orients_label = get_orient_label(part_proj, n_orient)

    if K_num is not None:
        # vert_orients = np.array( [ part_orients_label[constants.vertex_to_part[i]] for i in range(len(constants.vertex_to_part))])
        # vert_orients N, V
        # if jointsKernel_included and K_kernel_skeleton_sphere:
        #     v2p = torch.tensor(constants.vertex_to_part_voge_kernels, device=part_orients_label.device)
        # elif jointsKernel_included:
        #     v2p = torch.tensor(constants.vertex_to_part_voge_joints, device=part_orients_label.device)
        # elif K_kernel_skeleton_ellipsoid and K_kernel_skeleton_sphere:
        #     v2p = torch.tensor(constants.vertex_to_part_voge_Kkernels_es, device=part_orients_label.device)
        # elif K_kernel_skeleton_sphere:
        #     v2p = torch.tensor(constants.vertex_to_part_voge_Kkernels_s, device=part_orients_label.device)
        # elif K_kernel_skeleton_ellipsoid:
        #     v2p = torch.tensor(constants.vertex_to_part_voge_Kkernels_e, device=part_orients_label.device)
        # else:
        #     v2p = torch.tensor(constants.vertex_to_part_voge, device=part_orients_label.device)

        v2p = (
            torch.cat(
                (
                    torch.Tensor(constants.vertex_to_part_voge_kernels).repeat(
                        K_num + 1
                    ),
                    torch.Tensor(constants.vertex_to_part_voge_joints),
                )
            )
            .to(part_orients_label.device)
            .long()
        )
        v2p = torch.cat(
            (torch.Tensor(constants.vertex_to_part).to(part_orients_label.device), v2p)
        ).long()
    else:
        v2p = torch.tensor(constants.vertex_to_part, device=part_orients_label.device)
    vert_orients = part_orients_label[:, v2p]

    return vert_orients


def get_vert_pof(openpose_joints):
    part_orients = []
    for part in constants.nemo_part_list:
        lnk = constants.openpose_links[part]
        part_vec = openpose_joints[:, lnk[1]] - openpose_joints[:, lnk[0]]
        part_orients.append(part_vec.unsqueeze(1))

    part_orients = F.normalize(
        torch.cat(tuple(part_orients), dim=1), p=2, dim=2
    )  # N, P, 3

    v2p = torch.tensor(constants.vertex_to_part, device=part_orients.device)
    vert_pof = part_orients[:, v2p]

    return vert_pof


# def get_part_orient(pof, openpose_kps):
#     """
#     Args:
#         pof: B, 3, H, W
#         openpose_kps: B, 25, 3
#     Returns:
#         part_orients: B, P, 3
#     """
#     part_orients = []
#     for part in constants.nemo_part_list:
#         lnk = constants.openpose_links[part]
#         openpose_kps[lnk[0]]
#         # part_vec = openpose_joints[:, lnk[1]] - openpose_joints[:, lnk[0]]
#         # part_orients.append(part_vec.unsqueeze(1))


def sample_feature(feat, xs):
    """
    Args:
        feat: B, C, H, W
        xs: B, N, 2
    """
    xs = xs.unsqueeze(1)  # B, 1, N, 2
    feat = F.grid_sample(
        feat, xs, mode='bilinear', padding_mode='border', align_corners=True
    ).squeeze(
        2
    )  # B, C, 1, N
    return feat
