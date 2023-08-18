import math
import random

import cv2
import mmcv
import numpy as np
from PIL import Image
import os
from glob import glob

from mmhuman3d.core.conventions import constants
from mmhuman3d.core.conventions.keypoints_mapping import get_flip_pairs
from ..builder import PIPELINES
from .compose import Compose


def get_affine_transform(
    center, scale, rot, output_size, shift=(0.0, 0.0), inv=False, pixel_std=1.0
):
    """Get the affine transform matrix, given the center/scale/rot/output_size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)
    Returns:
        np.ndarray: The transform matrix.
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    assert len(shift) == 2

    scale_tmp = scale * pixel_std

    shift = np.array(shift)
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = rotate_point([0.0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0.0, dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, trans_mat):
    """Apply an affine transformation to the points.

    Args:
        pt (np.ndarray): a 2 dimensional point to be transformed
        trans_mat (np.ndarray): 2x3 matrix of an affine transform
    Returns:
        np.ndarray: Transformed points.
    """
    assert len(pt) == 2
    new_pt = np.array(trans_mat) @ np.array([pt[0], pt[1], 1.0])

    return new_pt


def _get_3rd_point(a, b):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.
    Args:
        a (np.ndarray): point(x,y)
        b (np.ndarray): point(x,y)
    Returns:
        np.ndarray: The 3rd point.
    """
    assert len(a) == 2
    assert len(b) == 2
    direction = a - b
    third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)

    return third_pt


def rotate_point(pt, angle_rad):
    """Rotate a point by an angle.

    Args:
        pt (list[float]): 2 dimensional point to be rotated
        angle_rad (float): rotation angle by radian
    Returns:
        list[float]: Rotated point.
    """
    assert len(pt) == 2
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    new_x = pt[0] * cs - pt[1] * sn
    new_y = pt[0] * sn + pt[1] * cs
    rotated_pt = [new_x, new_y]

    return rotated_pt


def get_warp_matrix(theta, size_input, size_dst, size_target):
    """Calculate the transformation matrix under the constraint of unbiased.

    Paper ref: Huang et al. The Devil is in the Details: Delving into Unbiased
    Data Processing for Human Pose Estimation (CVPR 2020).
    Args:
        theta (float): Rotation angle in degrees.
        size_input (np.ndarray): Size of input image [w, h].
        size_dst (np.ndarray): Size of output image [w, h].
        size_target (np.ndarray): Size of ROI in input plane [w, h].
    Returns:
        matrix (np.ndarray): A matrix for transformation.
    """
    theta = np.deg2rad(theta)
    matrix = np.zeros((2, 3), dtype=np.float32)
    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]
    matrix[0, 0] = math.cos(theta) * scale_x
    matrix[0, 1] = -math.sin(theta) * scale_x
    matrix[0, 2] = scale_x * (
        -0.5 * size_input[0] * math.cos(theta)
        + 0.5 * size_input[1] * math.sin(theta)
        + 0.5 * size_target[0]
    )
    matrix[1, 0] = math.sin(theta) * scale_y
    matrix[1, 1] = math.cos(theta) * scale_y
    matrix[1, 2] = scale_y * (
        -0.5 * size_input[0] * math.sin(theta)
        - 0.5 * size_input[1] * math.cos(theta)
        + 0.5 * size_target[1]
    )
    return matrix


def warp_affine_joints(joints, mat):
    """Apply affine transformation defined by the transform matrix on the
    joints.

    Args:
        joints (np.ndarray[..., 2]): Origin coordinate of joints.
        mat (np.ndarray[3, 2]): The affine matrix.
    Returns:
        matrix (np.ndarray[..., 2]): Result coordinate of joints.
    """
    joints = np.array(joints)
    shape = joints.shape
    joints = joints.reshape(-1, 2)
    return np.dot(
        np.concatenate((joints, joints[:, 0:1] * 0 + 1), axis=1), mat.T
    ).reshape(shape)


def _construct_rotation_matrix(rot, size=3):
    """Construct the in-plane rotation matrix.

    Args:
        rot (float): Rotation angle (degree).
        size (int): The size of the rotation matrix.
            Candidate Values: 2, 3. Defaults to 3.
    Returns:
        rot_mat (np.ndarray([size, size]): Rotation matrix.
    """
    rot_mat = np.eye(size, dtype=np.float32)
    if rot != 0:
        rot_rad = np.deg2rad(rot)
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]

    return rot_mat


def _flip_smpl_pose(pose):
    """Flip SMPL pose parameters horizontally.

    Args:
        pose (np.ndarray([72])): SMPL pose parameters
    Returns:
        pose_flipped
    """

    flippedParts = [
        0,
        1,
        2,
        6,
        7,
        8,
        3,
        4,
        5,
        9,
        10,
        11,
        15,
        16,
        17,
        12,
        13,
        14,
        18,
        19,
        20,
        24,
        25,
        26,
        21,
        22,
        23,
        27,
        28,
        29,
        33,
        34,
        35,
        30,
        31,
        32,
        36,
        37,
        38,
        42,
        43,
        44,
        39,
        40,
        41,
        45,
        46,
        47,
        51,
        52,
        53,
        48,
        49,
        50,
        57,
        58,
        59,
        54,
        55,
        56,
        63,
        64,
        65,
        60,
        61,
        62,
        69,
        70,
        71,
        66,
        67,
        68,
    ]
    pose_flipped = pose[flippedParts]
    # Negate the second and the third dimension of the axis-angle
    pose_flipped[1::3] = -pose_flipped[1::3]
    pose_flipped[2::3] = -pose_flipped[2::3]
    return pose_flipped


def _flip_smpl_pose_batch(pose):
    """Flip SMPL pose parameters horizontally.

    Args:
        pose (torch.Tensor([B, 72])): SMPL pose parameters
    Returns:
        pose_flipped
    """

    flippedParts = [
        0,
        1,
        2,
        6,
        7,
        8,
        3,
        4,
        5,
        9,
        10,
        11,
        15,
        16,
        17,
        12,
        13,
        14,
        18,
        19,
        20,
        24,
        25,
        26,
        21,
        22,
        23,
        27,
        28,
        29,
        33,
        34,
        35,
        30,
        31,
        32,
        36,
        37,
        38,
        42,
        43,
        44,
        39,
        40,
        41,
        45,
        46,
        47,
        51,
        52,
        53,
        48,
        49,
        50,
        57,
        58,
        59,
        54,
        55,
        56,
        63,
        64,
        65,
        60,
        61,
        62,
        69,
        70,
        71,
        66,
        67,
        68,
    ]
    pose_flipped = pose[:, flippedParts]
    # Negate the second and the third dimension of the axis-angle
    pose_flipped[:, 1::3] = -pose_flipped[:, 1::3]
    pose_flipped[:, 2::3] = -pose_flipped[:, 2::3]
    return pose_flipped


def _flip_keypoints(keypoints, flip_pairs, img_width=None):
    """Flip human joints horizontally.

    Note:
        num_keypoints: K
        num_dimension: D
    Args:
        keypoints (np.ndarray([K, D])): Coordinates of keypoints.
        flip_pairs (list[tuple()]): Pairs of keypoints which are mirrored
            (for example, left ear -- right ear).
        img_width (int | None, optional): The width of the original image.
            To flip 2D keypoints, image width is needed. To flip 3D keypoints,
            we simply negate the value of x-axis. Default: None.
    Returns:
        keypoints_flipped
    """

    keypoints_flipped = keypoints.copy()

    # Swap left-right parts
    for left, right in flip_pairs:
        keypoints_flipped[left, :] = keypoints[right, :]
        keypoints_flipped[right, :] = keypoints[left, :]

    # Flip horizontally
    if img_width is None:
        keypoints_flipped[:, 0] = -keypoints_flipped[:, 0]
    else:
        keypoints_flipped[:, 0] = img_width - 1 - keypoints_flipped[:, 0]

    return keypoints_flipped


def _rotate_joints_3d(joints_3d, rot):
    """Rotate the 3D joints in the local coordinates.

    Notes:
        Joints number: K
    Args:
        joints_3d (np.ndarray([K, 3])): Coordinates of keypoints.
        rot (float): Rotation angle (degree).
    Returns:
        joints_3d_rotated
    """
    # in-plane rotation
    # 3D joints are rotated counterclockwise,
    # so the rot angle is inversed.
    rot_mat = _construct_rotation_matrix(-rot, 3)

    joints_3d_rotated = np.einsum('ij,kj->ki', rot_mat, joints_3d)
    joints_3d_rotated = joints_3d_rotated.astype('float32')
    return joints_3d_rotated


def _rotate_smpl_pose(pose, rot):
    """Rotate SMPL pose parameters.

    SMPL (https://smpl.is.tue.mpg.de/) is a 3D
    human model.
    Args:
        pose (np.ndarray([72])): SMPL pose parameters
        rot (float): Rotation angle (degree).
    Returns:
        pose_rotated
    """
    pose_rotated = pose.copy()
    if rot != 0:
        rot_mat = _construct_rotation_matrix(-rot)
        orient = pose[:3]
        # find the rotation of the body in camera frame
        per_rdg, _ = cv2.Rodrigues(orient.astype(np.float32))
        # apply the global rotation to the global orientation
        res_rot, _ = cv2.Rodrigues(np.dot(rot_mat, per_rdg))
        pose_rotated[:3] = (res_rot.T)[0]

    return pose_rotated


@PIPELINES.register_module()
class RandomHorizontalFlip(object):
    """Flip the image randomly.

    Flip the image randomly based on flip probaility.

    Args:
        flip_prob (float): probability of the image being flipped. Default: 0.5
    """

    def __init__(self, flip_prob=0.5, convention=None):
        assert 0 <= flip_prob <= 1
        self.flip_prob = flip_prob
        self.flip_pairs = get_flip_pairs(convention)

    def __call__(self, results):
        """Call function to flip image and annotations.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip' key is added into
                result dict.
        """
        if np.random.rand() > self.flip_prob:
            results['is_flipped'] = np.array([0])
            return results

        results['is_flipped'] = np.array([1])

        # flip image
        for key in results.get('img_fields', ['img']):
            results[key] = mmcv.imflip(results[key], direction='horizontal')

        # flip keypoints2d
        if 'keypoints2d' in results:
            assert self.flip_pairs is not None
            width = results['img'][:, ::-1, :].shape[1]
            keypoints2d = results['keypoints2d'].copy()
            keypoints2d = _flip_keypoints(keypoints2d, self.flip_pairs, width)
            results['keypoints2d'] = keypoints2d

        # flip bbox center
        center = results['center']
        center[0] = width - 1 - center[0]
        results['center'] = center

        # flip keypoints3d
        if 'keypoints3d' in results:
            assert self.flip_pairs is not None
            keypoints3d = results['keypoints3d'].copy()
            keypoints3d = _flip_keypoints(keypoints3d, self.flip_pairs)
            results['keypoints3d'] = keypoints3d

        # flip smpl
        if 'smpl_body_pose' in results:
            global_orient = results['smpl_global_orient'].copy()
            body_pose = results['smpl_body_pose'].copy().reshape((-1))
            smpl_pose = np.concatenate((global_orient, body_pose), axis=-1)
            smpl_pose_flipped = _flip_smpl_pose(smpl_pose)
            global_orient = smpl_pose_flipped[:3]
            body_pose = smpl_pose_flipped[3:]
            results['smpl_global_orient'] = global_orient
            results['smpl_body_pose'] = body_pose.reshape((-1, 3))
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_prob={self.flip_prob})'


@PIPELINES.register_module()
class CenterCrop(object):
    r"""Center crop the image.

    Args:
        crop_size (int | tuple): Expected size after cropping with the format
            of (h, w).
        efficientnet_style (bool): Whether to use efficientnet style center
            crop. Defaults to False.
        crop_padding (int): The crop padding parameter in efficientnet style
            center crop. Only valid if efficientnet style is True. Defaults to
            32.
        interpolation (str): Interpolation method, accepted values are
            'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'. Only valid if
             efficientnet style is True. Defaults to 'bilinear'.
        backend (str): The image resize backend type, accpeted values are
            `cv2` and `pillow`. Only valid if efficientnet style is True.
            Defaults to `cv2`.


    Notes:
        If the image is smaller than the crop size, return the original image.
        If efficientnet_style is set to False, the pipeline would be a simple
        center crop using the crop_size.
        If efficientnet_style is set to True, the pipeline will be to first to
        perform the center crop with the crop_size_ as:

        .. math::
        crop\_size\_ = crop\_size / (crop\_size + crop\_padding) * short\_edge

        And then the pipeline resizes the img to the input crop size.
    """

    def __init__(
        self,
        crop_size,
        efficientnet_style=False,
        crop_padding=32,
        interpolation='bilinear',
        backend='cv2',
    ):
        if efficientnet_style:
            assert isinstance(crop_size, int)
            assert crop_padding >= 0
            assert interpolation in (
                'nearest',
                'bilinear',
                'bicubic',
                'area',
                'lanczos',
            )
            if backend not in ['cv2', 'pillow']:
                raise ValueError(
                    f'backend: {backend} is not supported for '
                    'resize. Supported backends are "cv2", "pillow"'
                )
        else:
            assert isinstance(crop_size, int) or (
                isinstance(crop_size, tuple) and len(crop_size) == 2
            )
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.efficientnet_style = efficientnet_style
        self.crop_padding = crop_padding
        self.interpolation = interpolation
        self.backend = backend

    def __call__(self, results):
        crop_height, crop_width = self.crop_size[0], self.crop_size[1]
        for key in results.get('img_fields', ['img']):
            img = results[key]
            # img.shape has length 2 for grayscale, length 3 for color
            img_height, img_width = img.shape[:2]

            # https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/preprocessing.py#L118 # noqa
            if self.efficientnet_style:
                img_short = min(img_height, img_width)
                crop_height = (
                    crop_height / (crop_height + self.crop_padding) * img_short
                )
                crop_width = crop_width / (crop_width + self.crop_padding) * img_short

            y1 = max(0, int(round((img_height - crop_height) / 2.0)))
            x1 = max(0, int(round((img_width - crop_width) / 2.0)))
            y2 = min(img_height, y1 + crop_height) - 1
            x2 = min(img_width, x1 + crop_width) - 1

            # crop the image
            img = mmcv.imcrop(img, bboxes=np.array([x1, y1, x2, y2]))

            if self.efficientnet_style:
                img = mmcv.imresize(
                    img,
                    tuple(self.crop_size[::-1]),
                    interpolation=self.interpolation,
                    backend=self.backend,
                )
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(crop_size={self.crop_size}'
        repr_str += f', efficientnet_style={self.efficientnet_style}'
        repr_str += f', crop_padding={self.crop_padding}'
        repr_str += f', interpolation={self.interpolation}'
        repr_str += f', backend={self.backend})'
        return repr_str


@PIPELINES.register_module()
class Normalize(object):
    """Normalize the image.

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            results[key] = mmcv.imnormalize(
                results[key], self.mean, self.std, self.to_rgb
            )
        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={list(self.mean)}, '
        repr_str += f'std={list(self.std)}, '
        repr_str += f'to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float): How much to jitter brightness.
            brightness_factor is chosen uniformly from
            [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast.
            contrast_factor is chosen uniformly from
            [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation.
            saturation_factor is chosen uniformly from
            [max(0, 1 - saturation), 1 + saturation].
    """

    def __init__(self, brightness, contrast, saturation):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, results):
        brightness_factor = random.uniform(0, self.brightness)
        contrast_factor = random.uniform(0, self.contrast)
        saturation_factor = random.uniform(0, self.saturation)
        color_jitter_transforms = [
            dict(
                type='Brightness',
                magnitude=brightness_factor,
                prob=1.0,
                random_negative_prob=0.5,
            ),
            dict(
                type='Contrast',
                magnitude=contrast_factor,
                prob=1.0,
                random_negative_prob=0.5,
            ),
            dict(
                type='ColorTransform',
                magnitude=saturation_factor,
                prob=1.0,
                random_negative_prob=0.5,
            ),
        ]
        random.shuffle(color_jitter_transforms)
        transform = Compose(color_jitter_transforms)
        return transform(results)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(brightness={self.brightness}, '
        repr_str += f'contrast={self.contrast}, '
        repr_str += f'saturation={self.saturation})'
        return repr_str


@PIPELINES.register_module()
class Lighting(object):
    """Adjust images lighting using AlexNet-style PCA jitter.

    Args:
        eigval (list): the eigenvalue of the convariance matrix of pixel
            values, respectively.
        eigvec (list[list]): the eigenvector of the convariance matrix of pixel
            values, respectively.
        alphastd (float): The standard deviation for distribution of alpha.
            Defaults to 0.1
        to_rgb (bool): Whether to convert img to rgb.
    """

    def __init__(self, eigval, eigvec, alphastd=0.1, to_rgb=True):
        assert isinstance(
            eigval, list
        ), f'eigval must be of type list, got {type(eigval)} instead.'
        assert isinstance(
            eigvec, list
        ), f'eigvec must be of type list, got {type(eigvec)} instead.'
        for vec in eigvec:
            assert isinstance(vec, list) and len(vec) == len(
                eigvec[0]
            ), 'eigvec must contains lists with equal length.'
        self.eigval = np.array(eigval)
        self.eigvec = np.array(eigvec)
        self.alphastd = alphastd
        self.to_rgb = to_rgb

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            results[key] = mmcv.adjust_lighting(
                img,
                self.eigval,
                self.eigvec,
                alphastd=self.alphastd,
                to_rgb=self.to_rgb,
            )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(eigval={self.eigval.tolist()}, '
        repr_str += f'eigvec={self.eigvec.tolist()}, '
        repr_str += f'alphastd={self.alphastd}, '
        repr_str += f'to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class RandomChannelNoise:
    """Data augmentation with random channel noise.

    Required keys: 'img'
    Modifies key: 'img'
    Args:
        noise_factor (float): Multiply each channel with
         a factor between``[1-scale_factor, 1+scale_factor]``
    """

    def __init__(self, noise_factor=0.4):
        self.noise_factor = noise_factor

    def __call__(self, results):
        """Perform data augmentation with random channel noise."""
        img = results['img']

        # Each channel is multiplied with a number
        # in the area [1-self.noise_factor, 1+self.noise_factor]
        pn = np.random.uniform(1 - self.noise_factor, 1 + self.noise_factor, (1, 3))
        img = cv2.multiply(img, pn)

        results['img'] = img
        return results


@PIPELINES.register_module()
class GetRandomScaleRotation:
    """Data augmentation with random scaling & rotating.

    Required key: 'scale'. Modifies key: 'scale' and 'rotation'.
    Args:
        rot_factor (int): Rotating to ``[-2*rot_factor, 2*rot_factor]``.
        scale_factor (float): Scaling to ``[1-scale_factor, 1+scale_factor]``.
        rot_prob (float): Probability of random rotation.
    """

    def __init__(self, rot_factor=30, scale_factor=0.25, rot_prob=0.6):
        self.rot_factor = rot_factor
        self.scale_factor = scale_factor
        self.rot_prob = rot_prob

    def __call__(self, results):
        """Perform data augmentation with random scaling & rotating."""
        s = results['scale']

        sf = self.scale_factor
        rf = self.rot_factor

        s_factor = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        s = s * s_factor

        r_factor = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
        r = r_factor if np.random.rand() <= self.rot_prob else 0.0

        results['scale'] = s
        results['rotation'] = r

        return results


@PIPELINES.register_module()
class ExtremeCrop:
    """Extreme cropping augmentation from EFT
    (HACK) only works for smpl_49 convention
    head index 43
    """

    def __init__(self, crop_prob=0.3):
        self.crop_prob = 0.3
        self.head_idx = 43  # TODO: obtain head_idx from `results`

    def __call__(self, results):
        if np.random.rand() <= self.crop_prob:
            # import pdb; pdb.set_trace()
            if 'keypoints2d' in results:
                head = results['keypoints2d'].copy()[self.head_idx][:2]
                new_center = 0.5 * results['center'] + 0.5 * head
                new_scale = max(np.abs(head - new_center)) * 2 * 1.2
                results['center'] = new_center
                results['scale'][:] = new_scale
        return results


@PIPELINES.register_module()
class SampleBGTexture:
    """Sample a background texture from DTD dataset."""

    def __init__(self, img_res, bg_root='data/dtd/train'):
        self.image_size = np.array([img_res, img_res])  # W, H
        self.bg_textures = glob(os.path.join(bg_root, 'images', '*', '*.jpg'))

    def __call__(self, results):
        texture_file = random.choice(self.bg_textures)
        texture = Image.open(texture_file)
        texture_size = min(texture.size)

        sf = self.image_size[0] / texture_size
        texture_scale = np.random.uniform(sf + 0.1, 2 * sf)

        texture = texture.resize(
            (
                int(texture.size[0] * texture_scale),
                int(texture.size[1] * texture_scale),
            ),
            resample=Image.BILINEAR,
        )
        bg_img, texture_crop_tl = self.crop_texture(texture, self.image_size)

        results['bg_img'] = np.array(bg_img)
        return results

    def crop_texture(self, texture, size, texture_crop_tl=None):
        """
        :texture: PIL texture image
        :size: (w, h) size of occlusion patch
        """
        if texture_crop_tl is None:
            x_tl = random.randint(0, max(texture.size[0] - size[0], 0))
            y_tl = random.randint(0, max(texture.size[1] - size[1], 0))
        else:
            x_tl, y_tl = texture_crop_tl
        return texture.crop((x_tl, y_tl, x_tl + size[0], y_tl + size[1])), (
            x_tl,
            y_tl,
        )  # xyxy


@PIPELINES.register_module()
class MeshAffine:
    """Affine transform the image to get input image.

    Affine transform the 2D keypoints, 3D kepoints. Required keys: 'img',
    'pose', 'img_shape', 'rotation' and 'center'. Modifies key: 'img',
    ''keypoints2d', 'keypoints3d', 'pose'.
    """

    def __init__(self, img_res):
        self.img_res = img_res
        self.image_size = np.array([img_res, img_res])

    def __call__(self, results):
        c = results['center']
        s = results['scale']
        r = results['rotation']
        img = results['img']
        trans = get_affine_transform(c, s, r, self.image_size)

        # if 'img' in results:
        if 'bg_img' in results:
            bg_img = results['bg_img']
            if (
                bg_img.shape[0] != self.image_size[0]
                or bg_img.shape[1] != self.image_size[1]
            ):
                bg_img = cv2.resize(bg_img, (self.image_size[1], self.image_size[0]))
            img = cv2.warpAffine(
                img,
                trans,
                (int(self.image_size[0]), int(self.image_size[1])),
                bg_img,
                borderMode=cv2.BORDER_TRANSPARENT,
            )
        else:
            img = cv2.warpAffine(
                img,
                trans,
                (int(self.image_size[0]), int(self.image_size[1])),
                flags=cv2.INTER_LINEAR,
            )
        results['img'] = img

        if 'mask' in results:
            results['mask'] = cv2.warpAffine(
                results['mask'],
                trans,
                (int(self.image_size[0]), int(self.image_size[1])),
                flags=cv2.INTER_NEAREST,
            )

        if 'keypoints2d' in results:
            keypoints2d = results['keypoints2d'].copy()
            num_keypoints = len(keypoints2d)
            for i in range(num_keypoints):
                if keypoints2d[i][2] > 0.0:
                    keypoints2d[i][:2] = affine_transform(keypoints2d[i][:2], trans)
            results['keypoints2d'] = keypoints2d

        if 'keypoints3d' in results:
            keypoints3d = results['keypoints3d'].copy()
            keypoints3d[:, :3] = _rotate_joints_3d(keypoints3d[:, :3], r)
            results['keypoints3d'] = keypoints3d

        if 'smpl_body_pose' in results:
            global_orient = results['smpl_global_orient'].copy()
            body_pose = results['smpl_body_pose'].copy().reshape((-1))
            pose = np.concatenate((global_orient, body_pose), axis=-1)
            pose = _rotate_smpl_pose(pose, r)
            results['smpl_global_orient'] = pose[:3]
            results['smpl_body_pose'] = pose[3:].reshape((-1, 3))

        return results


@PIPELINES.register_module()
class AddOcclusionPatch(object):
    def __init__(
        self,
        occ_size=40,
        occ_stride=40,
        dtd_root='data/dtd',
        prob=1.0,
        phase='test',
        occlude_center=None,
    ):
        self.occ_size = occ_size
        self.occ_stride = occ_stride
        self.dtd_root = dtd_root
        self.dtd_textures = glob(os.path.join(dtd_root, 'images', '*', '*.jpg'))
        # self.dtd_textures = glob(os.path.join(dtd_root, '*.jpg')) # for coco
        self.prob = prob
        self.phase = phase
        self.occlude_center = occlude_center

    def __call__(self, results):
        if random.random() > self.prob:
            results['have_occ'] = False
            results['occ_size'] = 0
            results['occ_stride'] = 0
            results['texture_crop_tl'] = (0, 0)
            results['occ_idx'] = np.array([0, 0])
            results['occ_mask'] = np.zeros(results['img'].shape[:2])  # (320, 320)
            return results
        results['have_occ'] = True
        occ_size = results['occ_size'] if 'occ_size' in results else self.occ_size

        occ_stride = (
            results['occ_stride'] if 'occ_stride' in results else self.occ_stride
        )
        results['occ_size'] = occ_size
        results['occ_stride'] = occ_stride
        if 'texture_file' not in results:
            texture_file = random.choice(self.dtd_textures)
            results['texture_file'] = texture_file
        else:
            texture_file = results['texture_file']

        img = results['img']
        img_size = int(img.shape[0])
        ratio = img_size / constants.IMG_RES
        occ_size = int(round(occ_size * ratio))
        occ_stride = int(round(occ_stride * ratio))
        if 'occ_idx' not in results:
            output_height = int(math.ceil((img_size - occ_size) / occ_stride + 1))
            output_width = int(math.ceil((img_size - occ_size) / occ_stride + 1))
            h = random.choice(np.arange(output_height))
            w = random.choice(np.arange(output_width))
            occ_idx = np.array([h, w])
            results['occ_idx'] = occ_idx
        else:
            occ_idx = results['occ_idx']

        if 'texture_crop_tl' not in results:
            texture_crop_tl = None
        else:
            texture_crop_tl = results['texture_crop_tl']
            texture_crop_tl = [texture_crop_tl[0] * ratio, texture_crop_tl[1] * ratio]

        occ_img, texture_crop_tl, occ_mask = self.get_dtd_occluded_img(
            img,
            occ_idx,
            occ_size,
            occ_stride,
            texture_file,
            texture_crop_tl,
            occlude_center=self.occlude_center,
        )
        # try:
        # except:
        #     import IPython; IPython.embed(); exit()

        results['texture_crop_tl'] = texture_crop_tl
        results['occ_mask'] = occ_mask
        # results['img_org'] = results['img']
        results['img'] = occ_img
        if 'occ_img' in results:
            results['occ_img'] = occ_img

        if self.phase == 'test':
            results['keypoints2d'][:, 2] = AddOcclusionPatch.get_occ_visibility(
                results['keypoints2d'],
                results['keypoints2d'][:, 2],
                img_size,
                occ_stride,
                occ_size,
                occ_idx,
            )

        return results

    @staticmethod
    def get_occ_visibility(
        keypoints_2d, kp_visibility, img_size, occ_stride, occ_size, occ_idx
    ):
        """
        Args:
            keypoints_2d (np.array): (K, 2|3)
            kp_visibility (np.array): (K,)
            img_size (int): (TODO) support tuple
            occ_idx: (x, y)
            occ_size (int): size of occlusion patch
            occ_stride (int): stride of sliding window
        Returns:
            visibility (np.array): K
        """
        h, w = occ_idx
        h_start = h * occ_stride
        w_start = w * occ_stride
        h_end = min(img_size, h_start + occ_size)
        w_end = min(img_size, w_start + occ_size)

        kp_visibility[
            np.logical_and(
                np.logical_and(
                    np.logical_and(
                        keypoints_2d[:, 1] < h_end, keypoints_2d[:, 1] > h_start
                    ),
                    keypoints_2d[:, 0] < w_end,
                ),
                keypoints_2d[:, 0] > w_start,
            )
        ] = 0
        return kp_visibility

    @staticmethod
    def crop_texture(texture, size, texture_crop_tl=None):
        """
        :texture: PIL texture image
        :size: (w, h) size of occlusion patch
        """
        if texture_crop_tl is None:
            x_tl = random.randint(0, texture.size[0] - size[0])
            y_tl = random.randint(0, texture.size[1] - size[1])
        else:
            x_tl, y_tl = texture_crop_tl
        return texture.crop((x_tl, y_tl, x_tl + size[0], y_tl + size[1])), (
            x_tl,
            y_tl,
        )  # xyxy

    @staticmethod
    def get_dtd_occluded_img(
        img,
        occ_idx,
        occ_size,
        occ_stride,
        texture_file,
        texture_crop_tl=None,
        occlude_center=False,
    ):
        """
        :img: numpy array (H, W, 3)
        :occ_idx: (h, w)
        :occ_size: int
        :occ_stride: int
        :texture_file: filename of texture in dsd
        """
        h, w = occ_idx
        img_size = int(img.shape[0])

        texture = Image.open(texture_file)
        texture_size = min(texture.size)
        # resize texture to have fixed scale relative to image
        scaleFactor = 1
        scale = scaleFactor * img_size / texture_size
        texture = texture.resize(
            (int(texture.size[0] * scale), int(texture.size[1] * scale)),
            resample=Image.BILINEAR,
        )

        if occlude_center is not None:
            h_start = int(occlude_center[0] * img_size - occ_size / 2)
            w_start = int(occlude_center[1] * img_size - occ_size / 2)
            h_end = h_start + occ_size
            w_end = w_start + occ_size
        else:
            h_start = h * occ_stride
            w_start = w * occ_stride
            h_end = min(img_size, h_start + occ_size)
            w_end = min(img_size, w_start + occ_size)

        # Getting the image copy, applying the occluding window:
        occ_image = img.copy()
        # import pdb; pdb.set_trace()
        texture_cropped, texture_crop_tl = AddOcclusionPatch.crop_texture(
            texture, (w_end - w_start, h_end - h_start), texture_crop_tl
        )
        occ_patch = np.array(texture_cropped)
        occ_image[h_start:h_end, w_start:w_end, :] = occ_patch

        occ_mask = np.zeros(occ_image.shape[:2])
        occ_mask[h_start:h_end, w_start:w_end] = 1

        return occ_image, texture_crop_tl, occ_mask


@PIPELINES.register_module()
class BlendBackground(object):
    """Randomly place the image on a background image.

    Args:
        scale (float): the scale of the foreground image relative to the background Default: 0.75
    """

    def __init__(self, fg_scale=0.75, prob=0.5, bg_root='data/dtd/train'):
        self.fg_scale = fg_scale
        self.prob = prob
        self.bg_textures = glob(os.path.join(bg_root, 'images', '*', '*.jpg'))

    def __call__(self, results):
        """Call function to flip image and annotations.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Blended results, 'flip' key is added into
                result dict.
        """
        if np.random.rand() > self.prob:
            results['is_blended'] = np.array([0])
            return results

        results['is_blended'] = np.array([1])

        fg_scale = np.clip(np.random.randn() * 0.1 + self.fg_scale, 0.5, 1.0)
        # scale image
        img_size = int(results['img'].shape[0])  # assume square image
        img_resize = mmcv.imresize(
            results['img'], (int(img_size * fg_scale), int(img_size * fg_scale))
        )
        results['scale'] = np.array(results['scale'] * fg_scale)  # adjust human scale

        texture_file = random.choice(self.bg_textures)
        texture = Image.open(texture_file)
        texture_size = min(texture.size)

        sf = img_size / texture_size
        texture_scale = np.random.uniform(sf + 0.1, 2 * sf)

        texture = texture.resize(
            (
                int(texture.size[0] * texture_scale),
                int(texture.size[1] * texture_scale),
            ),
            resample=Image.BILINEAR,
        )
        texture_cropped, texture_crop_tl = self.crop_texture(
            texture, (img_size, img_size)
        )

        # randomly place foreground image on background
        x_tl = random.randint(0, max(texture_cropped.size[0] - img_resize.shape[1], 0))
        y_tl = random.randint(0, max(texture_cropped.size[1] - img_resize.shape[0], 0))
        results['center'] = results['center'] * fg_scale + np.array(
            [x_tl, y_tl]
        )  # adjust human center
        texture_cropped = np.array(texture_cropped)
        texture_cropped[
            y_tl : y_tl + img_resize.shape[0], x_tl : x_tl + img_resize.shape[1], :
        ] = img_resize
        results['img'] = texture_cropped

        # flip keypoints2d
        if 'keypoints2d' in results:
            keypoints2d = results['keypoints2d'].copy()
            keypoints2d[:, :2] = keypoints2d[:, :2] * fg_scale + np.array([x_tl, y_tl])
            results['keypoints2d'] = keypoints2d

        return results

    def crop_texture(self, texture, size, texture_crop_tl=None):
        """
        :texture: PIL texture image
        :size: (w, h) size of occlusion patch
        """
        if texture_crop_tl is None:
            x_tl = random.randint(0, max(texture.size[0] - size[0], 0))
            y_tl = random.randint(0, max(texture.size[1] - size[1], 0))
        else:
            x_tl, y_tl = texture_crop_tl
        return texture.crop((x_tl, y_tl, x_tl + size[0], y_tl + size[1])), (
            x_tl,
            y_tl,
        )  # xyxy

    def __repr__(self):
        return self.__class__.__name__ + f'(scale={self.fg_scale},prob={self.prob})'
