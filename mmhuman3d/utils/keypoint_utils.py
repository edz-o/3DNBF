from typing import Optional, Tuple, Union

import numpy as np

from mmhuman3d.core.conventions.keypoints_mapping import KEYPOINTS_FACTORY
from mmhuman3d.core.conventions.keypoints_mapping.human_data import (
    HUMAN_DATA_LIMBS_INDEX,
    HUMAN_DATA_PALETTE,
)


def search_limbs(
        data_source: str,
        mask: Optional[Union[np.ndarray, tuple, list]] = None,
        keypoints_factory: dict = KEYPOINTS_FACTORY) -> Tuple[dict, dict]:
    """Search the corresponding limbs following the basis human_data limbs. The
    mask could mask out the incorrect keypoints.

    Args:
        data_source (str): data source type.
        mask (Optional[Union[np.ndarray, tuple, list]], optional):
            refer to keypoints_mapping. Defaults to None.
        keypoints_factory (dict, optional): Dict of all the conventions.
            Defaults to KEYPOINTS_FACTORY.
    Returns:
        Tuple[dict, dict]: (limbs_target, limbs_palette).
    """
    limbs_source = HUMAN_DATA_LIMBS_INDEX
    limbs_palette = HUMAN_DATA_PALETTE
    keypoints_source = keypoints_factory['human_data']
    keypoints_target = keypoints_factory[data_source]
    limbs_target = {}
    for k, part_limbs in limbs_source.items():
        limbs_target[k] = []
        for limb in part_limbs:
            flag = False
            if (keypoints_source[limb[0]]
                    in keypoints_target) and (keypoints_source[limb[1]]
                                              in keypoints_target):
                if mask is not None:
                    if mask[keypoints_target.index(keypoints_source[
                            limb[0]])] != 0 and mask[keypoints_target.index(
                                keypoints_source[limb[1]])] != 0:
                        flag = True
                else:
                    flag = True
                if flag:
                    limbs_target.setdefault(k, []).append([
                        keypoints_target.index(keypoints_source[limb[0]]),
                        keypoints_target.index(keypoints_source[limb[1]])
                    ])
        if k in limbs_target:
            if k == 'body':
                np.random.seed(0)
                limbs_palette[k] = np.random.randint(
                    0, high=255, size=(len(limbs_target[k]), 3))
            else:
                limbs_palette[k] = np.array(limbs_palette[k])
    return limbs_target, limbs_palette

def remove_outside_batch(kp2d, bbox_xywh):
    """ 
    kp: B, K, 3
    bbox_xywh: B, 4 - top left xy and w, h
    """
    kp2d_rel = kp2d[:, :, :2] - bbox_xywh[:, None, :2]
    invisible = (np.sum(kp2d_rel < 0, axis=2) + np.sum(kp2d_rel > bbox_xywh[:, None, 2:] - 1, axis=2)) > 0
    # vis = vis * (~invisible)
    # kp2d[invisible] = 0 # remove this?
    kp2d[:, :, 2] = kp2d[:, :, 2] * (~invisible)
    return kp2d