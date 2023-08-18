import argparse
import os
import os.path as osp
import pickle as pkl
from distutils.util import strtobool
import ipdb
import json
import cv2
import numpy as np
import torch
from tqdm import tqdm

import mmcv
from mmcv import DictAction

from mmhuman3d.data.datasets import build_dataset

from mmhuman3d.utils.vis_utils import SMPLVisualizer, get_part_seg_vis
from mmhuman3d.utils.neural_renderer import get_cameras
from mmhuman3d.utils.geometry import convert_weak_perspective_to_perspective
from mmhuman3d.models.builder import build_body_model

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize predictions')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument(
        '--output_file', help='the results to be visualized')
    parser.add_argument('--outdir', help='output image directory')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument('--vis_partseg', action='store_true', help='whether to visualize part segmentation')
    parser.add_argument(
        '--error_file', default='', help='the element-wise error file')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    args = parser.parse_args()
    return args


def visualize_prediction(img, pose, betas, camera, visualizer, color=(0.6, 0.6, 1), vis_res=512, device='cpu'):
    img_tensor = torch.tensor(img[None], device=device)
    pred_pose = torch.tensor(pose[None], device=device).float()
    pred_betas = torch.tensor(betas[None], device=device).float()
    pred_cam = torch.tensor(camera[None], device=device).float()
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
                            color=color, background=img_tensor)
    res_rot = visualizer.visualize(pose=pred_pose, betas=pred_betas, 
                            color=color, background=None,
                            rotate_around_y=90)
    return res['image'], res_rot['image']

def visualize_prediction_pth(img, pose, betas, camera, visualizer, vis_res=512, device='cpu'):
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
                            color=(0.6, 0.6, 1), background=img)
    res_rot = visualizer.visualize(pose=pred_pose, betas=pred_betas, 
                            color=(0.6, 0.6, 1), background=None,
                            rotate_around_y=90)
    return res['image'], res_rot['image']

def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.data.test.test_mode = True

    if args.output_file.endswith('.pkl'):
        outputs = pkl.load(open(args.output_file, 'rb'))
    elif args.output_file.endswith('.json'):
        outputs = mmcv.load(args.output_file)
    else:
        raise NotImplementedError

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = build_dataset(cfg.data.visualization)
    if cfg.data.visualization.type in ['OccludedHumanImageDataset', 
                                       'OcclusionVaryingHumanImageDataset']:
        cfg.data.visualization = cfg.data.visualization.orig_cfg

    body_model = build_body_model(cfg.data.visualization.body_model).to(device)

    visualizer = SMPLVisualizer(body_model, 'cuda', None,
             image_size=(cfg.hparams.VISUALIZER.IMG_RES, cfg.hparams.VISUALIZER.IMG_RES),
             point_light_location=((0,0,-3.0),)) 
    
    if cfg.data.visualization.hparams.get('indices', None) is not None:
        indices = json.load(open(cfg.data.visualization.hparams.indices))[:len(dataset)]
    else:
        indices = np.arange(0, len(dataset), 1)

    color = (1, 0.6, 0.6)

    error_type = 'MPJPE'
    if os.path.isfile(args.error_file) and args.error_file.endswith('.npy'):
        errors = np.load(args.error_file)
    elif osp.isfile(osp.join(osp.dirname(args.output_file), 'mpjpe_elementwise.npy')):
        errors = np.load(osp.join(osp.dirname(args.output_file), 'mpjpe_elementwise.npy'))
    else:
        errors = None

    for i in tqdm(indices):
        data = dataset[i]
        img = data['img']
        pose = np.array(outputs['poses'][i])
        betas = np.array(outputs['betas'][i])
        camera = np.array(outputs['cameras'][i])

        kp2d_op = np.array(outputs['keypoints2d'][i][:25])
        kp2d_op[:, :2] = kp2d_op[:, :2] * cfg.hparams.VISUALIZER.IMG_RES / cfg.hparams.DATASET.IMG_RES
        if (hasattr(cfg.hparams.MODEL, 'NON_STANDARD_WEAK_CAM')) and cfg.hparams.MODEL.NON_STANDARD_WEAK_CAM:
            camera[1:] = camera[1:] / camera[0:1]

        img_vis, img_vis_rot = visualize_prediction(img, pose, betas, camera, visualizer, color=color,
                        vis_res=cfg.hparams.VISUALIZER.IMG_RES, device=device)

        # for j in range(kp2d_op.shape[0]):
        #     cv2.circle(img_vis[0], (int(kp2d_op[j][0]), int(kp2d_op[j][1])), 2, (0, 0, 255), 2)

        # Add error metric
        if errors is not None:
            error = errors[i]
            cv2.putText(img_vis[0], f'{error_type}: {error:.1f}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img_vis_rot[0], f'{error_type}: {error:.1f}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        save_path = osp.join(args.outdir, f'{i:06d}.png')

        vis_final = np.concatenate((img, img_vis[0], img_vis_rot[0]), axis=1)
        if args.vis_partseg:
            part_seg = np.load(osp.join(osp.dirname(args.output_file), 'result_pred_segm_mask', f'{i}.npy')).transpose(1, 2, 0)
            part_seg = np.array(get_part_seg_vis(part_seg, background=img))[:, :, :3]
            part_seg_vis = cv2.resize(part_seg, (cfg.hparams.VISUALIZER.IMG_RES, cfg.hparams.VISUALIZER.IMG_RES))
            vis_final = np.concatenate((vis_final, part_seg_vis), axis=1)
        cv2.imwrite(save_path, vis_final)

if __name__ == '__main__':
    main()
