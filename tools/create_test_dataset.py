# coding: utf-8
import os, sys
import os.path as osp
import numpy as np
import json
import cv2

from mmhuman3d.data.data_structures.human_data import HumanData
from glob import glob

name = 'demo'
root = f"data/datasets/{name}" # path of demo image dataset

fmts = ['*.jpg', '*.png', '*.jpeg']
imgs = []
for fmt in fmts:
    imgs.extend(sorted(glob(osp.join(root, fmt))))

human_data = HumanData()

bbox_xywh_ = []
for img in imgs:
    h, w, _ = cv2.imread(img).shape
    bbox_xywh_.append((0, 0, w, h, 1))
    
bbox_xywh_ = np.array(bbox_xywh_)
human_data['bbox_xywh'] = bbox_xywh_

image_path_ = [osp.basename(x) for x in imgs]
human_data['image_path'] = image_path_

human_data.dump(f'data/preprocessed_datasets/{name}_dataset.npz')
