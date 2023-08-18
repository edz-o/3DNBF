import sys
import os
from collections import OrderedDict
import numpy as np
import json

FOCAL_LENGTH = 5000.
IMG_RES = 224
NUM_SMPL_VERT = 6890
# Mean and standard deviation for normalizing input image
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]

"""
We create a superset of joints containing the OpenPose joints together with the ones that each dataset provides.
We keep a superset of 24 joints such that we include all joints from every dataset.
If a dataset doesn't provide annotations for a specific joint, we simply ignore it.
The joints used here are the following:
"""
JOINT_NAMES = [
# 25 OpenPose joints (in the order provided by OpenPose)
'OP Nose', # 0
'OP Neck', # 0
'OP RShoulder', # 0
'OP RElbow', # 6
'OP RWrist', # 5
'OP LShoulder', # 0
'OP LElbow', # 2
'OP LWrist', # 1
'OP MidHip', # 0
'OP RHip', # 0
'OP RKnee', # 7
'OP RAnkle', # 8
'OP LHip', # 0
'OP LKnee', # 3
'OP LAnkle', # 4
'OP REye', # 0
'OP LEye', # 0
'OP REar', # 0
'OP LEar', # 0
'OP LBigToe', # 4
'OP LSmallToe', # 4
'OP LHeel', # 4
'OP RBigToe', # 8
'OP RSmallToe', # 8
'OP RHeel', # 8
# 24 Ground Truth joints (superset of joints from different datasets)
'Right Ankle',
'Right Knee',
'Right Hip',
'Left Hip',
'Left Knee',
'Left Ankle',
'Right Wrist',
'Right Elbow',
'Right Shoulder',
'Left Shoulder',
'Left Elbow',
'Left Wrist',
'Neck (LSP)',
'Top of Head (LSP)',
'Pelvis (MPII)',
'Thorax (MPII)',
'Spine (H36M)',
'Jaw (H36M)',
'Head (H36M)',
'Nose',
'Left Eye',
'Right Eye',
'Left Ear',
'Right Ear'
]

# Dict containing the joints in numerical order
JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}

# Map joints to SMPL joints
JOINT_MAP = {
'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
'Spine (H36M)': 51, 'Jaw (H36M)': 52,
'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}

LSP_JOINTS_JOINT_REGRESSOR_H36M = [    
                                        'Left hip',
                                        'Left knee',
                                        'Left ankle',
                                        'Right ankle',
                                        'Right knee',
                                        'Right hip',
                                        'Right wrist',
                                        'Right elbow',
                                        'Right shoulder',
                                        'Left shoulder',
                                        'Left elbow',
                                        'Left wrist',
                                        'Neck',
                                        'Head top']

LSP_JOINT_MAP = {
    'Right ankle' : 0,
    'Right knee' : 1,
    'Right hip' : 2,
    'Left hip' : 3,
    'Left knee' : 4,
    'Left ankle' : 5,
    'Right wrist' : 6,
    'Right elbow' : 7,
    'Right shoulder' : 8,
    'Left shoulder' : 9,
    'Left elbow' : 10,
    'Left wrist' : 11,
    'Neck' : 12,
    'Head top' : 13,
}

LSP_TO_OPENPOSE = OrderedDict({
    0 : 11,
    1 : 10,
    2 : 9,
    3 : 12,
    4 : 13,
    5 : 14,
    6 : 4,
    7 : 3,
    8 : 2,
    9 : 5,
    10 : 6,
    11 : 7
})

HUMAN36M_RAW_JOINTS = [
    'Hips',
    'RightUpLeg', 
    'RightLeg',
    'RightFoot',
    'RightToeBase',
    'Site',
    'LeftUpLeg',
    'LeftLeg',
    'LeftFoot',
    'LeftToeBase',
    'Site',
    'Spine',
    'Spine1', # 12
    'Neck', 
    'Head',
    'Site',
    'LeftShoulder',
    'LeftArm',
    'LeftForeArm',
    'LeftHand',
    'LeftHandThumb', # 20
    'Site', 
    'L_Wrist_End',
    'Site',
    'RightShoulder',
    'RightArm',
    'RightForeArm',
    'RightHand', # 27
    'RightHandThumb',
    'Site',
    'R_Wrist_End',
    'Site'
]

# Showing original id, real id should be 0-16
# According to website 
HUMAN36M_JOINT_MAP = {
    'mid_hip' : 0,
    'right_hip' : 1,
    'right_knee' : 2,
    'right_foot' : 3,
    'left_hip' : 6,
    'left_knee' : 7,
    'left_foot' : 8,
    'spine' : 12,
    'thorax' : 13,
    'neck' : 14,
    'head' : 15,
    'left_shoulder' : 17,
    'left_elbow' : 18,
    'left_wrist' : 10,
    'right_shoulder' : 25,
    'right_elbow' : 26,
    'right_wrist' : 27 
}
# HUMAN36M_JOINT_MAP = {
#     'mid_hip' : 0,
#     'left_hip' : 1,
#     'left_knee' : 2,
#     'left_foot' : 3,
#     'right_hip' : 6,
#     'right_knee' : 7,
#     'right_foot' : 8,
#     'spine' : 12,
#     'thorax' : 13,
#     'neck' : 14,
#     'head' : 15,
#     'left_shoulder' : 17,
#     'left_elbow' : 18,
#     'left_wrist' : 10,
#     'right_shoulder' : 25,
#     'right_elbow' : 26,
#     'right_wrist' : 27 
# }

SMPL_JOINTS = [
    'pelvis', # 0
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee', # 5
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot', # 10
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow', #18
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hand',
    'right_hand' # 23
]

openpose_links = {
    'torso' : [8, 1], # update from [8, 0] to [8, 1]
    'left_lower_arm' : [6, 7], 
    'left_upper_arm' : [5, 6], 
    'left_thigh' : [12, 13], 
    'left_calf' : [13, 14],
    'right_lower_arm' : [3, 4], 
    'right_upper_arm' : [2, 3], 
    'right_thigh' : [9, 10], 
    'right_calf' : [10, 11]
}
# Joint selectors
# Indices to get the 14 LSP joints from the 17 H36M joints
# H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9] # should be 1,2,3,6,5,4?
H36M_TO_J17 = [1, 2, 3, 6, 5, 4, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9] # should be 1,2,3,6,5,4?
H36M_TO_J14 = H36M_TO_J17[:14]
SMPL_TO_J17 = [6605, 4504, 4303,  807, 1023, 3182, 5382, 5090, 5311, 821, 1740, 1961, 1312, 81, 4423, 3505, 3808]
SMPL_TO_J17_SPARSE = [821, 621, 492, 60, 121, 380, 734, 712, 725, 187, 252, 292, 62, 2, 494, 795, 450]
# Indices to get the 14 LSP joints from the ground truth joints
J24_TO_J17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14, 16, 17]
J24_TO_J14 = J24_TO_J17[:14]

# Permutation of SMPL pose parameters when flipping the shape
SMPL_JOINTS_FLIP_PERM = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
SMPL_POSE_FLIP_PERM = []
for i in SMPL_JOINTS_FLIP_PERM:
    SMPL_POSE_FLIP_PERM.append(3*i)
    SMPL_POSE_FLIP_PERM.append(3*i+1)
    SMPL_POSE_FLIP_PERM.append(3*i+2)
# Permutation indices for the 24 ground truth joints
J24_FLIP_PERM = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18, 19, 21, 20, 23, 22]
# Permutation indices for the full set of 49 joints
J49_FLIP_PERM = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]\
              + [25+i for i in J24_FLIP_PERM]
# Permutation indices for the OpenPose joints
OPENPOSE_FLIP_PERM = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]


nemo_part_list = [
    'torso', 
    'left_lower_arm', 
    'left_upper_arm', 
    'left_thigh', 
    'left_calf', 
    'right_lower_arm', 
    'right_upper_arm', 
    'right_thigh', 
    'right_calf'
]
nemo_part_list_flip_permute = [0, 5, 6, 7, 8, 1, 2, 3, 4]


# LSP dataset has six parts
lsp_part_list = ['Torso', 'Upper Leg', 'Lower Leg', 'Upper Arm', 'Forearm', 'Head']
merged_parts_smpl = {
    'Torso': ['leftShoulder', 'rightShoulder', 'spine', 'spine1', 'spine2', 'hips'],
    'Upper Leg': ['rightUpLeg', 'leftUpLeg'],
    'Lower Leg': ['leftLeg', 'leftFoot', 'leftToeBase', 'rightLeg', 'rightFoot', 'rightToeBase'],
    'Upper Arm': ['leftArm', 'rightArm'],
    'Forearm': ['leftForeArm', 'leftHand', 'leftHandIndex1', 'rightForeArm', 'rightHand', 'rightHandIndex1'],
    'Head': ['head', 'neck'],
    }


# smplpartname_to_lspid = {}
# for p in smpl_vert_segmentation.keys():
#     for k, v in merged_parts_smpl.items():
#         if p in v:
#             smplpartname_to_lspid[p] = lsp_part_list.index(k)


mesh_downsample_method = 'uniform'
mesh_downsample_ratio = 8
mesh_downsample_info_postfix = '2021-04-05'
mesh_sample_data = np.load(
        os.path.expanduser(
            os.path.join(
                'data', 'sample_params', mesh_downsample_method,
                'sample_data_{}-{}.npz'.format(
                    mesh_downsample_ratio,
                    mesh_downsample_info_postfix))))
ds_indices = mesh_sample_data['indices'] 
# vertex_to_part_lsp = []
# for i in range(len(ds_indices)):
#     vertex_to_part_lsp.append(smplpartname_to_lspid[smpl_vert_segmentation_inverse[ds_indices[i]]])

with open('data/vertex_to_part.json', 'r') as f:
    vertex_to_part = json.load(f)
# with open('data/vertex_to_part_voge.json', 'r') as f:
#     vertex_to_part_voge = json.load(f)
# with open('vertex_to_part_voge_kernel.json', 'r') as f:
#     vertex_to_part_voge_kernels = json.load(f)
# with open('vertex_to_part_voge_joints.json', 'r') as f:
#     vertex_to_part_voge_joints = json.load(f)
# with open('vertex_to_part_voge_Kkernel_s.json', 'r') as f:
#     vertex_to_part_voge_Kkernels_s = json.load(f)
# with open('vertex_to_part_voge_Kkernel_e.json', 'r') as f:
#     vertex_to_part_voge_Kkernels_e = json.load(f)
# with open('vertex_to_part_voge_Kkernel_es.json', 'r') as f:
#     vertex_to_part_voge_Kkernels_es = json.load(f)