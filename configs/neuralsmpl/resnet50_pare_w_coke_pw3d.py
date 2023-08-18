_base_ = ['../_base_/default_runtime.py']
load_from = 'data/pretrained/res50_coco_384x288-e6f795e9_20200709.pth'
find_unused_parameters = True
use_adversarial_train = True

checkpoint_config = dict(interval=5)

# evaluate
evaluation = dict(metric=['pa-mpjpe', 'mpjpe'])

# img_res = 224
img_res = 320
d_feature = 128
n_orient = 3
n_vert = 858
num_noise = 10
max_group = 216
num_neg = num_noise * max_group

loss_weight = 60.0
workflow = [('train', 1)]

body_model = dict(
    type='SMPL',
    keypoint_src='smpl_54',
    keypoint_dst='smpl_49',
    keypoint_approximate=True,
    model_path='data/body_models/smpl',
    extra_joints_regressor='data/body_models/J_regressor_extra.npy')

registrant = dict(
    type='NeuralSMPLFitting',
    body_model=dict(
        type='SMPL',
        keypoint_src='smpl_54',
        keypoint_dst='smpl_49',
        keypoint_approximate=True,
        model_path='data/body_models/smpl',
        extra_joints_regressor='data/body_models/J_regressor_extra.npy'),
    num_epochs=1,
    stages=[
        dict(
            num_iter=50,
            fit_global_orient=True,
            fit_transl=True,
            fit_body_pose=False,
            fit_betas=False),
        dict(
            num_iter=50,
            fit_global_orient=True,
            fit_transl=False,
            fit_body_pose=True,
            fit_betas=True)
    ],
    optimizer=dict(type='Adam', lr=0.01, betas=(0.9, 0.999)),
    keypoints2d_loss=dict(
        type='KeypointMSELoss', loss_weight=1.0, reduction='sum', sigma=100),
    shape_prior_loss=dict(
        type='ShapePriorLoss', loss_weight=25.0, reduction='sum'),
    joint_prior_loss=dict(
        type='JointPriorLoss', loss_weight=231.04, reduction='sum'),
    pose_prior_loss=dict(
        type='MaxMixturePrior',
        prior_folder='data',
        num_gaussians=8,
        loss_weight=22.8484,
        reduction='sum'),
    ignore_keypoints=[
        'neck_openpose', 'right_hip_openpose', 'left_hip_openpose',
        'right_hip_extra', 'left_hip_extra'
    ],
    segm_mask_loss=dict(
        type='CrossEntropyLoss',
        loss_weight=1.0,
    ),
    camera=dict(
        type='PerspectiveCameras',
        convention='opencv',
        in_ndc=False,
        focal_length=5000,
        image_size=(224, 224),
        principal_point=(112.0, 112.0)),
    hparams=dict(
                SIGMA=1e-3,
                GAMMA=1e-2,
                FACES_PER_PIXEL=40,
                RENDER_RES=img_res//4,
                n_orient=n_orient,
                theta=10,
                FOCAL_LENGTH=5000.0, 
                IMG_RES=img_res,
                disable_occ=True,
                ),
    )

# optimizer
optimizer = dict(
    backbone=dict(type='Adam', lr=1e-4), head=dict(type='Adam', lr=1e-4))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', gamma=0.1, step=[400])
runner = dict(type='EpochBasedRunner', max_epochs=300)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='')
        dict(type='TensorboardLoggerHook', interval=10)
    ])

# model settings
model = dict(
    type='ImageBodyModelEstimatorSE',
    backbone=dict(
        type='ResNet',
        depth=50,
        out_indices=[0,1,2,3],
        norm_eval=False,
        # norm_cfg=dict(type='BN', requires_grad=True),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
        # init_cfg=dict(type='Pretrained', checkpoint='data/pretrained/spin_official.pth')),
    head=dict(
        type='PareHeadwCoKe',
        num_joints=24, 
        num_input_features=2048,
        use_heatmaps='part_segm',
        smpl_mean_params='data/body_models/smpl_mean_params.npz',
        coke_cfg=dict(
        type='CoKeHead',
        d_coke_feat=d_feature, ),
        ),
    body_model_train=body_model,
    body_model_test=dict(
        type='SMPL',
        keypoint_src='h36m',
        keypoint_dst='h36m',
        model_path='data/body_models/smpl',
        joints_regressor='data/body_models/J_regressor_h36m.npy'),
    registrant=None, #registrant,
    convention='smpl_49',
    feature_bank=dict(
        type='Nearest3DMemoryManager',
        inputSize=d_feature,
        outputSize=n_vert*n_orient+num_neg,
        K=1,
        num_noise=num_noise,
        num_pos=n_vert,
        num_orient=n_orient,
        momentum=0.9,
        feat_normalization=True,
    ),
    loss_contrastive=dict(type='CoKeLoss', 
                            loss_contrastive_weight=1.0,
                            loss_noise_reg_weight=1.0,
                            n_noise_points=num_noise,
                            num_neg=num_neg,
                            weight_noise=5e-3,
                            n_orient=n_orient,
                            feat_normalization=True,
                            local_size=1,
                            T=0.07,),
    # loss_keypoints3d=dict(type='SmoothL1Loss', loss_weight=100),
    # loss_keypoints2d=dict(type='SmoothL1Loss', loss_weight=10),
    loss_keypoints3d=dict(type='MSELoss', loss_weight=5*loss_weight),
    loss_keypoints2d=dict(type='MSELoss', loss_weight=5*loss_weight),
    # loss_vertex=dict(type='L1Loss', loss_weight=0),
    loss_smpl_pose=dict(type='MSELoss', loss_weight=1.0*loss_weight),
    loss_smpl_betas=dict(type='MSELoss', loss_weight=0.001*loss_weight),
    loss_camera=dict(type='CameraPriorLoss', loss_weight=60.0),
    loss_segm_mask=dict(type='CrossEntropyLoss', loss_weight=1.0*loss_weight),
    # loss_part_segm=dict(type='MSELoss', loss_weight=1.0),
    # init_cfg=dict(
        # type='Pretrained', checkpoint='data/pretrained/resnet50_spin_pw3d-e1857270_20211201.pth'),
        # type='Pretrained', checkpoint='data/pretrained/spin_pretrain.pth'),
        # type='Pretrained', checkpoint='data/pretrained/pare_official.pth'),
    hparams=dict(
        dict(
            DATASET=dict(
                DATASETS_AND_RATIOS='h36m_mpii_lspet_coco_mpi-inf-3dhp_0.35_0.05_0.05_0.2_0.35',
                FOCAL_LENGTH=5000.0, 
                IMG_RES=img_res),
            MODEL=dict(
                RENDERER_PRED=dict(
                    SIGMA=1e-3,
                    GAMMA=1e-2,
                    FACES_PER_PIXEL=40,
                    RENDER_RES=img_res//4),
                RENDERER_GT=dict(
                    SIGMA=0,
                    GAMMA=1e-2,
                    FACES_PER_PIXEL=1,
                    RENDER_RES=img_res//4),
                # LOSS_WEIGHT=60.0,
                N_ORIENT=n_orient, 
                d_feature = 128,
                n_orient = 3,
                n_vert = 858,
                num_noise = 10,
                max_group = 216,
                num_neg = num_noise * max_group,
            ),
            DEBUG=dict(
                # save_dir='exp/3dnbf/3dpw_test_ds30_occ80stride10_pare_r50_multihypo_nopartseg_disableocc_blendbg_fixblackbg_noiseinbb_kpdrop_trainimg_gt/activation_vis/epoch_5',
                # save_dir='vmf_soft_nc15_2x',
                save_dir='gmm_nc15_km++_2x_mixture9',
                save_dir_vis='gmm_nc15_km++_2x_cmp_vis',
                # '3dnbf_resnet_bgrand_rmblk_occ80str10_dfeat64_0_epoch100.png'
                eval_occ_seg=False,
                DEBUG_LOG_DIR='debug_log',
            ),
            renderer_type='base',
            mesh_sample_param_path='data/sample_params/uniform/sample_data_8-2021-04-05.npz',
            cumulative_iters=2,
            disable_inference_loss=False,
        )
    ))
# dataset settings
dataset_type = 'HumanImageDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data_keys = [
    'has_smpl', 'smpl_body_pose', 'smpl_global_orient', 'smpl_betas',
    'smpl_transl', 'keypoints2d', 'keypoints3d', 'is_flipped', 'center',
    'scale', 'rotation', 'sample_idx', 'has_kp3d'
]
meta_data_keys = ['dataset_name', 'image_path']
train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='RandomChannelNoise', noise_factor=0.0),
    # dict(type='RandomHorizontalFlip', flip_prob=0.0, convention='smpl_49'),
    # dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0.0),
    # dict(type='ExtremeCrop', crop_prob=0.3),
    dict(type='RandomChannelNoise', noise_factor=0.4),
    dict(type='RandomHorizontalFlip', flip_prob=0.5, convention='smpl_49'),
    dict(type='GetRandomScaleRotation', rot_factor=30, scale_factor=0.25),
    dict(type='SampleBGTexture', img_res=img_res, bg_root='data/dtd/train'),
    dict(type='MeshAffine', img_res=img_res),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect', keys=['img', *data_keys], meta_keys=meta_data_keys)
]
data_keys.remove('is_flipped')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='MeshAffine', img_res=img_res),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect', keys=['img', *data_keys], meta_keys=meta_data_keys)
]

inference_pipeline = [
    dict(type='MeshAffine', img_res=img_res),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(
        type='Collect',
        keys=['img', 'sample_idx'],
        meta_keys=['image_path', 'center', 'scale', 'rotation'])
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    train=dict(
                type=dataset_type,
                dataset_name='coco',
                data_prefix='data',
                pipeline=train_pipeline,
                convention='smpl_49',
                ann_file='eft_coco_all.npz'),
    # test=dict(
    #     type=dataset_type,
    #     body_model=body_model,
    #     # dataset_name='h36m',
    #     dataset_name='mpi_inf_3dhp',
    #     data_prefix='data',
    #     pipeline=test_pipeline,
    #     convention='h36m',
    #     # ann_file='spin_h36m_train_reproduce.npz'
    #     # ann_file='spin_h36m_train_mosh.npz'
    #     ann_file='spin_mpi_inf_3dhp_train_new_correct.npz'
    #     ),
    val=dict(
        type=dataset_type,
        body_model=dict(
            type='GenderedSMPL',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='data/body_models/smpl',
            joints_regressor='data/body_models/J_regressor_h36m.npy'),
        dataset_name='pw3d',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file='spin_pw3d_val.npz'),
    test=dict(
        type=dataset_type,
        body_model=dict(
            type='GenderedSMPL',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='data/body_models/smpl',
            joints_regressor='data/body_models/J_regressor_h36m.npy'),
        dataset_name='pw3d',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file='spin_pw3d_test.npz'),
)
