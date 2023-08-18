_base_ = ['../_base_/default_runtime.py']
# load_from = 'exp/pare_IN_lr2e-4_8gpu_stage1/epoch_140.pth'
# load_from = 'exp/pare_IN_lr5e-5_8gpu_stage1/epoch_145.pth'
# load_from = 'exp/pare_SPINmm_lr2e-4_8gpu_stage1/epoch_145.pth'
# load_from = 'exp/pare_SPIN_lr5e-5_8gpu_stage1/epoch_75.pth'
# resume_from = 'exp/pare_IN_lr5e-5_8gpu_stage2/epoch_10.pth'
resume_from = 'exp/pare_SPINmm_lr5e-5_8gpu_stage2_run2/epoch_3.pth'
use_adversarial_train = True

# evaluate
evaluation = dict(metric=['pa-mpjpe', 'mpjpe'], eval_saved_results=False)

img_res = 224
loss_weight = 60.0
workflow = [('train', 1)]

body_model = dict(
    type='SMPL',
    keypoint_src='smpl_54',
    keypoint_dst='smpl_49',
    keypoint_approximate=True,
    model_path='data/body_models/smpl',
    extra_joints_regressor='data/body_models/J_regressor_extra.npy')

# optimizer
optimizer = dict(
    backbone=dict(type='Adam', lr=5e-5), head=dict(type='Adam', lr=5e-5))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', gamma=0.1, step=[3])
runner = dict(type='EpochBasedRunner', max_epochs=20)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='')
        dict(type='TensorboardLoggerHook', interval=10)
    ])

hparams = dict(
        dict(
            DATASET=dict(
                DATASETS_AND_RATIOS='h36m_mpii_lspet_coco_mpi-inf-3dhp_0.35_0.05_0.05_0.2_0.35',
                FOCAL_LENGTH=5000.0, 
                IMG_RES=img_res,
                eval_visible_joints=False),
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
            ),
        )
    )
# model settings
model = dict(
    type='ImageBodyModelEstimator',
    backbone=dict(
        type='ResNet',
        depth=50,
        out_indices=[3],
        norm_eval=False,
        # norm_cfg=dict(type='BN', requires_grad=True)),
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    head=dict(
        type='PareHead',
        num_joints=24, 
        num_input_features=2048,
        use_heatmaps='part_segm',
        smpl_mean_params='data/body_models/smpl_mean_params.npz'),
    body_model_train=body_model,
    body_model_test=dict(
        type='SMPL',
        keypoint_src='h36m',
        keypoint_dst='h36m',
        model_path='data/body_models/smpl',
        joints_regressor='data/body_models/J_regressor_h36m.npy'),
    registrant=None,
    convention='smpl_49',
    # loss_keypoints3d=dict(type='SmoothL1Loss', loss_weight=100),
    # loss_keypoints2d=dict(type='SmoothL1Loss', loss_weight=10),
    loss_keypoints3d=dict(type='MSELoss', loss_weight=5*loss_weight),
    loss_keypoints2d=dict(type='MSELoss', loss_weight=5*loss_weight),
    loss_vertex=dict(type='L1Loss', loss_weight=0),
    loss_smpl_pose=dict(type='MSELoss', loss_weight=1.0*loss_weight),
    loss_smpl_betas=dict(type='MSELoss', loss_weight=0.001*loss_weight),
    loss_camera=dict(type='CameraPriorLoss', loss_weight=60.0),
    loss_segm_mask=dict(type='CrossEntropyLoss', loss_weight=1.0*loss_weight),
    # loss_part_segm=dict(type='MSELoss', loss_weight=1.0),
    # init_cfg=dict(
    #     type='Pretrained', checkpoint='data/pretrained/spin_pretrain.pth'),
        # type='Pretrained', checkpoint='data/pretrained/pare_official.pth'),
        
    hparams=hparams)
# dataset settings
dataset_type = 'HumanImageDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data_keys = [
    'has_smpl', 'smpl_body_pose', 'smpl_global_orient', 'smpl_betas',
    'smpl_transl', 'keypoints2d', 'keypoints3d', 'is_flipped', 'center',
    'scale', 'rotation', 'sample_idx'
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
    samples_per_gpu=32,
    workers_per_gpu=1,
    train=dict(
        type='MixedDataset',
        configs=[
            dict(
                type=dataset_type,
                dataset_name='h36m',
                data_prefix='data',
                pipeline=train_pipeline,
                convention='smpl_49',
                ann_file='spin_h36m_train_mosh.npz'), # 
            dict(
                type=dataset_type,
                dataset_name='mpi_inf_3dhp',
                data_prefix='data',
                pipeline=train_pipeline,
                convention='smpl_49',
                ann_file='spin_mpi_inf_3dhp_train_new_correct.npz'),
            # dict(
            #     type=dataset_type,
            #     dataset_name='lsp',
            #     data_prefix='data',
            #     pipeline=train_pipeline,
            #     convention='smpl_49',
            #     ann_file='spin_lsp_train.npz'),
            dict(
                type=dataset_type,
                dataset_name='lspet',
                data_prefix='data',
                pipeline=train_pipeline,
                convention='smpl_49',
                ann_file='eft_lspet.npz'), # size = 6829
            dict(
                type=dataset_type,
                dataset_name='mpii',
                data_prefix='data',
                pipeline=train_pipeline,
                convention='smpl_49',
                ann_file='eft_mpii.npz'), # size = 14667
            dict(
                type=dataset_type,
                dataset_name='coco',
                data_prefix='data',
                pipeline=train_pipeline,
                convention='smpl_49',
                ann_file='eft_coco_all.npz'), # size = 74834
        ],
        partition=[0.5, 0.20, 0.021, 0.046, 0.233],
        num_data=100000,
    ),
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
        body_model=body_model,
        dataset_name='h36m',
        # dataset_name='mpi_inf_3dhp',
        data_prefix='data',
        pipeline=test_pipeline,
        convention='h36m',
        ann_file='h36m_valid_protocol2.npz',
        # ann_file='spin_h36m_train_reproduce.npz'
        # ann_file='spin_h36m_train_mosh.npz'
        # ann_file='spin_mpi_inf_3dhp_train_new_correct.npz'
        hparams=hparams['DATASET'], # spin_pw3d_test.npz
        ),
    # test=dict(
    #     type=dataset_type,
    #     body_model=dict(
    #         type='GenderedSMPL',
    #         keypoint_src='h36m',
    #         keypoint_dst='h36m',
    #         model_path='data/body_models/smpl',
    #         joints_regressor='data/body_models/J_regressor_h36m.npy'),
    #     dataset_name='pw3d',
    #     data_prefix='data',
    #     convention='h36m',
    #     pipeline=test_pipeline,
    #     ann_file='pw3d_test_w_kp2d_ds30_op.npz', #, pw3d_test_w_kp2d_ds30_op_occ80_pare_r50.npz 'spin_pw3d_test.npz'
    #     hparams=hparams['DATASET']),
    # test=dict(
    #     type=dataset_type,
    #     body_model=dict(
    #         type='SMPL',
    #         keypoint_src='h36m',
    #         keypoint_dst='h36m',
    #         model_path='data/body_models/smpl',
    #         joints_regressor='data/body_models/J_regressor_h36m.npy'),
    #     dataset_name='3doh50k',
    #     convention='h36m',
    #     data_prefix='data',
    #     pipeline=test_pipeline,
    #     ann_file='3doh_test_w_kp2d.npz', 
    #     # pw3d_test_w_kp2d_ds30_op.npz
    #     # pw3d_test_w_kp2d_ds30_op_occ80_pare_r50.npz
    #     # pw3d_test_w_kp2d_ds30_op_occ40_pare_r50.npz
    #     hparams=hparams['DATASET']),
)
