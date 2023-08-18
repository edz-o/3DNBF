_base_ = ['../_base_/default_runtime.py']
use_adversarial_train = True

checkpoint_config = dict(interval=5)

# evaluate
evaluation = dict(metric=['pa-mpjpe', 'mpjpe'], eval_saved_results=False)

img_res = 224

loss_weight = 60.0
workflow = [('train', 1)]

OCC_SIZE_TEST = 80
OCC_STRIDE_TEST = 10
OCCLUDE_CENTER = None
test_data = '3dpw'

body_model = dict(
    type='SMPL',
    keypoint_src='smpl_54',
    keypoint_dst='smpl_49',
    keypoint_approximate=True,
    model_path='data/body_models/smpl',
    extra_joints_regressor='data/body_models/J_regressor_extra.npy',
)

# optimizer
optimizer = dict(backbone=dict(type='Adam', lr=5e-5), head=dict(type='Adam', lr=5e-5))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', gamma=0.1, step=[200])
runner = dict(type='EpochBasedRunner', max_epochs=160)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='')
        dict(type='TensorboardLoggerHook', interval=10),
    ],
)

hparams = dict(
    DATASET=dict(
        DATASETS_AND_RATIOS='h36m_mpii_lspet_coco_mpi-inf-3dhp_0.35_0.05_0.05_0.2_0.35',
        FOCAL_LENGTH=5000.0,
        IMG_RES=img_res,
        eval_visible_joints=True,
        load_mask=False,
        test_indices=[],
    ),
    MODEL=dict(
        RENDERER_PRED=dict(
            SIGMA=1e-3, GAMMA=1e-2, FACES_PER_PIXEL=40, RENDER_RES=img_res // 4
        ),
        RENDERER_GT=dict(
            SIGMA=0, GAMMA=1e-2, FACES_PER_PIXEL=1, RENDER_RES=img_res // 4
        ),
    ),
    REGISTRANT=dict(
        use_otf_openpose=True,
    ),
    VISUALIZER=dict(
        IMG_RES=512,
        DEBUG_LOG_DIR='debug_logs'),
    save_partseg=False, 
    return_vertices=True,
)

registrant = dict(
    type='SMPLify',
    body_model=body_model,
    stages=[
        # stage 1
        dict(
            num_iter=20,
            fit_global_orient=True,
            fit_transl=True,
            fit_body_pose=False,
            fit_betas=False,
            joint_weights=dict(
                body_weight=5.0,
                use_shoulder_hip_only=True,
            ),
        ),
        # stage 2
        dict(
            num_iter=10,
            fit_global_orient=True,
            fit_transl=True,
            fit_body_pose=True,
            fit_betas=True,
            joint_weights=dict(body_weight=5.0, use_shoulder_hip_only=False),
        ),
    ],
    optimizer=dict(type='LBFGS', max_iter=20, lr=1e-2, line_search_fn='strong_wolfe'),
    keypoints2d_loss=dict(
        type='KeypointMSELoss', loss_weight=1.0, reduction='sum', sigma=100
    ),
    keypoints3d_loss=dict(
        type='KeypointMSELoss', loss_weight=10, reduction='sum', sigma=100
    ),
    shape_prior_loss=dict(type='ShapePriorLoss', loss_weight=1, reduction='sum'),
    joint_prior_loss=dict(
        type='JointPriorLoss',
        loss_weight=20,
        reduction='sum',
        smooth_spine=True,
        smooth_spine_loss_weight=20,
        use_full_body=True,
    ),
    smooth_loss=dict(type='SmoothJointLoss', loss_weight=0, reduction='sum'),
    pose_prior_loss=dict(
        type='MaxMixturePrior',
        prior_folder='data',
        num_gaussians=8,
        loss_weight=4.78**2,
        reduction='sum',
    ),
    ignore_keypoints=[
        'neck_openpose',
        'right_hip_openpose',
        'left_hip_openpose',
        'right_hip_extra',
        'left_hip_extra',
    ],
    camera=dict(
        type='PerspectiveCameras',
        convention='opencv',
        in_ndc=False,
        focal_length=5000,
        image_size=(224, 224),
        principal_point=(112, 112),
    ),
)

registrant_new = dict(
    type='SMPLify',
    body_model=body_model,
    num_epochs=1,
    stages=[
        # stage 1
        dict(
            num_iter=50,
            fit_global_orient=True,
            fit_transl=True,
            fit_body_pose=False,
            fit_betas=False,
        ),
        # stage 2
        dict(
            num_iter=50,
            fit_global_orient=True,
            fit_transl=False,
            fit_body_pose=True,
            fit_betas=True,
        ),
    ],
    optimizer=dict(type='Adam', lr=1e-2, betas=(0.9, 0.999)),
    keypoints2d_loss=dict(
        type='KeypointMSELoss', loss_weight=1.0, reduction='sum', sigma=100
    ),
    shape_prior_loss=dict(type='ShapePriorLoss', loss_weight=5.0**2, reduction='sum'),
    joint_prior_loss=dict(
        type='JointPriorLoss', loss_weight=15.2**2, reduction='sum'
    ),
    pose_prior_loss=dict(
        type='MaxMixturePrior',
        prior_folder='data',
        num_gaussians=8,
        loss_weight=4.78**2,
        reduction='sum',
    ),
    ignore_keypoints=[
        'neck_openpose',
        'right_hip_openpose',
        'left_hip_openpose',
        'right_hip_extra',
        'left_hip_extra',
    ],
    camera=dict(
        type='PerspectiveCameras',
        convention='opencv',
        in_ndc=False,
        focal_length=5000,
        image_size=(img_res, img_res),
        principal_point=(img_res / 2, img_res / 2),
    ),
)
# model settings
model = dict(
    type='ImageBodyModelEstimator',
    backbone=dict(
        type='ResNet',
        depth=50,
        out_indices=[3],
        norm_eval=False,
        # norm_cfg=dict(type='BN', requires_grad=True),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    # init_cfg=dict(type='Pretrained', checkpoint='data/pretrained/spin_official.pth')),
    head=dict(
        type='PareHead',
        num_joints=24,
        num_input_features=2048,
        use_heatmaps='part_segm',
        smpl_mean_params='data/body_models/smpl_mean_params.npz',
    ),
    body_model_train=body_model,
    body_model_test=dict(
        type='SMPL',
        keypoint_src='h36m',
        keypoint_dst='h36m',
        model_path='data/body_models/smpl',
        joints_regressor='data/body_models/J_regressor_h36m.npy',
    ),
    registrant=None,  # registrant_new, #registrant, #None,
    convention='smpl_49',
    loss_keypoints3d=dict(type='MSELoss', loss_weight=5 * loss_weight),
    loss_keypoints2d=dict(type='MSELoss', loss_weight=5 * loss_weight),
    loss_vertex=dict(type='L1Loss', loss_weight=0),
    loss_smpl_pose=dict(type='MSELoss', loss_weight=1.0 * loss_weight),
    loss_smpl_betas=dict(type='MSELoss', loss_weight=0.001 * loss_weight),
    loss_camera=dict(type='CameraPriorLoss', loss_weight=60.0),
    loss_segm_mask=dict(type='CrossEntropyLoss', loss_weight=1.0 * loss_weight),
    hparams=hparams,
)
# dataset settings
dataset_type = 'HumanImageDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
data_keys = [
    'has_smpl',
    'smpl_body_pose',
    'smpl_global_orient',
    'smpl_betas',
    'smpl_transl',
    'keypoints2d',
    'keypoints3d',
    'is_flipped',
    'center',
    'scale',
    'rotation',
    'sample_idx',
    'idx',
]
meta_data_keys = [
    'dataset_name',
    'img_norm_cfg',
    'image_path',
    'occ_size',
    'occ_stride',
    'occ_idx',
    'texture_file',
    'texture_crop_tl',
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomChannelNoise', noise_factor=0.4),
    dict(type='RandomHorizontalFlip', flip_prob=0.5, convention='smpl_49'),
    dict(type='GetRandomScaleRotation', rot_factor=30, scale_factor=0.25),
    dict(type='MeshAffine', img_res=img_res),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect', keys=['img', *data_keys], meta_keys=meta_data_keys),
]
data_keys.remove('is_flipped')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='MeshAffine', img_res=img_res),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect', keys=['img', *data_keys], meta_keys=meta_data_keys),
]
test_pipeline_occ = [
    dict(type='LoadImageFromFile'),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='MeshAffine', img_res=img_res),
    dict(
        type='AddOcclusionPatch',
        occ_size=OCC_SIZE_TEST,
        occ_stride=10,
        dtd_root='data/dtd/test',
        occlude_center=OCCLUDE_CENTER,
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect', keys=['img', *data_keys], meta_keys=meta_data_keys),
]
vis_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='MeshAffine', img_res=hparams['VISUALIZER']['IMG_RES']),
    dict(type='Collect', keys=['img', 'center', 'scale', 'image_path'], meta_keys=[]),
]

vis_pipeline_occ = [
    dict(type='LoadImageFromFile'),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='MeshAffine', img_res=hparams['VISUALIZER']['IMG_RES']),
    dict(
        type='AddOcclusionPatch',
        occ_size=OCC_SIZE_TEST,
        occ_stride=10,
        dtd_root='data/dtd/test',
        occlude_center=OCCLUDE_CENTER,
    ),
    dict(type='Collect', keys=['img', 'center', 'scale', 'image_path'], meta_keys=[]),
]

dataset_gen_pipeline_occ = [
    dict(type='LoadImageFromFile'),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='MeshAffine', img_res=224),  # image size
    dict(
        type='AddOcclusionPatch', occ_size=80, occ_stride=10, dtd_root='data/dtd/test'
    ),
    dict(
        type='Collect',
        keys=['img', 'center', 'scale', 'image_path', 'keypoints2d'],
        meta_keys=meta_data_keys,
        use_dc=False,
    ),
]

dataset_gen_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='MeshAffine', img_res=224),  # image size
    dict(
        type='Collect',
        keys=['img', 'center', 'scale', 'image_path', 'keypoints2d'],
        meta_keys=meta_data_keys,
        use_dc=False,
    ),
]

inference_pipeline = [
    dict(type='MeshAffine', img_res=img_res),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(
        type='Collect',
        keys=['img', 'sample_idx'],
        meta_keys=['image_path', 'center', 'scale', 'rotation'],
    ),
]

if test_data == '3dpw_advocc_grid':
    test_cfg = dict(
        type='OccludedHumanImageDataset',
        orig_cfg=dict(
            type=dataset_type,
            body_model=dict(
                type='GenderedSMPL',
                keypoint_src='h36m',
                keypoint_dst='h36m',
                model_path='data/body_models/smpl',
                joints_regressor='data/body_models/J_regressor_h36m.npy',
            ),
            dataset_name='pw3d',  #'pw3d',
            convention='smpl_49',  # h36m
            data_prefix='data',
            pipeline=test_pipeline_occ,
            ann_file='pw3d_test_w_kp2d_ds30_op.npz',
            hparams=hparams['DATASET'],
        ),
        occ_size=OCC_SIZE_TEST,
        occ_stride=OCC_STRIDE_TEST,
    )
    vis_cfg = dict(
        type='OccludedHumanImageDataset',
        orig_cfg=dict(
            type=dataset_type,
            body_model=dict(
                type='GenderedSMPL',
                keypoint_src='h36m',
                keypoint_dst='h36m',
                model_path='data/body_models/smpl',
                joints_regressor='data/body_models/J_regressor_h36m.npy',
            ),
            dataset_name='pw3d',  #'pw3d',
            convention='smpl_49',  # h36m
            data_prefix='data',
            pipeline=vis_pipeline_occ,
            ann_file='pw3d_test_w_kp2d_ds30_op.npz',
            hparams=hparams['DATASET'],
        ),
        occ_size=OCC_SIZE_TEST,
        occ_stride=OCC_STRIDE_TEST,
    )
if test_data == '3dpw_advocc':
    test_cfg = dict(
        type=dataset_type,
        body_model=dict(
            type='GenderedSMPL',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='data/body_models/smpl',
            joints_regressor='data/body_models/J_regressor_h36m.npy',
        ),
        dataset_name='pw3d',
        convention='smpl_49',
        data_prefix='data',
        pipeline=test_pipeline_occ,
        ann_file='pw3d_test_w_kp2d_ds30_op_w_pareinit.npz',
        hparams=hparams['DATASET'],
    )
    vis_cfg = dict(
        type=dataset_type,
        body_model=dict(
            type='GenderedSMPL',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='data/body_models/smpl',
            joints_regressor='data/body_models/J_regressor_h36m.npy',
        ),
        dataset_name='pw3d',
        convention='smpl_49',
        data_prefix='data',
        pipeline=vis_pipeline_occ,
        ann_file='pw3d_test_w_kp2d_ds30_op_w_pareinit.npz',
        hparams=hparams['DATASET'],
    )
elif test_data == '3dpw':
    test_cfg = dict(
        type=dataset_type,
        body_model=dict(
            type='GenderedSMPL',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='data/body_models/smpl',
            joints_regressor='data/body_models/J_regressor_h36m.npy',
        ),
        dataset_name='pw3d',
        convention='smpl_49',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file='pw3d_test_w_kp2d_ds30_op_w_pareinit.npz',
        hparams=hparams['DATASET'],
    )
    vis_cfg = dict(
        type=dataset_type,
        body_model=dict(
            type='GenderedSMPL',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='data/body_models/smpl',
            joints_regressor='data/body_models/J_regressor_h36m.npy',
        ),
        dataset_name='pw3d',
        convention='smpl_49',
        data_prefix='data',
        pipeline=vis_pipeline,
        ann_file='pw3d_test_w_kp2d_ds30_op_w_pareinit.npz',
        hparams=hparams['DATASET'],
    )
elif test_data == '3dpw_test':
    test_cfg = dict(
        type=dataset_type,
        body_model=dict(
            type='GenderedSMPL',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='data/body_models/smpl',
            joints_regressor='data/body_models/J_regressor_h36m.npy',
        ),
        dataset_name='pw3d',
        convention='smpl_49',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file='pw3d_test_w_kp2d_correct.npz',
        hparams=hparams['DATASET'],
    )
    vis_cfg = dict(
        type=dataset_type,
        body_model=dict(
            type='GenderedSMPL',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='data/body_models/smpl',
            joints_regressor='data/body_models/J_regressor_h36m.npy',
        ),
        dataset_name='pw3d',
        convention='smpl_49',
        data_prefix='data',
        pipeline=vis_pipeline,
        ann_file='pw3d_test_w_kp2d_correct.npz',
        hparams=hparams['DATASET'],
    )
elif test_data == '3dpw_occ':
    test_cfg = dict(
        type=dataset_type,
        body_model=dict(
            type='GenderedSMPL',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='data/body_models/smpl',
            joints_regressor='data/body_models/J_regressor_h36m.npy',
        ),
        dataset_name='pw3d',
        convention='smpl_49',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file='pw3d_occ_w_kp2d_ds30_correct.npz',
        hparams=hparams['DATASET'],
    )
    vis_cfg = dict(
        type=dataset_type,
        body_model=dict(
            type='GenderedSMPL',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='data/body_models/smpl',
            joints_regressor='data/body_models/J_regressor_h36m.npy',
        ),
        dataset_name='pw3d',
        convention='smpl_49',
        data_prefix='data',
        pipeline=vis_pipeline,
        ann_file='pw3d_occ_w_kp2d_ds30_correct.npz',
        hparams=hparams['DATASET'],
    )
elif test_data == 'h36m':
    test_cfg = dict(
        type=dataset_type,
        body_model=dict(
            type='GenderedSMPL',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='data/body_models/smpl',
            joints_regressor='data/body_models/J_regressor_h36m.npy',
        ),
        dataset_name='h36m',
        convention='h36m',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file='h36m_valid_protocol2.npz',
        hparams=hparams['DATASET'],
    )
elif test_data == 'mpi_inf_3dhp':
    test_cfg = dict(
        type=dataset_type,
        body_model=body_model,
        dataset_name='mpi_inf_3dhp',
        data_prefix='data',
        pipeline=test_pipeline,
        convention='h36m',
        ann_file='mpi_inf_3dhp_test_op.npz',
        hparams=hparams['DATASET'],
    )
elif test_data == '3doh50k':
    test_cfg = dict(
        type=dataset_type,
        body_model=dict(
            type='SMPL',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='data/body_models/smpl',
            joints_regressor='data/body_models/J_regressor_h36m.npy',
        ),
        dataset_name='3doh50k',
        convention='smpl_49',  #'h36m',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file='3doh_test_w_kp2d_rendered_op_w_pareinit.npz',
        hparams=hparams['DATASET'],
    )
    vis_cfg = dict(
        type=dataset_type,
        body_model=dict(
            type='SMPL',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='data/body_models/smpl',
            joints_regressor='data/body_models/J_regressor_h36m.npy',
        ),
        dataset_name='3doh50k',
        convention='smpl_49',  #'h36m',
        data_prefix='data',
        pipeline=vis_pipeline,
        ann_file='3doh_test_w_kp2d_rendered_op_w_pareinit.npz',
        hparams=hparams['DATASET'],
    )
elif test_data == '3dpw_occ_varying':
    test_cfg = dict(
        type='OcclusionVaryingHumanImageDataset',
        orig_cfg=dict(
            type=dataset_type,
            body_model=dict(
                type='GenderedSMPL',
                keypoint_src='h36m',
                keypoint_dst='h36m',
                model_path='data/body_models/smpl',
                joints_regressor='data/body_models/J_regressor_h36m.npy',
            ),
            dataset_name='pw3d',
            convention='smpl_49',
            data_prefix='data',
            pipeline=test_pipeline_occ,
            ann_file='pw3d_test_w_kp2d_ds30_op_w_pareinit.npz',
            hparams=hparams['DATASET'],
        ),
        occ_levels=list(range(0, 160, 5)),
    )
    vis_cfg = dict(
        type='OcclusionVaryingHumanImageDataset',
        orig_cfg=dict(
            type=dataset_type,
            body_model=dict(
                type='GenderedSMPL',
                keypoint_src='h36m',
                keypoint_dst='h36m',
                model_path='data/body_models/smpl',
                joints_regressor='data/body_models/J_regressor_h36m.npy',
            ),
            dataset_name='pw3d',
            convention='smpl_49',
            data_prefix='data',
            pipeline=vis_pipeline_occ,
            ann_file='pw3d_test_w_kp2d_ds30_op_w_pareinit.npz',
            hparams=hparams['DATASET'],
        ),
        occ_levels=list(range(0, 160, 5)),
    )

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        dataset_name='coco',
        data_prefix='data',
        pipeline=train_pipeline,
        convention='smpl_49',
        ann_file='eft_coco_all.npz',
    ),
    val=dict(
        type=dataset_type,
        body_model=dict(
            type='GenderedSMPL',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='data/body_models/smpl',
            joints_regressor='data/body_models/J_regressor_h36m.npy',
        ),
        dataset_name='pw3d',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file='spin_pw3d_val.npz',
    ),
    test=test_cfg,
    visualization=vis_cfg,
)
