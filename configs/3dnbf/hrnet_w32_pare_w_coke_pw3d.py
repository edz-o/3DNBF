_base_ = ['../_base_/default_runtime.py']
load_from = 'data/pretrained/pose_hrnet_w32_384x288_mm.pth'

use_adversarial_train = True
find_unused_parameters = True  # Debug
checkpoint_config = dict(interval=5)

# evaluate
evaluation = dict(metric=['pa-mpjpe', 'mpjpe'], eval_saved_results=False)  #

img_res = 320
downsample_rate = 4
d_feature = 128
n_orient = 3
n_vert = 858
num_noise = 10
max_group = 216
num_neg = num_noise * max_group

loss_weight = 60.0
workflow = [('train', 1)]
phase = 'train'  #'test'

hparams = dict(
    DATASET=dict(
        DATASETS_AND_RATIOS='h36m_mpii_lspet_coco_mpi-inf-3dhp_0.35_0.05_0.05_0.2_0.35',
        FOCAL_LENGTH=5000.0,
        IMG_RES=img_res,
        eval_visible_joints=True,
    ),
    MODEL=dict(
        RENDERER_PRED=dict(
            SIGMA=1e-3, GAMMA=1e-2, FACES_PER_PIXEL=40, RENDER_RES=img_res // 4
        ),
        RENDERER_GT=dict(
            SIGMA=0, GAMMA=1e-2, FACES_PER_PIXEL=1, RENDER_RES=img_res // 4
        ),
        N_ORIENT=n_orient,
        downsample_rate=downsample_rate,
        VOGE_SAMPLE=True,
    ),
    REGISTRANT=dict(
        SIGMA=1e-3,
        GAMMA=1e-2,
        FACES_PER_PIXEL=10,
        RENDER_RES=img_res // downsample_rate,
        downsample_rate=downsample_rate,
        n_orient=n_orient,
        n_vert=n_vert,
        theta=10,
        FOCAL_LENGTH=5000.0,
        IMG_RES=img_res,
        disable_occ=True,  # , #True, #False,
        show_debug_info=False,
        thr_detected_joints=7,
        thr_openpose_conf=0.2,
        thr_vert_det=0.1,
        use_other_init=False,
        use_saved_coke=False,
        use_saved_partseg=False,
        use_otf_openpose=True,  # True for 3dpw-AdvOcc
        use_gt_smpl=False,
        optimize_twoside=True,
        debug_feature_map=False,
        fit_z=False,  # False,
        noise_100=False,
        noise_90=False,
        RUN_VISUALIZER=False,
    ),
    VISUALIZER=dict(
        IMG_RES=512,
        DEBUG_LOG_DIR='debug_log',),
    mesh_sample_param_path='data/sample_params/uniform/sample_data_8-2021-04-05.npz',
    renderer_type='VoGE',
    disable_inference_loss=False,
    cumulative_iters=2,
)

body_model = dict(
    type='SMPL',
    keypoint_src='smpl_54',
    keypoint_dst='smpl_49',
    keypoint_approximate=True,
    model_path='data/body_models/smpl',
    extra_joints_regressor='data/body_models/J_regressor_extra.npy',
)

registrant = dict(
    type='NeuralSMPLFittingVoGE',
    body_model=dict(
        type='SMPL',
        keypoint_src='smpl_54',
        keypoint_dst='smpl_49',
        keypoint_approximate=True,
        model_path='data/body_models/smpl',
        extra_joints_regressor='data/body_models/J_regressor_extra.npy',
    ),
    num_epochs=0 if hparams['REGISTRANT']['use_gt_smpl'] else 1,
    img_res=img_res,
    stages=[
        dict(
            num_iter=40,
            fit_global_orient=True,
            fit_transl=True,
            fit_body_pose=False,
            fit_betas=False,
            likelihood_weight=1.0,
        ),
        dict(
            num_iter=80,
            fit_global_orient=True,
            fit_transl=False,
            fit_body_pose=True,
            fit_betas=True,
        ),
    ],
    optimizer=dict(type='Adam', lr=0.02, betas=(0.9, 0.99)),
    keypoints2d_loss=dict(
        type='KeypointMSELoss', loss_weight=1e-2, reduction='sum', sigma=100
    ),
    vertices2d_loss=dict(
        type='KeypointMSELoss', loss_weight=1e-3, reduction='sum', sigma=100
    ),
    # shape_prior_loss=dict(
    #     type='ShapePriorLoss', loss_weight=1e-1, reduction='sum'),
    joint_prior_loss=dict(type='JointPriorLoss', loss_weight=1e-1, reduction='sum'),
    # pose_prior_loss=dict(
    #     type='MaxMixturePrior',
    #     prior_folder='data',
    #     num_gaussians=8,
    #     loss_weight=1e-1,
    #     reduction='sum'),
    pose_prior_loss=dict(
        type='VAEPosePrior', prior_folder='data', loss_weight=1 * 5e-4, reduction='sum'
    ),
    likelihood_loss=dict(
        type='LikelihoodLoss',
        loss_weight=1.0,
        reduction='sum',
    ),
    segm_mask_loss=dict(type='CrossEntropyLoss', loss_weight=0 * 1.0, reduction='sum'),
    ignore_keypoints=[],
    camera=dict(
        type='PerspectiveCameras',
        convention='opencv',
        in_ndc=False,
        focal_length=5000,
        image_size=(img_res, img_res),
        principal_point=(112.0, 112.0),
    ),
    hparams=hparams['REGISTRANT'],
)

# optimizer
optimizer = dict(backbone=dict(type='Adam', lr=2e-4), head=dict(type='Adam', lr=2e-4))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', gamma=0.1, step=[400])
runner = dict(type='EpochBasedRunner', max_epochs=300)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='')
        dict(type='TensorboardLoggerHook', interval=10),
    ],
)

# model settings
width = 32
downsample = False
use_conv = True
hrnet_extra = dict(
    stage1=dict(
        num_modules=1,
        num_branches=1,
        block='BOTTLENECK',
        num_blocks=(4,),
        num_channels=(64,),
    ),
    stage2=dict(
        num_modules=1,
        num_branches=2,
        block='BASIC',
        num_blocks=(4, 4),
        num_channels=(width, width * 2),
    ),
    stage3=dict(
        num_modules=4,
        num_branches=3,
        block='BASIC',
        num_blocks=(4, 4, 4),
        num_channels=(width, width * 2, width * 4),
    ),
    stage4=dict(
        num_modules=3,
        num_branches=4,
        block='BASIC',
        num_blocks=(4, 4, 4, 4),
        num_channels=(width, width * 2, width * 4, width * 8),
    ),
    downsample=downsample,
    use_conv=use_conv,
    pretrained_layers=[
        'conv1',
        'bn1',
        'conv2',
        'bn2',
        'layer1',
        'transition1',
        'stage2',
        'transition2',
        'stage3',
        'transition3',
        'stage4',
    ],
    final_conv_kernel=1,
    return_list=False,
)

model = dict(
    type='ImageVoGEBodyModelEstimatorSE',
    backbone=dict(
        type='PoseHighResolutionNet',
        extra=hrnet_extra,
        num_joints=24,
        init_cfg=dict(
            type='Pretrained', checkpoint='data/pretrained/hrnet_pretrain_cpu.pth'
        ),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
    head=dict(
        type='PareHeadwCoKe',
        num_joints=24,
        num_input_features=480,
        use_heatmaps='part_segm',
        smpl_mean_params='data/body_models/smpl_mean_params.npz',
        coke_cfg=dict(
            type='CoKeHeadHRNet',
            in_channels=480,
            hidden_channels=512,
            d_coke_feat=d_feature,
        ),
    ),
    body_model_train=body_model,
    body_model_test=dict(
        type='SMPL',
        keypoint_src='h36m',
        keypoint_dst='h36m',
        model_path='data/body_models/smpl',
        joints_regressor='data/body_models/J_regressor_h36m.npy',
    ),
    registrant=None if phase == 'train' else registrant,
    convention='smpl_49',
    feature_bank=dict(
        type='Nearest3DMemoryManager',
        inputSize=d_feature,
        outputSize=n_vert * n_orient + num_neg,
        K=1,
        num_noise=num_noise,
        num_pos=n_vert,
        num_orient=n_orient,
        momentum=0.9,
        feat_normalization=True,
    ),
    loss_contrastive=dict(
        type='CoKeLoss',
        loss_contrastive_weight=1.0,
        loss_noise_reg_weight=1.0,
        n_noise_points=num_noise,
        num_neg=num_neg,
        weight_noise=5e-3,
        n_orient=n_orient,
        feat_normalization=True,
        local_size=1,
        T=0.07,
    ),
    loss_keypoints3d=dict(type='MSELoss', loss_weight=5 * loss_weight),
    loss_keypoints2d=dict(type='MSELoss', loss_weight=5 * loss_weight),
    # loss_vertex=dict(type='L1Loss', loss_weight=0),
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
    'bbox_xywh',
    'poses_init',
    'betas_init',
    'cameras_init',
    'pred_segm_mask',
    'occ_mask',
    'has_kp3d',
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
    dict(type='SampleBGTexture', img_res=img_res, bg_root='data/dtd/train'),
    dict(type='MeshAffine', img_res=img_res),
    dict(
        type='AddOcclusionPatch',
        occ_size=40,
        occ_stride=20,
        dtd_root='data/dtd/train',
        prob=0.0,
        phase='train',
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(
        type='Collect',
        keys=['img', 'have_occ', 'occ_size', 'occ_stride', 'occ_idx', *data_keys],
        meta_keys=meta_data_keys,
    ),
]
data_keys.remove('is_flipped')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='SampleBGTexture', img_res=img_res, bg_root='data/dtd/train'),
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
        type='AddOcclusionPatch', occ_size=40, occ_stride=10, dtd_root='data/dtd/test'
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect', keys=['img', *data_keys], meta_keys=meta_data_keys),
]
occimgen_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='MeshAffine', img_res=img_res),
    dict(
        type='AddOcclusionPatch', occ_size=40, occ_stride=10, dtd_root='data/dtd/test'
    ),
    dict(type='Collect', keys=['img', 'center', 'scale', 'image_path'], meta_keys=[]),
]
vis_pipeline_occ = [
    dict(type='LoadImageFromFile'),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='MeshAffine', img_res=hparams['VISUALIZER']['IMG_RES']),
    dict(
        type='AddOcclusionPatch', occ_size=40, occ_stride=10, dtd_root='data/dtd/test'
    ),
    dict(type='Collect', keys=['img', 'center', 'scale', 'image_path'], meta_keys=[]),
]
vis_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='MeshAffine', img_res=hparams['VISUALIZER']['IMG_RES']),
    dict(type='Collect', keys=['img', 'center', 'scale', 'image_path'], meta_keys=[]),
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

data = dict(
    samples_per_gpu=12 if phase == 'train' else 8,
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
    test=dict(
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
    visualization=dict(
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
        ann_file='pw3d_test_w_kp2d_ds30_op.npz',
        hparams=hparams['DATASET'],
    ),
)
