_base_ = ['../_base_/default_runtime.py']
# load_from = 'exp/neuralsmpl_pare/epoch_130.pth'
# load_from = 'exp/neuralsmpl_pare_spininit_eftcoco/epoch_45.pth'
# resume_from = 'exp/neuralsmpl_pare_stage2/epoch_1.pth'
load_from = 'exp/neuralsmpl_pare_kp3d_accu_2dinit/epoch_70.pth'
use_adversarial_train = True
find_unused_parameters = True
checkpoint_config = dict(interval=1)

# evaluate
evaluation = dict(metric=['pa-mpjpe', 'mpjpe'], eval_saved_results=False)  #

# img_res = 224
img_res = 320
downsample_rate = 4
d_feature = 128
n_orient = 3
n_vert = 858
num_noise = 10
max_group = 216
num_neg = num_noise * max_group

loss_weight = 60.0
OCC_SIZE_TEST = 80
OCC_STRIDE_TEST = 10
ANN_FILE = None
test_data = '3dpw_occ'

workflow = [('train', 1)]
phase = 'test'
hparams = dict(
    DATASET=dict(
        DATASETS_AND_RATIOS='h36m_mpii_lspet_coco_mpi-inf-3dhp_0.35_0.05_0.05_0.2_0.35',
        FOCAL_LENGTH=5000.0,
        IMG_RES=img_res,
        eval_visible_joints=True,
        load_mask=False,
        test_indices=None,
        occ_info_file='exp/pare/3dpw_test_ds30_occ80stride10_pare_r50_grid_reproduce/result_occ_info_mpjpe.json',
        pred_initialization='exp/pare/3dpw_occ80str10_pare_r50/result_keypoints.json',
    ),
    MODEL=dict(
        RENDERER_PRED=dict(
            SIGMA=1e-3,
            GAMMA=1e-2,  # 1e-2
            FACES_PER_PIXEL=40,  # 40,
            RENDER_RES=img_res // 4,
        ),
        RENDERER_GT=dict(
            SIGMA=0, GAMMA=1e-2, FACES_PER_PIXEL=1, RENDER_RES=img_res // 4
        ),
        # LOSS_WEIGHT=60.0,
        N_ORIENT=n_orient,
        downsample_rate=downsample_rate,
    ),
    REGISTRANT=dict(
        SIGMA=1e-3,  # 1e-3,
        GAMMA=1e-2,  # 1e-2
        FACES_PER_PIXEL=1,  # 40,
        RENDER_RES=img_res // downsample_rate,
        downsample_rate=downsample_rate,
        n_orient=n_orient,
        n_vert=n_vert,
        theta=10,
        FOCAL_LENGTH=5000.0,
        IMG_RES=img_res,
        disable_occ=True,
        show_debug_info=False,
        thr_detected_joints=7,
        thr_openpose_conf=0.2,
        thr_vert_det=0.1,
        use_other_init=True,
        use_saved_coke=False,
        use_saved_partseg=False,
        use_otf_openpose=True,
        optimize_twoside=False,
        RUN_VISUALIZER=False,
    ),
    VISUALIZER=dict(
        IMG_RES=512,
        SHADER_TYPE='hard_phong',  # 'simple
        DEBUG_LOG_DIR='debug_log',
    ),
    mesh_sample_param_path='data/sample_params/uniform/sample_data_8-2021-04-05.npz',
    renderer_type='base',
    cumulative_iters=2,
    disable_inference_loss=False,
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
    type='NeuralSMPLFitting',
    body_model=dict(
        type='SMPL',
        keypoint_src='smpl_54',
        keypoint_dst='smpl_49',
        keypoint_approximate=True,
        model_path='data/body_models/smpl',
        extra_joints_regressor='data/body_models/J_regressor_extra.npy',
    ),
    num_epochs=1,
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
            fit_betas=False,  # True
        )
        # dict(
        #     num_iter=200,
        #     fit_global_orient=False,
        #     fit_transl=False,
        #     fit_body_pose=False,
        #     fit_betas=True,
        #     silhouette_loss_weight=10.0,
        #     # likelihood_weight=0.0,
        #     # keypoints2d_weight=0.0,
        #     # pose_prior_weight=0.0,
        #     ),
    ],
    optimizer=dict(type='Adam', lr=0.02, betas=(0.9, 0.99)),
    # optimizer=dict(type='LBFGS', lr=1, max_iter=20, max_eval=25, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100),
    keypoints2d_loss=dict(
        type='KeypointMSELoss', loss_weight=1e-2, reduction='sum', sigma=100
    ),
    vertices2d_loss=dict(
        type='KeypointMSELoss', loss_weight=0 * 1e-3, reduction='sum', sigma=100
    ),
    # shape_prior_loss=dict(
    #     type='ShapePriorLoss', loss_weight=0, reduction='sum'),
    joint_prior_loss=dict(type='JointPriorLoss', loss_weight=1e-1, reduction='sum'),
    # pose_prior_loss=dict(
    #     type='MaxMixturePrior',
    #     prior_folder='data',
    #     num_gaussians=8,
    #     loss_weight=0,
    #     reduction='sum'),
    pose_prior_loss=dict(
        type='VAEPosePrior', prior_folder='data', loss_weight=5e-4, reduction='sum'
    ),
    likelihood_loss=dict(
        type='LikelihoodLoss',
        loss_weight=1.0,
        reduction='sum',
    ),
    # segm_mask_loss=dict(
    #     type='CrossEntropyLoss',
    #     loss_weight=0*1.0,
    #     reduction='sum'
    # ),
    # silhouette_loss=dict(
    #     type='DICELoss',
    #     loss_weight=0*1.0,
    # ),
    ignore_keypoints=[
        'neck_openpose',
        'right_hip_openpose',
        'left_hip_openpose',
        'right_hip_extra',
        'left_hip_extra',
        'right_shoulder_openpose',
        'left_shoulder_openpose',
        'pelvis_openpose',
        'left_bigtoe_openpose',
        'left_smalltoe_openpose',
        'left_heel_openpose',
        'right_bigtoe_openpose',
        'right_smalltoe_openpose',
        'right_heel_openpose',
    ],
    camera=dict(
        type='PerspectiveCameras',
        convention='opencv',
        in_ndc=False,
        focal_length=5000,
        image_size=(224, 224),
        principal_point=(112.0, 112.0),
    ),
    hparams=hparams['REGISTRANT'],
)

# optimizer
optimizer = dict(backbone=dict(type='Adam', lr=5e-5), head=dict(type='Adam', lr=5e-5))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', gamma=0.1, step=[400])
runner = dict(type='EpochBasedRunner', max_epochs=50)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='')
        dict(type='TensorboardLoggerHook', interval=10),
    ],
)

# model settings
model = dict(
    type='ImageBodyModelEstimatorSE',
    backbone=dict(
        type='ResNet',
        depth=50,
        out_indices=[0, 1, 2, 3],
        norm_eval=False,
        # norm_cfg=dict(type='BN', requires_grad=True),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    # init_cfg=dict(type='Pretrained', checkpoint='data/pretrained/spin_official.pth')),
    head=dict(
        type='PareHeadwCoKe',
        num_joints=24,
        num_input_features=2048,
        use_heatmaps='part_segm',
        smpl_mean_params='data/body_models/smpl_mean_params.npz',
        coke_cfg=dict(
            type='CoKeHead',
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
    # loss_keypoints3d=dict(type='SmoothL1Loss', loss_weight=100),
    # loss_keypoints2d=dict(type='SmoothL1Loss', loss_weight=10),
    loss_keypoints3d=dict(type='MSELoss', loss_weight=5 * loss_weight),
    loss_keypoints2d=dict(type='MSELoss', loss_weight=2.5 * loss_weight),
    # loss_vertex=dict(type='L1Loss', loss_weight=0),
    loss_smpl_pose=dict(type='MSELoss', loss_weight=1.0 * loss_weight),
    loss_smpl_betas=dict(type='MSELoss', loss_weight=0.001 * loss_weight),
    loss_camera=dict(type='CameraPriorLoss', loss_weight=1.0),  # 60.0
    # loss_segm_mask=dict(type='CrossEntropyLoss', loss_weight=1.0*loss_weight),
    # loss_part_segm=dict(type='MSELoss', loss_weight=1.0),
    # init_cfg=dict(
    # type='Pretrained', checkpoint='data/pretrained/resnet50_spin_pw3d-e1857270_20211201.pth'),
    # type='Pretrained', checkpoint='data/pretrained/spin_pretrain.pth'),
    # type='Pretrained', checkpoint='data/pretrained/pare_official.pth'),
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
    'has_kp3d',
    'mask',
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
    dict(
        type='Collect',
        keys=['img', 'keypoints3d_init', *data_keys],
        meta_keys=meta_data_keys,
    ),
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
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(
        type='Collect',
        keys=['img', 'keypoints3d_init', *data_keys],
        meta_keys=meta_data_keys,
    ),
]
occimgen_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='MeshAffine', img_res=img_res),
    dict(
        type='AddOcclusionPatch',
        occ_size=OCC_SIZE_TEST,
        occ_stride=10,
        dtd_root='data/dtd/test',
    ),
    dict(type='Collect', keys=['img', 'center', 'scale', 'image_path'], meta_keys=[]),
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
    ),
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
        # convention='smpl_49',
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
elif test_data == 'briar':
    test_cfg = dict(
        type=dataset_type,
        body_model=dict(
            type='GenderedSMPL',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='data/body_models/smpl',
            joints_regressor='data/body_models/J_regressor_h36m.npy',
        ),
        dataset_name='briar',
        convention='smpl_49',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file=ANN_FILE,
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
        dataset_name='briar',
        convention='smpl_49',
        data_prefix='data',
        pipeline=vis_pipeline,
        ann_file=ANN_FILE,
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
    samples_per_gpu=16 if phase == 'train' else 6,
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
                ann_file='spin_h36m_train_mosh.npz',
            ),
            dict(
                type=dataset_type,
                dataset_name='mpi_inf_3dhp',
                data_prefix='data',
                pipeline=train_pipeline,
                convention='smpl_49',
                ann_file='spin_mpi_inf_3dhp_train_new_correct.npz',
            ),
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
                ann_file='eft_lspet.npz',
            ),  # size = 6829
            dict(
                type=dataset_type,
                dataset_name='mpii',
                data_prefix='data',
                pipeline=train_pipeline,
                convention='smpl_49',
                ann_file='eft_mpii.npz',
            ),  # size = 14667
            dict(
                type=dataset_type,
                dataset_name='coco',
                data_prefix='data',
                pipeline=train_pipeline,
                convention='smpl_49',
                ann_file='eft_coco_all.npz',
            ),  # size = 74834
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
            joints_regressor='data/body_models/J_regressor_h36m.npy',
        ),
        dataset_name='pw3d',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file='spin_pw3d_val.npz',
    ),
    test=test_cfg,
    visualization=vis_cfg,
    # test=dict(
    #     type=dataset_type,
    #     body_model=dict(
    #         type='GenderedSMPL',
    #         keypoint_src='h36m',
    #         keypoint_dst='h36m',
    #         model_path='data/body_models/smpl',
    #         joints_regressor='data/body_models/J_regressor_h36m.npy'),
    #     dataset_name='demo_dataset', #'pw3d',
    #     convention='smpl_49', #h36m
    #     data_prefix='data',
    #     pipeline=test_pipeline,
    #     ann_file='demo_dataset.npz',
    #     # pw3d_test_w_kp2d_ds30_op.npz,
    #     # pw3d_test_w_kp2d_ds30_op_occ80_pare_r50.npz,
    #     # pw3d_test_w_kp2d_ds30_op_occ80_pare_r50_w_parereproduceinit.npz
    #     # pw3d_test_w_kp2d_ds30_op_occ80_pare_r50_w_pareinit.npz
    #     # pw3d_test_w_kp2d_ds30_op_occ40_pare_r50_w_pareinit.npz
    #     # pw3d_test_w_kp2d_ds30_op_w_pareinit.npz
    #     # pw3d_test_w_kp2d_ds30_op_w_parereproduceinit.npz
    #     hparams=hparams['DATASET'],
    #     ), # spin_pw3d_test.npz, spin_pw3d_val.npz
    # test=dict(
    #     type=dataset_type,
    #     body_model=dict(
    #         type='SMPL',
    #         keypoint_src='h36m',
    #         keypoint_dst='h36m',
    #         model_path='data/body_models/smpl',
    #         joints_regressor='data/body_models/J_regressor_h36m.npy'),
    #     dataset_name='3doh50k',
    #     convention='smpl_49',
    #     data_prefix='data',
    #     pipeline=test_pipeline,
    #     ann_file='3doh_test_w_kp2d_rendered_op_w_pareinit.npz',
    #     # pw3d_test_w_kp2d_ds30_op.npz
    #     # pw3d_test_w_kp2d_ds30_op_occ80_pare_r50.npz
    #     # pw3d_test_w_kp2d_ds30_op_occ40_pare_r50.npz
    #     # 3doh_test_w_kp2d_op_w_pareinit.npz
    #     hparams=hparams['DATASET']),
    # test=dict(
    #     type='OccludedHumanImageDataset',
    #     orig_cfg=dict(
    #         type=dataset_type,
    #         body_model=dict(
    #             type='GenderedSMPL',
    #             keypoint_src='h36m',
    #             keypoint_dst='h36m',
    #             model_path='data/body_models/smpl',
    #             joints_regressor='data/body_models/J_regressor_h36m.npy'),
    #         dataset_name='pw3d', #'pw3d',
    #         convention='smpl_49', #h36m
    #         data_prefix='data',
    #         pipeline=test_pipeline_occ,
    #         ann_file='pw3d_test_w_kp2d_ds30_op.npz',
    #         hparams=hparams['DATASET'],
    #         ),
    #     occ_size=80,
    #     occ_stride=10,
    #     # textures_file='../nemo_for_human_pose/occ_analysis/texture_files_pare_r50_occ80.0_stride40_pampjpe.txt',
    #     ), # spin_pw3d_test.npz, spin_pw3d_val.npz
    # test=dict(
    #     type=dataset_type,
    #     body_model=dict(
    #         type='GenderedSMPL',
    #         keypoint_src='h36m',
    #         keypoint_dst='h36m',
    #         model_path='data/body_models/smpl',
    #         joints_regressor='data/body_models/J_regressor_h36m.npy'),
    #     dataset_name='pw3d',
    #     convention='smpl_49',
    #     data_prefix='data',
    #     pipeline=test_pipeline_occ, #vis_pipeline, #occimgen_pipeline,
    #     ann_file='pw3d_test_w_kp2d_ds30_op_w_pareinit.npz',
    #     hparams=hparams['DATASET'],),
    # visualization=dict(
    #     type=dataset_type,
    #     body_model=dict(
    #         type='GenderedSMPL',
    #         keypoint_src='h36m',
    #         keypoint_dst='h36m',
    #         model_path='data/body_models/smpl',
    #         joints_regressor='data/body_models/J_regressor_h36m.npy'),
    #     dataset_name='pw3d',
    #     convention='smpl_49',
    #     data_prefix='data',
    #     pipeline=vis_pipeline_occ, #vis_pipeline, #occimgen_pipeline,
    #     ann_file='pw3d_test_w_kp2d_ds30_op_w_pareinit.npz',
    #     hparams=hparams['DATASET'],),
    # test=dict(
    #     type=dataset_type,
    #     body_model=dict(
    #         type='GenderedSMPL',
    #         keypoint_src='h36m',
    #         keypoint_dst='h36m',
    #         model_path='data/body_models/smpl',
    #         joints_regressor='data/body_models/J_regressor_h36m.npy'),
    #     dataset_name='briar',
    #     convention='smpl_49',
    #     data_prefix='data',
    #     pipeline=test_pipeline,
    #     ann_file="G00017_set1_rand_P3245-VE_ACCC8EF273B3_e0079950.npz",
    #     hparams=hparams['DATASET'],
    #     ),
    # visualization=dict(
    #     type=dataset_type,
    #     body_model=dict(
    #         type='GenderedSMPL',
    #         keypoint_src='h36m',
    #         keypoint_dst='h36m',
    #         model_path='data/body_models/smpl',
    #         joints_regressor='data/body_models/J_regressor_h36m.npy'),
    #     dataset_name='briar',
    #     convention='smpl_49',
    #     data_prefix='data',
    #     pipeline=vis_pipeline,
    #     ann_file="G00017_set1_rand_P3245-VE_ACCC8EF273B3_e0079950.npz",
    #     hparams=hparams['DATASET'],),
    # test=dict(
    #     type=dataset_type,
    #     body_model=dict(
    #         type='GenderedSMPL',
    #         keypoint_src='h36m',
    #         keypoint_dst='h36m',
    #         model_path='data/body_models/smpl',
    #         joints_regressor='data/body_models/J_regressor_h36m.npy'),
    #     dataset_name='pw3d',
    #     convention='smpl_49',
    #     data_prefix='data',
    #     pipeline=test_pipeline, #vis_pipeline, #occimgen_pipeline,
    #     ann_file='pw3d_test_w_kp2d_ds30_op_w_pareinit.npz',
    #     hparams=hparams['DATASET'],)
)
