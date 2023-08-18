_base_ = ['../_base_/default_runtime.py']
use_adversarial_train = True
find_unused_parameters = True # Debug

load_from = 'exp/pare_hrnet_w32_2dinit/epoch_120.pth'
checkpoint_config = dict(interval=1)

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
lr_config = dict(policy='step', gamma=0.1, step=[200])
runner = dict(type='EpochBasedRunner', max_epochs=120)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='')
        dict(type='TensorboardLoggerHook', interval=10)
    ])

hparams = dict(
            DATASET=dict(
                DATASETS_AND_RATIOS='h36m_mpii_lspet_coco_mpi-inf-3dhp_0.35_0.05_0.05_0.2_0.35',
                FOCAL_LENGTH=5000.0, 
                IMG_RES=img_res,
                eval_visible_joints=True,
                # occ_info_file='exp/pymaf/occ40str10/result_occ_info_mpjpe.json',
                ),
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
            REGISTRANT=dict(
                use_otf_openpose=True,
            ),
            VISUALIZER=dict(
                IMG_RES=512
            ),
            # DEBUG=dict(
            #     eval_occ_seg=False,
            # ),
            save_partseg=False,
            cumulative_iters=16,
        )

registrant=dict(
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
            )),
        # stage 2
        dict(
            num_iter=10,
            fit_global_orient=True,
            fit_transl=True,
            fit_body_pose=True,
            fit_betas=True,
            joint_weights=dict(body_weight=5.0, use_shoulder_hip_only=False))
    ],
    optimizer=dict(
        type='LBFGS', max_iter=20, lr=1e-2, line_search_fn='strong_wolfe'),
    keypoints2d_loss=dict(
        type='KeypointMSELoss', loss_weight=1.0, reduction='sum', sigma=100),
    keypoints3d_loss=dict(
        type='KeypointMSELoss', loss_weight=10, reduction='sum', sigma=100),
    shape_prior_loss=dict(type='ShapePriorLoss', loss_weight=1, reduction='sum'),
    joint_prior_loss=dict(
        type='JointPriorLoss',
        loss_weight=20,
        reduction='sum',
        smooth_spine=True,
        smooth_spine_loss_weight=20,
        use_full_body=True),
    smooth_loss=dict(type='SmoothJointLoss', loss_weight=0, reduction='sum'),
    pose_prior_loss=dict(
        type='MaxMixturePrior',
        prior_folder='data',
        num_gaussians=8,
        loss_weight=4.78**2,
        reduction='sum'),
    ignore_keypoints=[
        'neck_openpose', 'right_hip_openpose', 'left_hip_openpose',
        'right_hip_extra', 'left_hip_extra'
    ],
    camera=dict(
        type='PerspectiveCameras',
        convention='opencv',
        in_ndc=False,
        focal_length=5000,
        image_size=(224, 224),
        principal_point=(112, 112)),
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
            fit_betas=False),
        # stage 2
        dict(
            num_iter=50,
            fit_global_orient=True,
            fit_transl=False,
            fit_body_pose=True,
            fit_betas=True),
    ],
    optimizer=dict(type='Adam', lr=1e-2, betas=(0.9, 0.999)),
    keypoints2d_loss=dict(
        type='KeypointMSELoss', loss_weight=1.0, reduction='sum', sigma=100),
    shape_prior_loss=dict(
        type='ShapePriorLoss', loss_weight=5.0**2, reduction='sum'),
    joint_prior_loss=dict(
        type='JointPriorLoss', loss_weight=15.2**2, reduction='sum'),
    pose_prior_loss=dict(
        type='MaxMixturePrior',
        prior_folder='data',
        num_gaussians=8,
        loss_weight=4.78**2,
        reduction='sum'),
    ignore_keypoints=[
        'neck_openpose', 'right_hip_openpose', 'left_hip_openpose',
        'right_hip_extra', 'left_hip_extra'
    ],
    camera=dict(
        type='PerspectiveCameras',
        convention='opencv',
        in_ndc=False,
        focal_length=5000,
        image_size=(img_res, img_res),
        principal_point=(img_res / 2, img_res / 2)))

# model settings
width = 32
downsample = False
use_conv = True
hrnet_extra = dict(
    stage1=dict(
        num_modules=1,
        num_branches=1,
        block='BOTTLENECK',
        num_blocks=(4, ),
        num_channels=(64, )),
    stage2=dict(
        num_modules=1,
        num_branches=2,
        block='BASIC',
        num_blocks=(4, 4),
        num_channels=(width, width * 2)),
    stage3=dict(
        num_modules=4,
        num_branches=3,
        block='BASIC',
        num_blocks=(4, 4, 4),
        num_channels=(width, width * 2, width * 4)),
    stage4=dict(
        num_modules=3,
        num_branches=4,
        block='BASIC',
        num_blocks=(4, 4, 4, 4),
        num_channels=(width, width * 2, width * 4, width * 8)),
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
    type='ImageBodyModelEstimator',
    backbone=dict(
        type='PoseHighResolutionNet',
        extra=hrnet_extra,
        num_joints=24,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='data/pretrained/hrnet_pretrain_cpu.pth'
            ),
        norm_cfg=dict(type='SyncBN', requires_grad=True),),
    head=dict(
        type='PareHead',
        num_joints=24, 
        num_input_features=480,
        use_heatmaps='part_segm',
        smpl_mean_params='data/body_models/smpl_mean_params.npz'),
    body_model_train=body_model,
    body_model_test=dict(
        type='SMPL',
        keypoint_src='h36m',
        keypoint_dst='h36m',
        model_path='data/body_models/smpl',
        joints_regressor='data/body_models/J_regressor_h36m.npy'),
    registrant=None, #registrant_new, #registrant, #None,
    convention='smpl_49',
    # loss_keypoints3d=dict(type='SmoothL1Loss', loss_weight=100),
    # loss_keypoints2d=dict(type='SmoothL1Loss', loss_weight=10),
    loss_keypoints3d=dict(type='MSELoss', loss_weight=5*loss_weight),
    loss_keypoints2d=dict(type='MSELoss', loss_weight=2.5*loss_weight),
    # loss_vertex=dict(type='L1Loss', loss_weight=0),
    loss_smpl_pose=dict(type='MSELoss', loss_weight=1.0*loss_weight),
    loss_smpl_betas=dict(type='MSELoss', loss_weight=0.001*loss_weight),
    loss_camera=dict(type='CameraPriorLoss', loss_weight=1.0),
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
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data_keys = [
    'has_smpl', 'smpl_body_pose', 'smpl_global_orient', 'smpl_betas',
    'smpl_transl', 'keypoints2d', 'keypoints3d', 'is_flipped', 'center',
    'scale', 'rotation', 'sample_idx', 'idx'
]
meta_data_keys = ['dataset_name', 'img_norm_cfg', 'image_path', 'occ_size', 
                'occ_stride', 'occ_idx', 'texture_file', 'texture_crop_tl']
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
test_pipeline_occ = [
    dict(type='LoadImageFromFile'),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='MeshAffine', img_res=img_res),
    dict(type='AddOcclusionPatch', occ_size=80, occ_stride=10, dtd_root='data/dtd/test'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect', keys=['img', *data_keys], meta_keys=meta_data_keys)
]

vis_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='MeshAffine', img_res=hparams['VISUALIZER']['IMG_RES']),
    # dict(type='AddOcclusionPatch', occ_size=80, occ_stride=10, dtd_root='data/dtd/test'),
    dict(type='Collect', keys=['img', 'center', 'scale', 'image_path'], meta_keys=[])
]

vis_pipeline_occ = [
    dict(type='LoadImageFromFile'),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='MeshAffine', img_res=hparams['VISUALIZER']['IMG_RES']),
    dict(type='AddOcclusionPatch', occ_size=80, occ_stride=10, dtd_root='data/dtd/test'),
    dict(type='Collect', keys=['img', 'center', 'scale', 'image_path'], meta_keys=[])
]

dataset_gen_pipeline_occ = [
    dict(type='LoadImageFromFile'),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='MeshAffine', img_res=224), # image size
    dict(type='AddOcclusionPatch', occ_size=80, occ_stride=10, dtd_root='data/dtd/test'),
    # dict(type='ImageToTensor', keys=['img']),
    # dict(type='ToTensor', keys=['keypoints2d']),
    dict(type='Collect', keys=['img', 'center', 'scale', 'image_path', 'keypoints2d'], meta_keys=meta_data_keys, use_dc=False)
]

dataset_gen_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='MeshAffine', img_res=224), # image size
    # dict(type='AddOcclusionPatch', occ_size=80, occ_stride=10, dtd_root='data/dtd/test'),
    # dict(type='ImageToTensor', keys=['img']),
    # dict(type='ToTensor', keys=['keypoints2d']),
    dict(type='Collect', keys=['img', 'center', 'scale', 'image_path', 'keypoints2d'], meta_keys=meta_data_keys, use_dc=False)
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
    samples_per_gpu=4, #32,
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
                ann_file='spin_h36m_train_mosh.npz'), 
            dict(
                type=dataset_type,
                dataset_name='mpi_inf_3dhp',
                data_prefix='data',
                pipeline=train_pipeline,
                convention='smpl_49',
                ann_file='spin_mpi_inf_3dhp_train_new_correct_nosmpl.npz'),
                # spin_mpi_inf_3dhp_train_new_correct.npz
                # spin_mpi_inf_3dhp_train_new_correct_nosmpl.npz
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
    # test=dict(
    #     type=dataset_type,
    #     body_model=body_model,
    #     # dataset_name='h36m',
    #     dataset_name='mpi_inf_3dhp',
    #     data_prefix='data',
    #     pipeline=test_pipeline,
    #     convention='h36m',
    #     ann_file='mpi_inf_3dhp_test.npz',
    #     # ann_file='h36m_valid_protocol2.npz',
    #     # ann_file='spin_h36m_train_reproduce.npz'
    #     # ann_file='spin_h36m_train_mosh.npz'
    #     # ann_file='spin_mpi_inf_3dhp_train_new_correct.npz'
    #     hparams=dict(
    #         DATASETS_AND_RATIOS='h36m_mpii_lspet_coco_mpi-inf-3dhp_0.35_0.05_0.05_0.2_0.35',
    #         FOCAL_LENGTH=5000.0, 
    #         IMG_RES=img_res,
    #         eval_visible_joints=False), # spin_pw3d_test.npz
    #     ),
    
    # test=dict(
    #     type=dataset_type,
    #     body_model=dict(
    #         type='GenderedSMPL',
    #         keypoint_src='h36m',
    #         keypoint_dst='h36m',
    #         model_path='data/body_models/smpl',
    #         joints_regressor='data/body_models/J_regressor_h36m.npy'),
    #     dataset_name='pw3d',
    #     convention='h36m',
    #     data_prefix='data',
    #     pipeline=test_pipeline,
    #     ann_file='pw3d_test_w_kp2d_ds30_op.npz', 
    #     # pw3d_test_w_kp2d_ds30_op_occ80stride10_pare_r50.npz
    #     # pw3d_test_w_kp2d_ds30_op_occ80_pare_r50.npz
    #     # pw3d_test_w_kp2d_ds30_op_occ40_pare_r50.npz
    #     # pw3d_test_w_kp2d_ds30_op.npz
    #     # pw3d_test_w_kp2d_ds30_op_occ80_pare_r50.npz
    #     # 3doh_test_w_kp2d.npz
    #     hparams=dict(
    #             DATASETS_AND_RATIOS='h36m_mpii_lspet_coco_mpi-inf-3dhp_0.35_0.05_0.05_0.2_0.35',
    #             FOCAL_LENGTH=5000.0, 
    #             IMG_RES=img_res,
    #             eval_visible_joints=True)), # spin_pw3d_test.npz
    # test=dict(
    #     type=dataset_type,
    #     body_model=dict(
    #         type='GenderedSMPL',
    #         keypoint_src='h36m',
    #         keypoint_dst='h36m',
    #         model_path='data/body_models/smpl',
    #         joints_regressor='data/body_models/J_regressor_h36m.npy'),
    #     dataset_name='pw3d',
    #     convention='h36m',
    #     data_prefix='data',
    #     pipeline=test_pipeline,
    #     ann_file='pw3d_occ_w_kp2d_ds30_correct.npz', 
    #     # pw3d_test_w_kp2d_ds30_op_occ80stride10_pare_r50.npz
    #     # pw3d_test_w_kp2d_ds30_op_occ80_pare_r50.npz
    #     # pw3d_test_w_kp2d_ds30_op_occ40_pare_r50.npz
    #     # pw3d_test_w_kp2d_ds30_op.npz
    #     # pw3d_test_w_kp2d_ds30_op_occ80_pare_r50.npz
    #     # 3doh_test_w_kp2d.npz
    #     hparams=dict(
    #             DATASETS_AND_RATIOS='h36m_mpii_lspet_coco_mpi-inf-3dhp_0.35_0.05_0.05_0.2_0.35',
    #             FOCAL_LENGTH=5000.0, 
    #             IMG_RES=img_res,
    #             eval_visible_joints=True)), # spin_pw3d_test.npz

    test=dict(
        type='OccludedHumanImageDataset',
        orig_cfg=dict(
            type=dataset_type,
            body_model=dict(
                type='GenderedSMPL',
                keypoint_src='h36m',
                keypoint_dst='h36m',
                model_path='data/body_models/smpl',
                joints_regressor='data/body_models/J_regressor_h36m.npy'),
            dataset_name='pw3d',
            convention='smpl_49', # 'h36m', # TODO check why h36m is not working
            data_prefix='data',
            pipeline=test_pipeline_occ, #dataset_gen_pipeline, #test_pipeline_occ,
            ann_file='pw3d_test_w_kp2d_ds30_op_1184.npz', 
            # pw3d_test_w_kp2d_ds30_op_occ40_pare_r50.npz
            # pw3d_test_w_kp2d_ds30_op.npz
            # pw3d_test_w_kp2d_ds30_op_occ80_pare_r50.npz
            # 3doh_test_w_kp2d.npz
            hparams=dict(
                    DATASETS_AND_RATIOS='h36m_mpii_lspet_coco_mpi-inf-3dhp_0.35_0.05_0.05_0.2_0.35',
                    FOCAL_LENGTH=5000.0, 
                    IMG_RES=img_res,
                    eval_visible_joints=True)), # spin_pw3d_test.npz
        # here occ_size and occ_stride are only used to calculate n_grid
        # the actual occ_size and occ_stride are set in test_pipeline
        occ_size=80, 
        occ_stride=10,
        hparams=dict(
                occ_info_file='exp/3dnbf_pare_step2_kp3d_accu_2dinit/3dpw_occ80str10_fitz5e-5_1184_opt_e20_lr0.02/result_occ_info.json'),
        # textures_file='../nemo_for_human_pose/occ_analysis/texture_files_pare_r50_occ80.0_stride40_pampjpe.txt',
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
    #     convention='h36m', #,'smpl_49'
    #     data_prefix='data',
    #     pipeline=test_pipeline_occ, # vis_pipeline
    #     ann_file='pw3d_test_w_kp2d_ds30_op.npz', 
    #     hparams=dict(
    #             DATASETS_AND_RATIOS='h36m_mpii_lspet_coco_mpi-inf-3dhp_0.35_0.05_0.05_0.2_0.35',
    #             FOCAL_LENGTH=5000.0, 
    #             IMG_RES=img_res,
    #             eval_visible_joints=True,
    #             occ_info_file='exp/3dnbf/3dpw_test_ds30_occ80stride10_grid_pare_r50_multihypo_d128_reproduce_e15_newpthop_fixopbug/result_occ_info_mpjpe.json'
    #             # occ_info_file='exp/mesh_graphormer/result_occ_info_mpjpe.json'
    #             # occ_info_file='exp/mesh_graphormer/occ40str10/result_occ_info_mpjpe.json'
    #             # occ_info_file='exp/pare/3dpw_test_ds30_occ80stride40_pare_r50_grid/result_occ_info_mpjpe.json'
    #             # occ_info_file='exp/eft/3dpw_test_ds30_occ40stride10_grid/result_occ_info_mpjpe.json'
    #             # occ_info_file='exp/spin_official/3dpw_test_ds30_occ40stride10_grid/result_occ_info_mpjpe.json'
    #             )
    #     ),

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
    #     ann_file='3doh_test_w_kp2d_rendered.npz', 
    #     # pw3d_test_w_kp2d_ds30_op.npz
    #     # pw3d_test_w_kp2d_ds30_op_occ80_pare_r50.npz
    #     # pw3d_test_w_kp2d_ds30_op_occ40_pare_r50.npz
    #     hparams=dict(
    #             DATASETS_AND_RATIOS='h36m_mpii_lspet_coco_mpi-inf-3dhp_0.35_0.05_0.05_0.2_0.35',
    #             FOCAL_LENGTH=5000.0, 
    #             IMG_RES=img_res,
    #             eval_visible_joints=True)), # spin_pw3d_test.npz
    # test=dict(
    #     type=dataset_type,
    #     body_model=dict(
    #         type='GenderedSMPL',
    #         keypoint_src='h36m',
    #         keypoint_dst='h36m',
    #         model_path='data/body_models/smpl',
    #         joints_regressor='data/body_models/J_regressor_h36m.npy'),
    #     dataset_name='wb_img_v2',
    #     convention='smpl_49',
    #     data_prefix='data',
    #     pipeline=test_pipeline, #vis_pipeline, test_pipeline_occ, 
    #     ann_file='briar_v2.npz', 
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
    #     dataset_name='wb_img_v2',
    #     convention='smpl_49',
    #     data_prefix='data',
    #     pipeline=vis_pipeline, #vis_pipeline, test_pipeline_occ, 
    #     ann_file='briar_v2.npz', 
    #     hparams=hparams['DATASET'],
    #     ),

    # visualization=dict(
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
    #     pipeline=vis_pipeline,
    #     ann_file='3doh_test_w_kp2d_rendered.npz', 
    #     # pw3d_test_w_kp2d_ds30_op.npz
    #     # pw3d_test_w_kp2d_ds30_op_occ80_pare_r50.npz
    #     # pw3d_test_w_kp2d_ds30_op_occ40_pare_r50.npz
    #     hparams=hparams['DATASET'], # spin_pw3d_test.npz
    # ),

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
    #     pipeline=vis_pipeline, #_occ, #vis_pipeline, #occimgen_pipeline,
    #     ann_file='pw3d_occ_w_kp2d_ds30_correct.npz', 
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
    #     pipeline=vis_pipeline, #_occ, #vis_pipeline, #occimgen_pipeline,
    #     ann_file='pw3d_test_w_kp2d_ds30_op_w_pareinit.npz', 
    #     hparams=hparams['DATASET'],),

    visualization=dict(
        type='OccludedHumanImageDataset',
        orig_cfg=dict(
            type=dataset_type,
            body_model=dict(
                type='GenderedSMPL',
                keypoint_src='h36m',
                keypoint_dst='h36m',
                model_path='data/body_models/smpl',
                joints_regressor='data/body_models/J_regressor_h36m.npy'),
            dataset_name='pw3d',
            convention='h36m', #'smpl_49', # TODO check why h36m is not working
            data_prefix='data',
            pipeline=vis_pipeline_occ, #dataset_gen_pipeline, #test_pipeline_occ,
            ann_file='pw3d_test_w_kp2d_ds30_op_sample.npz',  #1184
            # pw3d_test_w_kp2d_ds30_op_occ40_pare_r50.npz
            # pw3d_test_w_kp2d_ds30_op.npz
            # pw3d_test_w_kp2d_ds30_op_occ80_pare_r50.npz
            # 3doh_test_w_kp2d.npz
            hparams=dict(
                    DATASETS_AND_RATIOS='h36m_mpii_lspet_coco_mpi-inf-3dhp_0.35_0.05_0.05_0.2_0.35',
                    FOCAL_LENGTH=5000.0, 
                    IMG_RES=img_res,
                    eval_visible_joints=True)), # spin_pw3d_test.npz
        # here occ_size and occ_stride are only used to calculate n_grid
        # the actual occ_size and occ_stride are set in test_pipeline
        occ_size=80, 
        occ_stride=10,
        hparams=dict(
                # occ_info_file='exp/3dnbf_pare_step2_kp3d_accu_2dinit/3dpw_occ80str10_fitz5e-5_1184_opt_e20_lr0.02/result_occ_info.json'),
                occ_info_file='exp/3dnbf_pare_step2_kp3d_accu_2dinit/3dpw_occ80str10_sample_opt_e20_lr0.02/result_occ_info.json'),
        # textures_file='../nemo_for_human_pose/occ_analysis/texture_files_pare_r50_occ80.0_stride40_pampjpe.txt',
    ),

    dataset_gen=dict(
        type='OccludedHumanImageDataset',
        orig_cfg=dict(
            type=dataset_type,
            body_model=dict(
                type='GenderedSMPL',
                keypoint_src='h36m',
                keypoint_dst='h36m',
                model_path='data/body_models/smpl',
                joints_regressor='data/body_models/J_regressor_h36m.npy'),
            dataset_name='pw3d',
            convention='h36m',
            data_prefix='data',
            pipeline=dataset_gen_pipeline_occ, #dataset_gen_pipeline, #test_pipeline_occ,
            ann_file='pw3d_test_w_kp2d_ds30_op.npz', 
            # pw3d_test_w_kp2d_ds30_op_occ40_pare_r50.npz
            # pw3d_test_w_kp2d_ds30_op.npz
            # pw3d_test_w_kp2d_ds30_op_occ80_pare_r50.npz
            # 3doh_test_w_kp2d.npz
            hparams=dict(
                    DATASETS_AND_RATIOS='h36m_mpii_lspet_coco_mpi-inf-3dhp_0.35_0.05_0.05_0.2_0.35',
                    FOCAL_LENGTH=5000.0, 
                    IMG_RES=img_res,
                    eval_visible_joints=True)), # spin_pw3d_test.npz
        # here occ_size and occ_stride are only used to calculate n_grid
        # the actual occ_size and occ_stride are set in test_pipeline
        occ_size=80, 
        occ_stride=10,
        # textures_file='../nemo_for_human_pose/occ_analysis/texture_files_pare_r50_occ80.0_stride40_pampjpe.txt',
    ),

    # dataset_gen=dict(
    #     type=dataset_type,
    #     body_model=dict(
    #         type='GenderedSMPL',
    #         keypoint_src='h36m',
    #         keypoint_dst='h36m',
    #         model_path='data/body_models/smpl',
    #         joints_regressor='data/body_models/J_regressor_h36m.npy'),
    #     dataset_name='pw3d',
    #     convention='h36m',
    #     data_prefix='data',
    #     pipeline=dataset_gen_pipeline,
    #     ann_file='pw3d_test_w_kp2d_ds30_op.npz', 
    #     # pw3d_test_w_kp2d_ds30_op_occ80stride10_pare_r50.npz
    #     # pw3d_test_w_kp2d_ds30_op_occ80_pare_r50.npz
    #     # pw3d_test_w_kp2d_ds30_op_occ40_pare_r50.npz
    #     # pw3d_test_w_kp2d_ds30_op.npz
    #     # pw3d_test_w_kp2d_ds30_op_occ80_pare_r50.npz
    #     # 3doh_test_w_kp2d.npz
    #     hparams=dict(
    #             DATASETS_AND_RATIOS='h36m_mpii_lspet_coco_mpi-inf-3dhp_0.35_0.05_0.05_0.2_0.35',
    #             FOCAL_LENGTH=5000.0, 
    #             IMG_RES=img_res,
    #             eval_visible_joints=True)), # spin_pw3d_test.npz

    # dataset_gen=dict(
    #     type=dataset_type,
    #     body_model=dict(
    #         type='GenderedSMPL',
    #         keypoint_src='h36m',
    #         keypoint_dst='h36m',
    #         model_path='data/body_models/smpl',
    #         joints_regressor='data/body_models/J_regressor_h36m.npy'),
    #     dataset_name='3doh50k',
    #     convention='h36m',
    #     data_prefix='data',
    #     pipeline=dataset_gen_pipeline,
    #     ann_file='3doh_test_w_kp2d_rendered.npz', 
    #     # pw3d_test_w_kp2d_ds30_op.npz
    #     # pw3d_test_w_kp2d_ds30_op_occ80_pare_r50.npz
    #     # pw3d_test_w_kp2d_ds30_op_occ40_pare_r50.npz
    #     hparams=dict(
    #             DATASETS_AND_RATIOS='h36m_mpii_lspet_coco_mpi-inf-3dhp_0.35_0.05_0.05_0.2_0.35',
    #             FOCAL_LENGTH=5000.0, 
    #             IMG_RES=img_res,
    #             eval_visible_joints=True)), # spin_pw3d_test.npz
                
    # test=dict(
    #     type=dataset_type,
    #     body_model=dict(
    #         type='GenderedSMPL',
    #         keypoint_src='h36m',
    #         keypoint_dst='h36m',
    #         model_path='data/body_models/smpl',
    #         joints_regressor='data/body_models/J_regressor_h36m.npy'),
    #     dataset_name='briar_synthetic_part_allShape',
    #     convention='smpl_49',
    #     data_prefix='data',
    #     pipeline=test_pipeline, #vis_pipeline, test_pipeline_occ, 
    #     ann_file='sim_briar_test_w_kp2d.npz', 
    #     hparams=hparams['DATASET'],
    #     ),
)
