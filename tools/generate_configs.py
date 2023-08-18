import os
import os.path as osp


def create_exp(
    phase="test",
    OCC_SIZE_TEST=40,
    OCC_STRIDE_TEST=10,
    OCC_INFO_FILE=None,
    PRED_INITIALIZATION=None,
    OPT_LR=0.002,
    N_ORIENT=3,
    COKE_LOSS_ON=1,
    COKE_LOSS_WEIGHT=1,
    VERTICES_LOSS_WEIGHT=0,
    KP2D_LOSS_ON=1,
    KP2D_LOSS_WEIGHT=1,
    D_FEATURE=128,
    SILHOUETTE_LOSS_WEIGHT=0,
    TWO_SIDE=False,
    FIT_Z=False,
    test_data="3dpw_advocc_grid",
    output_file="exps/my_exp.py",
    template_file="configs/3dnbf/resnet50_pare_w_coke_pw3d_step2_template.py",
    ann_file='',
):
    assert phase in ["train", "test"]
    assert test_data in [
        "3dpw_advocc_grid",
        "3dpw_advocc",
        "3dpw_occ",
        "3dpw",
        "h36m",
        "mpi_inf_3dhp",
        "3doh50k",
        "briar",
    ]

    template = open(template_file).read()

    params = [
        (phase, "$PHASE"),
        (OCC_SIZE_TEST, "$OCC_SIZE_TEST"),
        (OCC_STRIDE_TEST, "$OCC_STRIDE_TEST"),
        (TWO_SIDE, "$TWO_SIDE"),
        (FIT_Z, "$FIT_Z"),
        (OCC_INFO_FILE, "$OCC_INFO_FILE"),
        (PRED_INITIALIZATION, "$PRED_INITIALIZATION"),
        (OPT_LR, "$OPT_LR"),
        (N_ORIENT, "$N_ORIENT"),
        (COKE_LOSS_ON, "$COKE_LOSS_ON"),
        (COKE_LOSS_WEIGHT, "$COKE_LOSS_WEIGHT"),
        (VERTICES_LOSS_WEIGHT, "$VERTICES_LOSS_WEIGHT"),
        (KP2D_LOSS_ON, "$KP2D_LOSS_ON"),
        (KP2D_LOSS_WEIGHT, "$KP2D_LOSS_WEIGHT"),
        (D_FEATURE, "$D_FEATURE"),
        (SILHOUETTE_LOSS_WEIGHT, "$SILHOUETTE_LOSS_WEIGHT"),
        (test_data, "$TEST_DATA"),
        (ann_file, "$ANN_FILE"),
    ]

    for v, k in params:
        if isinstance(v, str):
            template = template.replace('"' + k + '"', f'"{v}"')
        else:
            template = template.replace('"' + k + '"', str(v))

    os.makedirs(osp.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write(template)


root = "exps_r50"
os.makedirs(osp.join(root, '_base_'), exist_ok=True)
os.system(f"cp configs/_base_/default_runtime.py {osp.join(root, '_base_')}")

for occ, stride in [(40, 10), (80, 10)]:
    create_exp(
        phase='train',
        OCC_SIZE_TEST=occ,
        OCC_STRIDE_TEST=stride,
        OCC_INFO_FILE=None, 
        PRED_INITIALIZATION=None,
        OPT_LR=0.02,
        N_ORIENT=3,
        COKE_LOSS_ON=1,
        KP2D_LOSS_ON=1,
        D_FEATURE=128,
        VERTICES_LOSS_WEIGHT=0.0,
        test_data="3dpw_advocc_grid",
        template_file="configs/3dnbf/resnet50_pare_w_coke_pw3d_step2_template.py",
        output_file=f"{root}/3dpw_advocc_grid/occ{occ}str{stride}_grid.py",
    )
    create_exp(
        phase='train',
        OCC_SIZE_TEST=occ,
        OCC_STRIDE_TEST=stride,
        OCC_INFO_FILE=f'{root}/3dpw_advocc_grid/occ80str10_grid/result_occ_info_mpjpe.json',
        PRED_INITIALIZATION=None,
        OPT_LR=0.02,
        N_ORIENT=3,
        COKE_LOSS_ON=1,
        KP2D_LOSS_ON=1,
        D_FEATURE=128,
        VERTICES_LOSS_WEIGHT=0.0,
        test_data="3dpw_advocc",
        template_file="configs/3dnbf/resnet50_pare_w_coke_pw3d_step2_template.py",
        output_file=f"{root}/3dpw_advocc/occ{occ}str{stride}_regressor.py",
    )
    create_exp(
        phase='test',
        OCC_SIZE_TEST=occ,
        OCC_STRIDE_TEST=stride,
        OCC_INFO_FILE=f'{root}/3dpw_advocc_grid/occ80str10_grid/result_occ_info_mpjpe.json',
        PRED_INITIALIZATION=None,
        OPT_LR=0.02,
        N_ORIENT=3,
        COKE_LOSS_ON=1,
        KP2D_LOSS_ON=1,
        D_FEATURE=128,
        TWO_SIDE=True,
        VERTICES_LOSS_WEIGHT=0.0,
        test_data="3dpw_advocc",
        template_file="configs/3dnbf/resnet50_pare_w_coke_pw3d_step2_template.py",
        output_file=f"{root}/3dpw_advocc/occ{occ}str{stride}.py",
    )
    create_exp(
        phase='test',
        OCC_SIZE_TEST=occ,
        OCC_STRIDE_TEST=stride,
        OCC_INFO_FILE=f'{root}/3dpw_advocc_grid/occ80str10_grid/result_occ_info_mpjpe.json',
        PRED_INITIALIZATION=None,
        OPT_LR=0.02,
        N_ORIENT=3,
        COKE_LOSS_ON=1,
        KP2D_LOSS_ON=1,
        D_FEATURE=128,
        TWO_SIDE=True,
        VERTICES_LOSS_WEIGHT=0.0,
        test_data="3dpw_advocc",
        template_file="configs/3dnbf/resnet50_pare_w_coke_pw3d_step2_template.py",
        output_file=f"{root}/3dpw_advocc/occ{occ}str{stride}.py",
    )
create_exp(
    phase='test',
    PRED_INITIALIZATION=None,
    OPT_LR=0.002,
    N_ORIENT=3,
    COKE_LOSS_ON=1,
    KP2D_LOSS_ON=1,
    D_FEATURE=128,
    TWO_SIDE=False,
    VERTICES_LOSS_WEIGHT=0.0,
    test_data="3dpw_occ",
    template_file="configs/3dnbf/resnet50_pare_w_coke_pw3d_step2_template.py",
    output_file=f"{root}/3dpw_occ/3dpw_occ.py",
)
create_exp(
    phase='test',
    PRED_INITIALIZATION=None,
    OPT_LR=0.002,
    N_ORIENT=3,
    COKE_LOSS_ON=1,
    KP2D_LOSS_ON=1,
    D_FEATURE=128,
    TWO_SIDE=False,
    VERTICES_LOSS_WEIGHT=0.0,
    test_data="3doh50k",
    template_file="configs/3dnbf/resnet50_pare_w_coke_pw3d_step2_template.py",
    output_file=f"{root}/3doh50k/3doh50k.py",
)
create_exp(
    phase='test',
    PRED_INITIALIZATION=None,
    OPT_LR=0.002,
    N_ORIENT=3,
    COKE_LOSS_ON=1,
    KP2D_LOSS_ON=1,
    D_FEATURE=128,
    TWO_SIDE=False,
    VERTICES_LOSS_WEIGHT=0.0,
    test_data="3dpw",
    template_file="configs/3dnbf/resnet50_pare_w_coke_pw3d_step2_template.py",
    output_file=f"{root}/3dpw/3dpw.py",
)