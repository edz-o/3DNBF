import sys
import os
import os.path as osp
from glob import glob

INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
CHECKPOINT = sys.argv[3]
if len(sys.argv) > 4:
    REST = sys.argv[4:]
else:
    REST = ''
for cfg in glob(f'{INPUT_DIR}/*.py'):
    task_name = osp.basename(osp.splitext(cfg)[0])
    os.makedirs(f'{OUTPUT_DIR}/{task_name}', exist_ok=True)
    os.system(f'cp {cfg} {OUTPUT_DIR}/{task_name}')
    # evaluation
    cmd = f'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash tools/dist_test.sh ' \
            f'{cfg} {OUTPUT_DIR}/{task_name} {CHECKPOINT} 8 --metrics pa-mpjpe mpjpe pckh ' \
             + ' '.join(REST) + \
            f' 2>&1 | tee {OUTPUT_DIR}/{task_name}/log'
    print('running command: \n'+ cmd)
    os.system(cmd)

    # # visualization
    # cmd = f'CUDA_VISIBLE_DEVICES=0 bash tools/run_visualization.sh ' \
    #         f'{cfg} {OUTPUT_DIR}/{task_name} '
    # print('running command: \n'+ cmd)
    # os.system(cmd)