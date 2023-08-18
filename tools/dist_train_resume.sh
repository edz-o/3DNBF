CONFIG=$1
WORK_DIR=$2
# RESUME_FROM=$3
GPUS=$3
PORT=29500 #${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py --config=$CONFIG --work-dir=${WORK_DIR} --resume-from=${RESUME_FROM} --launcher pytorch ${@:4}
