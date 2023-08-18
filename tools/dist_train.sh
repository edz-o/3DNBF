CONFIG=$1
WORK_DIR=$2
GPUS=$3
PORT=29536 #${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py --config=$CONFIG --work-dir=${WORK_DIR} --launcher pytorch ${@:4} 2>&1 | tee ${WORK_DIR}/train.log
