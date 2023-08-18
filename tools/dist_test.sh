CONFIG=$1
WORK_DIR=$2
CHECKPOINT=$3
GPUS=$4
PORT=29536 #${PORT:-29500}

mkdir ${WORK_DIR}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py --config=$CONFIG --work-dir=${WORK_DIR} --checkpoint=$CHECKPOINT --launcher pytorch ${@:5} \
    2>&1 | tee ${WORK_DIR}/log
cp $CONFIG ${WORK_DIR}
