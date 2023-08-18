CONFIG=$1
OUT_DIR=$2

python tools/visualize_predictions.py --config $CONFIG \
                    --output_file ${OUT_DIR}/result_keypoints.json \
                    --outdir ${OUT_DIR}/visualization ${@:3}