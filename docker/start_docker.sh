docker run \
    --gpus all \
    --shm-size 32g \
    --ipc=host \
    -it \
    -v /home:/home \
    -v /mnt:/mnt \
    3dnbf:py38 \
    /bin/bash
