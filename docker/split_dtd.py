import os
import shutil

dtd_root = 'data/dtd'
splits = [('train', 'docker/dtd_splits/train.txt'), ('test', 'docker/dtd_splits/test.txt')]

for split, flist in splits:
    folders = [x.strip() for x in open(flist).readlines()]
    dst = os.path.join(dtd_root, split, 'images')
    os.makedirs(dst, exist_ok=True)
    for fd in folders:
        src = os.path.join(dtd_root, 'images', fd)
        shutil.copytree(src, os.path.join(dst, fd))