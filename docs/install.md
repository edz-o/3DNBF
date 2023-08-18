# Installation

<!-- TOC -->

- [Requirements](#requirements)
- [Prepare environment](#prepare-environment)
- [Install MMHuman3D](#install-mmhuman3d)
- [A from-scratch setup script](#a-from-scratch-setup-script)

<!-- TOC -->

## Requirements

- Linux
- ffmpeg
- Python 3.7+
- PyTorch >=1.7.0 (tested on 1.7.1)
- CUDA 9.2+
- GCC 5+
- PyTorch3D 0.4.0
- [MMCV](https://github.com/open-mmlab/mmcv)==1.5.0

## Prepare environment

a. Install ffmpeg

Install ffmpeg with conda directly and the libx264 will be built automatically.

```shell
conda install ffmpeg
```

b. Create a conda virtual environment and activate it.

```shell
conda create -n 3dnbf python=3.8 -y
conda activate 3dnbf
```

c. Install PyTorch 1.7.1 and torchvision following the [official instructions](https://pytorch.org/).
```shell
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

**Important:** Make sure that your compilation CUDA version and runtime CUDA version match.

d. Install PyTorch3D 0.4.0 and dependency libs.

```shell
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y

pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.4.0"
```
Please refer to [PyTorch3D-install](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for details.

Your installation is successful if you can do these in command line.

```shell
echo "import pytorch3d;print(pytorch3d.__version__); \
    from pytorch3d.renderer import MeshRenderer;print(MeshRenderer);\
    from pytorch3d.structures import Meshes;print(Meshes);\
    from pytorch3d.renderer import cameras;print(cameras);\
    from pytorch3d.transforms import Transform3d;print(Transform3d);"|python

echo "import torch;device=torch.device('cuda');\
    from pytorch3d.utils import torus;\
    Torus = torus(r=10, R=20, sides=100, rings=100, device=device);\
    print(Torus.verts_padded());"|python
```

## Install 3DNBF

Install build requirements and then install mmhuman3d.

```shell
pip install -r requirements/req.txt
pip install -v -e .  # or "python setup.py develop"

# Replace with modified epoch_based_runner.py in mmcv, e.g.
cp docker/mmcv/runner/epoch_based_runner.py /root/mambaforge/envs/3dnbf/lib/python3.8/site-packages/mmcv-1.5.0-py3.8.egg/mmcv/runner/
```

### Install Dependencies

Please refer to `install.sh` for installation. 

### Install Vposer
https://github.com/nghorbani/human_body_prior
Use `cvpr19` branch
```
mkdir third_party && cd third_party
git clone https://github.com/nghorbani/human_body_prior.git
cd human_body_prior
git checkout cvpr19
```

```python
# Replace three lines of `install_requires=...` in `setup.py` with
install_requires=['torch>=1.1.0', 'tensorboardX>=1.6', 'torchgeometry', 'opencv-python','configer>=1.4',
                        'configer', 'imageio', 'transforms3d', 'trimesh',
                        'smplx', 'pyrender', 'moviepy'],
```
Then run,
```
python setup.py develop
```

### Install VoGE
```
cd third_party
git clone https://github.com/edz-o/VoGE-3DNBF
cd VoGE-3DNBF && git checkout sampling && python setup.py develop
```

### Install pytorch_openpose_body_25

```
cd third_party
git clone git@github.com:edz-o/pytorch_openpose_body_25.git
cd pytorch_openpose_body_25 && bash download_models.sh
python setup.py develop
```

### Docker installation

Please use `docker/Dockerfile` and take a look at `docker/install_rest.sh`. 

## A from-scratch setup script

We provide a script for installation `install.sh`. NOTE: You still need to manually do some parts in it, e.g. downloading [data.zip](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/yzhan286_jh_edu/EfdBj5u9u_lPiVSQzcBhHdwBRsDyjk1xET5hFYKTGzOf5w?e=DRecfS) and replacing file in `mmcv`. 

```shell
conda create --name 3dnbf python=3.8
conda activate 3dnbf

# conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
conda install -c fvcore -c iopath -c conda-forge fvcore iopath && \
# conda install pytorch3d=0.4.0 -c pytorch3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.4.0"

pip install -r requirements/req.txt
python setup.py develop

# Replace with modified epoch_based_runner.py in mmcv, e.g.
cp docker/mmcv/runner/epoch_based_runner.py /root/mambaforge/envs/3dnbf/lib/python3.8/site-packages/mmcv-1.5.0-py3.8.egg/mmcv/runner/

# Download data.zip from https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/yzhan286_jh_edu/EfdBj5u9u_lPiVSQzcBhHdwBRsDyjk1xET5hFYKTGzOf5w?e=DRecfS
unzip data.zip

# Get dtd textures
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -zxf dtd-r1.0.1.tar.gz
mv dtd data
python docker/split_dtd.py
rm dtd-r1.0.1.tar.gz

# Install VPoser
mkdir third_party && cd third_party
git clone https://github.com/nghorbani/human_body_prior.git
cd human_body_prior
git checkout cvpr19
# (IMPORTANT) remove torch-1.1.0 dependency from setup.py
python setup.py develop
cd ../../

# Install VoGE
cd third_party
git clone git@github.com:edz-o/VoGE-3DNBF.git
cd VoGE-3DNBF && git checkout sampling && python setup.py develop
cd ../../

#Install pytorch_openpose_body_25
cd third_party
git clone git@github.com:edz-o/pytorch_openpose_body_25.git
cd pytorch_openpose_body_25 && bash download_models.sh
python setup.py develop
cd ../../
```
