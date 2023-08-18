conda create --name 3dnbf python=3.8
conda activate 3dnbf

# conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
conda install -c fvcore -c iopath -c conda-forge fvcore iopath && \
# conda install pytorch3d=0.4.0 -c pytorch3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.4.0"

pip install -r requirements/req.txt
python setup.py develop

# # (Temporary) Fix pytorch3d>=0.5.0 bug
# # vim /PATH/TO/pytorch3d/renderer/cameras.py
# # go to l353, remove **kwargs
# cp docker/pytorch3d/cameras.py /PATH/TO/pytorch3d/renderer/cameras.py

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
git clone https://github.com/edz-o/VoGE-3DNBF.git
cd VoGE-3DNBF && python setup.py develop
cd ../../

#Install pytorch_openpose_body_25
cd third_party
git clone https://github.com/edz-o/pytorch_openpose_body_25.git
cd pytorch_openpose_body_25 && bash download_models.sh
python setup.py develop
cd ../../