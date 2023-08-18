# Install dependencies and mmhuman3d
pip install -r requirements/req.txt
python setup.py develop

# # (Temporary) Fix pytorch3d>=0.5.0 bug
# # vim /pytorch3d/pytorch3d/renderer/cameras.py
# # go to l353, remove **kwargs
# cp docker/pytorch3d/cameras.py /root/mambaforge/envs/3dnbf/lib/python3.8/site-packages/pytorch3d/renderer/cameras.py

# Replace with modified epoch_based_runner.py in mmcv, e.g.
cp docker/mmcv/runner/epoch_based_runner.py /root/mambaforge/envs/3dnbf/lib/python3.8/site-packages/mmcv-1.5.0-py3.8.egg/mmcv/runner/

# Download data.zip from https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/yzhan286_jh_edu/Eejiw2J_dJhFp0L-tQGuCPcBP6LNVOKaTIRn-pH4zWIHFg?e=lYoSqh
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