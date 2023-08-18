FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04
ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "--login", "-c"]

# Install basics
RUN rm /etc/apt/sources.list.d/cuda.list \
&& apt-get update \
&& apt-get install curl -y \
      && distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            tee /etc/apt/sources.list.d/nvidia-container-toolkit.list 

RUN apt-get -y update && apt-get install -y software-properties-common \
&& add-apt-repository -y ppa:jonathonf/ffmpeg-4 && apt-get upgrade -y \
&& apt-get update -y \
        && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        cmake \
        git \
        vim \
        tmux \
        wget \
        bzip2 \
        unzip \
        g++ \
        ca-certificates \
        ffmpeg \
        libx264-dev \
        imagemagick \
        libnss3-dev \
        locales \
        libsm6 \
        libxext6 \
        freeglut3-dev \
        openssh-client \
        libavformat-dev libavfilter-dev libavdevice-dev libswscale-dev libopus-dev libvpx-dev pkg-config 

RUN locale-gen en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

ENV CONDA_DIR /root/anaconda
ENV PATH /root/anaconda/bin:$PATH
ENV PATH /root/mambaforge/bin:$PATH

# RUN curl -sLo /root/anaconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh
# RUN chmod +x /root/anaconda.sh \
# && /root/anaconda.sh -b -p $CONDA_DIR \
# && rm /root/anaconda.sh

# Anaconda: https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh 

# Mamba
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh" \
&& bash Mambaforge-$(uname)-$(uname -m).sh -b

# Pytorch3D 0.4.0 env
RUN conda create --name 3dnbf python=3.8
SHELL ["conda", "run", "-n", "3dnbf", "/bin/bash", "-c"]
# install pytorch
RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
# install pytorch3d
RUN mamba install -y -c fvcore -c iopath -c conda-forge fvcore iopath && \
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.4.0"