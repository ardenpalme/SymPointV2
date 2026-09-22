#!/bin/bash
sudo apt update
sudo apt-get install tmux
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh --output-document ~/Miniforge3-Linux-x86_64.sh
chmod +x  ~/Miniforge3-Linux-x86_64.sh && bash /home/ubuntu/Miniforge3-Linux-x86_64.sh
source ~/miniforge3/etc/profile.d/conda.sh

# export TORCH_CUDA_ARCH_LIST=`python -c "import torch; print('.'.join(map(str,torch.cuda.get_device_capability())))"`

conda init
conda create -n spv2 python=3.9 -y && conda activate spv2
conda install -y -c conda-forge ninja
conda install -y -c nvidia/label/cuda-12.8.0 cuda-toolkit
pip install torch torchvision torchaudio
pip install gdown mmcv==0.2.14 svgpathtools==1.6.1 munch==2.5.0 tensorboard==2.12.0 tensorboardx>=2.17
pip install wandb

# Detectron 2.0
git clone https://github.com/facebookresearch/detectron2.git && cd detectron2 
python -m pip install .
cd ..

# Pointops - May not be necessary?
cd modules/pointops
python setup.py install

cd ..
sudo echo "conda activate spv2" >> ~/.bashrc
