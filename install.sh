#!/bin/bash
set -e
cd "$(dirname "$0")"

# Install Miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash

# Create & activate the Conda environment
~/miniconda3/bin/conda create -n dm-env python=3.11 -y
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dm-env
uv pip install -r requirements.txt
pip install "mamba-ssm[causal-conv1d]"==2.2.4 --no-build-isolation
conda install ipykernel --update-deps --force-reinstall -y
conda deactivate