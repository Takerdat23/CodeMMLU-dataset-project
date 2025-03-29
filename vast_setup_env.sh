#!/bin/bash

touch ~/.no_auto_tmux

conda env create -f environment.yml

conda init

source /opt/conda/etc/profile.d/conda.sh

conda activate code_mmlu

apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

##### Run this after activate the environment #####
'
pip uninstall bitsandbytes -y
pip install --upgrade bitsandbytes
pip install h5py
pip install flash-attn --no-build-isolation
pip uninstall deepspeed -y
pip install deepspeed==0.15.4
'
