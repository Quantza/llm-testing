#!/bin/bash

set -e

PY_VER="3.10"
DEV_VENV="vllm_py""$PY_VER"

conda create -n "$DEV_VENV" python=$PY_VER -y
conda activate "$DEV_VENV"

pip install vllm ray openai skypilot

# Sources: https://vllm.readthedocs.io/en/latest/serving/distributed_serving.html 
#          https://vllm.readthedocs.io/en/latest/serving/run_on_sky.html

# From source
#docker run --gpus all -it --rm --ipc=host nvcr.io/nvidia/pytorch:22.12-py3
#git clone https://github.com/vllm-project/vllm.git
#cd vllm
#pip install -e .

