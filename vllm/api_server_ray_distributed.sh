#!/bin/bash

set -e

MODEL="facebook/opt-13b"

HOST="localhost"
PORT="8000"

PY_VER="3.10"
DEV_VENV="vllm_py""$PY_VER"

conda activate "$DEV_VENV"

python -m vllm.entrypoints.api_server \
--model "$MODEL" --host "$HOST" --port "$PORT" \
--tensor-parallel-size 2
