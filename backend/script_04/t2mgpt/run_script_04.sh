#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate T2M-GPT

python run_t2mgpt.py "$@"