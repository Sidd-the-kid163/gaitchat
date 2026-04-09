#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mgpt

python run_mgpt.py "$@"