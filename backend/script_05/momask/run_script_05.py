#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate momask

python run_momask.py "$@"