#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate motionclip

python run_motionclip.py "$@"