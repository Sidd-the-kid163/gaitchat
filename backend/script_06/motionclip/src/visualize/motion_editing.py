import os, psutil
import sys
sys.path.append('.')
from src.utils.get_model_only import get_model_only

import matplotlib.pyplot as plt
import torch
import joblib
import csv
import numpy as np
from src.utils.get_model_and_data import get_model_and_data
from src.parser.visualize import parser
from src.visualize.visualize import viz_clip_edit, get_gpu_device
from src.utils.misc import load_model_wo_clip

import src.utils.fixseed  # noqa

plt.switch_backend('agg')

SEQ_LEN = 100

def load_raw_amass_dbs(base_path="data/amass/amass_30fps_legacy_db.pt", splits=("train", "vald")):
    dbs = {}
    for split in splits:
        db_path = base_path.replace("_db.pt", f"_{split}.pt")
        dbs[split] = joblib.load(db_path, mmap_mode="r")
    return dbs

def load_motion_collection(cache_path):
    return np.load(cache_path, allow_pickle=True).item()

def main():
    parameters, folder, checkpointname, epoch = parser()
    gpu_device = get_gpu_device()
    parameters["device"] = f"cuda:{gpu_device}"

    assert os.path.isfile(parameters['input_file'])
    with open(parameters['input_file'], 'r') as fr:
        edit_csv = list(csv.DictReader(fr))

    model = get_model_only(parameters)

    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location="cpu")
    load_model_wo_clip(model, state_dict)
    model = model.to(parameters["device"])
    model.eval()

    need_data = any(line['motion_source'] == 'data' for line in edit_csv)

    raw_dbs = None
    motion_collection = None

    if need_data:
        base_path = "data/amass_db/amass_30fps_db.pt"
        cache_path = base_path.replace("_db.pt", "_all_text_labels.npy")

        raw_dbs = load_raw_amass_dbs(base_path=base_path, splits=("train", "vald"))
        motion_collection = load_motion_collection(cache_path)

    with torch.inference_mode():
        viz_clip_edit(model, raw_dbs, motion_collection, edit_csv, epoch, parameters, folder=folder)

if __name__ == '__main__':
    main()
