import os
import sys
sys.path.append('.')
from src.utils.get_model_only import get_model_only
import csv
import joblib
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.utils.get_model_and_data import get_model_and_data
from src.parser.visualize import parser
from src.visualize.visualize import viz_clip_interp, get_gpu_device
from src.utils.misc import load_model_wo_clip


import src.utils.fixseed  # noqa

plt.switch_backend('agg')

def load_raw_amass_dbs(base_path="data/amass/amass_30fps_legacy_db.pt", splits=("train", "vald")):
    dbs = {}
    for split in splits:
        db_path = base_path.replace("_db.pt", f"_{split}.pt")
        dbs[split] = joblib.load(db_path, mmap_mode="r")
    return dbs

def load_motion_collection(cache_path):
    return np.load(cache_path, allow_pickle=True).item()

def main():
    # parse options
    parameters, folder, checkpointname, epoch = parser()
    gpu_device = get_gpu_device()
    parameters["device"] = "cpu" if gpu_device is None else f"cuda:{gpu_device}"
    parameters["device"] = f"cuda:{gpu_device}"
    #model, datasets = get_model_and_data(parameters, split='all')
    model = get_model_only(parameters)
    base_path = "data/amass_db/amass_30fps_db.pt"
    cache_path = base_path.replace("_db.pt", "_all_text_labels.npy")
    raw_dbs = load_raw_amass_dbs(base_path=base_path, splits=("train", "vald"))
    motion_collection = load_motion_collection(cache_path)
    num_stops = 5  # FIXME - hardcoded

    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=parameters["device"])
    load_model_wo_clip(model, state_dict)

    assert os.path.isfile(parameters['input_file'])
    with open(parameters['input_file'], 'r') as fr:
        interp_csv = list(csv.DictReader(fr))
    viz_clip_interp(model, raw_dbs, motion_collection, interp_csv, num_stops, epoch, parameters, folder=folder)

    latent_path = os.path.join(folder, "output_latents_interpolation.npz")
    if os.path.isfile(latent_path):
        latent_bundle = np.load(latent_path, allow_pickle=True)
        z_motion_endpoints = latent_bundle["z_motion_endpoints"]
        z_interp_input = latent_bundle["z_interp_input"]
        """
        print(f"Latents saved to [{latent_path}]")
        print(f"z_motion_endpoints shape: {z_motion_endpoints.shape}")
        print(f"z_interp_input shape: {z_interp_input.shape}")
        if z_motion_endpoints.size > 0:
            print(f"z_motion_endpoints[0, 0, :8]: {z_motion_endpoints[0, 0, :8]}")
        if z_interp_input.size > 0:
            print(f"z_interp_input[0, :8]: {z_interp_input[0, :8]}")
        """
if __name__ == '__main__':
    main()
