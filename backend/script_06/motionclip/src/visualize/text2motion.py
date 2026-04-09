import os
import sys
from src.utils.get_model_only import get_model_only
sys.path.append('.')

import matplotlib.pyplot as plt
import numpy as np
import torch
from src.utils.get_model_and_data import get_model_and_data
from src.parser.visualize import parser
from src.visualize.visualize import viz_clip_text, get_gpu_device
from src.utils.misc import load_model_wo_clip
import src.utils.fixseed  # noqa

plt.switch_backend('agg')


def main():
    # parse options
    parameters, folder, checkpointname, epoch = parser()
    gpu_device = get_gpu_device()
    parameters["device"] = "cpu" if gpu_device is None else f"cuda:{gpu_device}"
    parameters["device"] = f"cuda:{gpu_device}"
    #model, datasets = get_model_and_data(parameters, split='vald')
    model = get_model_only(parameters)

    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=parameters["device"])
    load_model_wo_clip(model, state_dict)

    assert os.path.isfile(parameters['input_file'])
    with open(parameters['input_file'], 'r') as fr:
        texts = [s.strip() for s in fr if s.strip()]
    # Make a nearly square grid for compactness; check orig code if you want a different layout (e.g. single column)
    import math
    n = len(texts)
    if n == 0:
        grid = [[]]
    else:
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        grid = [texts[i*cols:(i+1)*cols] for i in range(rows)]
        # No padding: only pad the last row if needed
        if len(grid[-1]) < cols:
            grid[-1] += [''] * (cols - len(grid[-1]))

    viz_clip_text(model, grid, epoch, parameters, folder=folder)

    latent_path = os.path.join(folder, "output_latents_text2motion.npz")
    if os.path.isfile(latent_path):
        latent_bundle = np.load(latent_path, allow_pickle=True)
        z_text_input = latent_bundle["z_text_input"]
        z_motion_output = latent_bundle["z_motion_output"]
        """
        print(f"Latents saved to [{latent_path}]")
        print(f"z_text_input shape: {z_text_input.shape}")
        print(f"z_motion_output shape: {z_motion_output.shape}")
        if z_text_input.size > 0:
            print(f"z_text_input[0, :8]: {z_text_input[0, :8]}")
        if z_motion_output.size > 0:
            print(f"z_motion_output[0, :8]: {z_motion_output[0, :8]}")
        """

if __name__ == '__main__':
    main()
