import os
import imageio
import sys
sys.path.append('.')

import numpy as np
import torch
import torch.nn.functional as F
import joblib
from tqdm import tqdm
from src.visualize.anim import plot_3d_motion_dico, load_anim
import clip
from PIL import Image
import pickle
import src.utils.rotation_conversions as geometry
from textwrap import wrap
import shutil
import subprocess as sp
from copy import deepcopy

GPU_MINIMUM_MEMORY = 5500

def stack_images(real, real_gens, gen):
    nleft_cols = len(real_gens) + 1
    print("Stacking frames..")
    allframes = np.concatenate((real[:, None, ...], *[x[:, None, ...] for x in real_gens], gen), 1)
    nframes, nspa, nats, h, w, pix = allframes.shape
    blackborder = np.zeros((w//30, h*nats, pix), dtype=allframes.dtype)
    frames = []
    for frame_idx in tqdm(range(nframes)):
        columns = np.vstack(allframes[frame_idx].transpose(1, 2, 3, 4, 0)).transpose(3, 1, 0, 2)
        frame = np.concatenate((*columns[0:nleft_cols], blackborder, *columns[nleft_cols:]), 0).transpose(1, 0, 2)
        frames.append(frame)
    return np.stack(frames)

def stack_gen_and_images(gen, images):
    # nleft_cols = len(real_gens) + 1
    print("Stacking frames..")
    allframes = np.concatenate((images, gen), 2)
    nframes, nspa, nats, h, w, pix = allframes.shape
    blackborder = np.zeros((w//30, h*nats, pix), dtype=allframes.dtype)
    frames = []
    for frame_idx in tqdm(range(nframes)):
        columns = np.vstack(allframes[frame_idx].transpose(1, 2, 3, 4, 0)).transpose(3, 1, 0, 2)
        frame = np.concatenate((columns[:]), 0).transpose(1, 0, 2)
        frames.append(frame)
    return np.stack(frames)

def stack_gen_only(gen):
    # nleft_cols = len(real_gens) + 1
    print("Stacking frames..")
    # allframes = np.concatenate((real[:, None, ...], *[x[:, None, ...] for x in real_gens], gen), 1)
    allframes = gen
    nframes, nspa, nats, h, w, pix = allframes.shape
    blackborder = np.zeros((w//30, h*nats, pix), dtype=allframes.dtype)
    frames = []
    for frame_idx in tqdm(range(nframes)):
        columns = np.vstack(allframes[frame_idx].transpose(1, 2, 3, 4, 0)).transpose(3, 1, 0, 2)
        frame = np.concatenate((columns[:]), 0).transpose(1, 0, 2)
        frames.append(frame)
    return np.stack(frames)


def generate_by_video(visualization, reconstructions, generation,
                      label_to_action_name, params, nats, nspa, tmp_path, image_pathes=None, mode=None):
    # shape : (17, 3, 4, 480, 640, 3)
    # (nframes, row, column, h, w, 3)
    fps = params["fps"]

    params = params.copy()

    if "output_xyz" in visualization or "output_xyz" in generation:
        outputkey = "output_xyz"
        params["pose_rep"] = "xyz"
    else:
        outputkey = "poses"

    keep = [outputkey, "lengths", "y"]

    def _to_np(x):
        if type(x).__module__ == np.__name__:
            return x
        else:  # assume tensor
            return x.data.cpu().numpy()

    visu = {key: _to_np(visualization[key]) for key in keep if key in visualization.keys()}
    recons = {mode: {key: _to_np(reconstruction[key]) for key in keep if key in reconstruction.keys()}
              for mode, reconstruction in reconstructions.items()}
    gener = {key: _to_np(generation[key]) for key in keep if key in generation.keys()}

    def get_palette(i, nspa):
        if mode == 'edit' and i < 3:
            return 'orange'
        elif mode == 'interp' and i in [0, nspa-1]:
            return 'orange'
        return 'blue'


    if(len(visu) > 0):
        lenmax = max(gener["lengths"].max(),
                     visu["lengths"].max())
    else:
        lenmax = gener["lengths"].max()
    timesize = lenmax + 5
    # if params['appearance_mode'] == 'motionclip':
    #     timesize = lenmax + 20

    import multiprocessing

    def pool_job_with_desc(pool, iterator, desc, max_, save_path_format, isij):
        with tqdm(total=max_, desc=desc.format("Render")) as pbar:
            for _ in pool.imap_unordered(plot_3d_motion_dico, iterator):
                pbar.update()
        if isij:
            array = np.stack([[load_anim(save_path_format.format(i, j), timesize)
                               for j in range(nats)]
                              for i in tqdm(range(nspa), desc=desc.format("Load"))])
            return array.transpose(2, 0, 1, 3, 4, 5)
        else:
            array = np.stack([load_anim(save_path_format.format(i), timesize)
                              for i in tqdm(range(nats), desc=desc.format("Load"))])
            return array.transpose(1, 0, 2, 3, 4)

    with multiprocessing.Pool() as pool:
        # Generated samples
        save_path_format = os.path.join(tmp_path, "gen_{}_{}.gif")
        iterator = ((gener[outputkey][i, j],
                     gener["lengths"][i, j],
                     save_path_format.format(i, j),
                     # params, {"title": f"gen: {label_to_action_name(gener['y'][i, j])}", "interval": 1000/fps})
                     params, {"title": f"{label_to_action_name(gener['y'][i, j])}", "interval": 1000/fps, "palette": get_palette(i, nspa)})
                    for j in range(nats) for i in range(nspa))
        gener["frames"] = pool_job_with_desc(pool, iterator,
                                             "{} the generated samples",
                                             nats*nspa,
                                             save_path_format,
                                             True)

        # Make frames with no title blank
        frames_no_title = gener['y'] == ''
        gener["frames"][:, frames_no_title] = gener["frames"][:, 0, 0:1, 0:1, 0:1] # cast the corner pixel value for all blank box

        # Real samples
        if len(visu) > 0:
            save_path_format = os.path.join(tmp_path, "real_{}.gif")
            iterator = ((visu[outputkey][i],
                         visu["lengths"][i],
                         save_path_format.format(i),
                         params, {"title": f"real: {label_to_action_name(visu['y'][i])}", "interval": 1000/fps})
                        for i in range(nats))
            visu["frames"] = pool_job_with_desc(pool, iterator,
                                                "{} the real samples",
                                                nats,
                                                save_path_format,
                                                False)
        for mode, recon in recons.items():
            # Reconstructed samples
            save_path_format = os.path.join(tmp_path, f"reconstructed_{mode}_" + "{}.gif")
            iterator = ((recon[outputkey][i],
                         recon["lengths"][i],
                         save_path_format.format(i),
                         params, {"title": f"recons: {label_to_action_name(recon['y'][i])}",
                                  "interval": 1000/fps})
                        for i in range(nats))
            recon["frames"] = pool_job_with_desc(pool, iterator,
                                                 "{} the reconstructed samples",
                                                 nats,
                                                 save_path_format,
                                                 False)
    if image_pathes is not None:
        # visu["frames"] -> [timesize(65), nspa(n_samples), nats(1), h(290), w(260), n_ch(3)]
        assert nats == 1
        assert nspa == len(image_pathes)
        h, w = gener["frames"].shape[3:5]
        image_frames = []
        for im_path in image_pathes:
            im = Image.open(im_path).resize((w, h))
            image_frames.append(np.tile(np.expand_dims(np.asarray(im)[..., :3], axis=(0, 1, 2)), (timesize, 1, 1, 1, 1, 1)))
        image_frames = np.concatenate(image_frames, axis=1)
        assert image_frames.shape == gener["frames"].shape
        return stack_gen_and_images(gener["frames"], image_frames)

    if len(visu) == 0:
        frames = stack_gen_only(gener["frames"])
    else:
        frames = stack_images(visu["frames"], [recon["frames"] for recon in recons.values()], gener["frames"])
    return frames


def generate_by_video_sequences(visualization, label_to_action_name, params, nats, nspa, tmp_path):
    # shape : (17, 3, 4, 480, 640, 3)
    # (nframes, row, column, h, w, 3)
    fps = params["fps"]

    if "output_xyz" in visualization:
        outputkey = "output_xyz"
        params["pose_rep"] = "xyz"
    else:
        outputkey = "poses"

    keep = [outputkey, "lengths", "y"]
    visu = {key: visualization[key].data.cpu().numpy() for key in keep}
    lenmax = visu["lengths"].max()

    timesize = lenmax + 5
    import multiprocessing

    def pool_job_with_desc(pool, iterator, desc, max_, save_path_format):
        with tqdm(total=max_, desc=desc.format("Render")) as pbar:
            for _ in pool.imap_unordered(plot_3d_motion_dico, iterator):
                pbar.update()
        array = np.stack([[load_anim(save_path_format.format(i, j), timesize)
                           for j in range(nats)]
                          for i in tqdm(range(nspa), desc=desc.format("Load"))])
        return array.transpose(2, 0, 1, 3, 4, 5)

    with multiprocessing.Pool() as pool:
        # Real samples
        save_path_format = os.path.join(tmp_path, "real_{}_{}.gif")
        iterator = ((visu[outputkey][i, j],
                     visu["lengths"][i, j],
                     save_path_format.format(i, j),
                     params, {"title": f"real: {label_to_action_name(visu['y'][i, j])}", "interval": 1000/fps})
                    for j in range(nats) for i in range(nspa))
        visu["frames"] = pool_job_with_desc(pool, iterator,
                                            "{} the real samples",
                                            nats,
                                            save_path_format)
    frames = stack_images_sequence(visu["frames"])
    return frames


def viz_clip_text(model, text_grid, epoch, params, folder):
    """ Generate & viz samples """

    # visualize with joints3D
    model.outputxyz = True

    print(f"Visualization of the epoch {epoch}")

    # noise_same_action = params["noise_same_action"]
    # noise_diff_action = params["noise_diff_action"]

    fact = params["fact_latent"]
    figname = params["figname"].format(epoch)

    classes = np.array(text_grid, dtype=str)
    h, w = classes.shape

    texts = classes.reshape([-1])
    clip_tokens = clip.tokenize(texts).to(params['device'])
    clip_features = model.clip_model.encode_text(clip_tokens).float().unsqueeze(0)

    gendurations = torch.ones((h*w, 1), dtype=int) * params['num_frames']

    # generate the repr (joints3D/pose etc)
    model.eval()
    with torch.no_grad():

        generation = model.generate(clip_features, gendurations,
                                    is_amass=True,
                                    is_clip_features=True)
        generation['y'] = texts

        # Latent bundle for steering/visualization:
        # z_text_input is what conditions the decoder, z_motion_output is re-encoded generated motion.
        z_text_input = clip_features.reshape(-1, clip_features.shape[-1]).detach().cpu().numpy()
        z_motion_output = encode_motions_with_lengths(
            model,
            generation["output"],
            generation["lengths"],
            params['device']
        ).detach().cpu().numpy()
        text_latent_path = os.path.join(folder, "output_latents_text2motion.npz")
        np.savez(
            text_latent_path,
            z_text_input=z_text_input,
            z_motion_output=z_motion_output,
            lengths=generation["lengths"].detach().cpu().numpy(),
            texts=np.array(texts, dtype=object)
        )
        print(f"Saved text2motion latents to [{text_latent_path}]")
    
    # ----- ADD EXTRACTION HERE -----
    motions_tensor = generation['output_xyz'] if 'output_xyz' in generation else generation['poses']  # (B, 25, 6, T)
    motions_np = motions_tensor.detach().cpu().numpy()
    np.save(os.path.join(folder, "../../exps/paper-model/output.npy"), motions_np)
    # --------------------------------


    for key, val in generation.items():
        if len(generation[key].shape) == 1:
            generation[key] = val.reshape(h, w)
        else:
            generation[key] = val.reshape(h, w, *val.shape[1:])

    f_name = params['input_file']
    if os.path.isfile(params['input_file']):
        f_name = os.path.basename(params['input_file'].replace('.txt', ''))
    #finalpath = os.path.join(folder, 'clip_text_{}_{}'.format(f_name, 'trans_' if params['vertstrans'] else '') + figname + ".gif")
    finalpath = os.path.join(folder, "output.gif")
    tmp_path = os.path.join(folder, f"clip_text_subfigures_{figname}")
    os.makedirs(tmp_path, exist_ok=True)

    # save_pkl(generation['output'], generation['output_xyz'], texts, finalpath.replace('.gif', '.pkl'))

    print("Generate the videos..")
    frames = generate_by_video({}, {}, generation,
                               lambda x: str(x), params, w, h, tmp_path, mode='text')
    shutil.rmtree(tmp_path)
    print(f"Writing video [{finalpath}]")
    imageio.mimsave(finalpath, frames, fps=params["fps"], loop=0)

#added motion_collection and database as arguments
def viz_clip_interp(model, raw_dbs, motion_collection, interp_csv, num_stops, epoch, params, folder):
    """ Generate & viz samples """

    # visualize with joints3D
    model.outputxyz = True

    print(f"Visualization of the epoch {epoch}")
    figname = params["figname"].format(epoch)
    #motion_collection = get_motion_text_mapping(datasets)

    # prepare motion representations
    all_clip_features = []
    all_texts = []
    endpoint_z_motions = []
    text_pairs = []
    for line in interp_csv:
        # Get CLIP features
        texts = [line['start'], line['end']]
        retrieved_motions = retrieve_motions_raw(raw_dbs, motion_collection, texts, params['device'])
        clip_features = encode_motions(model, retrieved_motions, params['device'])
        endpoint_z_motions.append(clip_features.detach().cpu())
        text_pairs.append(texts)


        # Make interp
        end_factor = np.linspace(0., 1., num=num_stops)
        start_factor = 1. - end_factor
        interp_features = [(start_factor[i]*clip_features[0]) + (end_factor[i]*clip_features[1]) for i in range(num_stops)]
        all_clip_features.append(torch.stack(interp_features))
        texts = texts[:1] + [' '] * (num_stops-2) + texts[-1:]
        all_texts.append(texts)

    all_clip_features = torch.transpose(torch.stack(all_clip_features, axis=0), 0, 1)
    all_texts = np.array(all_texts).T
    h, w = all_clip_features.shape[:2]
    gendurations = torch.ones((h*w, 1), dtype=int) * params['num_frames']

    # generate the repr (joints3D/pose etc)
    model.eval()
    with torch.no_grad():
        generation = model.generate(all_clip_features, gendurations,
                                    is_amass=True,
                                    is_clip_features=True)
        generation['y'] = all_texts.reshape([-1])

        # Interpolation latent bundle (input-side only):
        # endpoint_z_motions: original motion latents for start/end pairs
        # z_interp_input: latent points fed to decoder
        z_interp_input = all_clip_features.reshape(-1, all_clip_features.shape[-1]).detach().cpu().numpy()
        interp_latent_path = os.path.join(folder, "output_latents_interpolation.npz")
        np.savez(
            interp_latent_path,
            z_motion_endpoints=torch.stack(endpoint_z_motions, dim=0).numpy(),
            z_interp_input=z_interp_input,
            lengths=generation["lengths"].detach().cpu().numpy(),
            text_pairs=np.array(text_pairs, dtype=object)
        )
        print(f"Saved interpolation latents to [{interp_latent_path}]")

    for key, val in generation.items():
        if len(generation[key].shape) == 1:
            generation[key] = val.reshape(h, w)
        else:
            generation[key] = val.reshape(h, w, *val.shape[1:])

    if os.path.isfile(params['input_file']):
        f_name = os.path.basename(params['input_file'].replace('.csv', ''))
    #finalpath = os.path.join(folder, f'clip_edit_{f_name}_' + figname + ".gif")
    finalpath = os.path.join(folder, "output.gif")
    tmp_path = os.path.join(folder, f"clip_edit_subfigures_{figname}")
    os.makedirs(tmp_path, exist_ok=True)

    print("Generate the videos..")
    frames = generate_by_video({}, {}, generation,
                               lambda x: str(x), params, w, h, tmp_path, mode='interp')
    shutil.rmtree(tmp_path)
    print(f"Writing video [{finalpath}]")
    imageio.mimsave(finalpath, frames, fps=params["fps"], loop=0)

def get_motion_text_mapping(datasets):
    print('Building text-motion mapping...')
    split_names = list(datasets.keys())
    collection_path = datasets[split_names[0]].datapath.replace('.pt', '_text_labels.txt')
    if len(split_names) > 1:
        assert split_names[0] in os.path.basename(collection_path)
        _base = os.path.basename(collection_path).replace(split_names[0], 'all')
        collection_path = os.path.join(os.path.dirname(collection_path), _base)
    cache_path = collection_path.replace('.txt', '.npy')

    # load if exists
    word = 'Loading' if os.path.isfile(cache_path) else 'Saving'
    print('{} list of text labels in current dataset to [{}]:'.format(word, collection_path))
    print('Look it up next time you want to retrieve new motions using textual labels.')

    if os.path.isfile(cache_path):
        return np.load(cache_path, allow_pickle=True)[None][0]

    motion_collection = {}
    for split_name, data in datasets.items():
        for i, d in tqdm(enumerate(data)):
            motion_collection[d['clip_text']] = motion_collection.get(d['clip_text'], []) + [(split_name, i)]

    with open(collection_path, 'w') as fw:
        text_labels = sorted(list(motion_collection.keys()))
        fw.write('\n'.join(text_labels) + '\n')
    np.save(cache_path, motion_collection)

    return motion_collection

def viz_clip_edit(model, raw_dbs, motion_collection, edit_csv, epoch, params, folder):
    # visualize with joints3D
    model.outputxyz = True

    print(f"Visualization of the epoch {epoch}")
    figname = params["figname"].format(epoch)
    #motion_collection = get_motion_text_mapping(datasets)

    # prepare motion representations
    all_clip_features = []
    all_texts = []
    for line in edit_csv:
        # Get CLIP features
        texts = [line['base'], line['v_start'], line['v_end']]
        if line['motion_source'] == 'data':
            retrieved_motions = retrieve_motions_raw(raw_dbs, motion_collection, texts, params['device'])
            clip_features = encode_motions(model, retrieved_motions, params['device'])
        elif line['motion_source'] == 'text':
            clip_tokens = clip.tokenize(texts).to(params['device'])
            clip_features = model.clip_model.encode_text(clip_tokens).float()
        else:
            raise ValueError

        # Make edit
        result_features = clip_features[0] - clip_features[1] + clip_features[2]
        all_clip_features.append(torch.cat([clip_features, result_features.unsqueeze(0)]))
        texts.append('Result')
        all_texts.append(texts)

    all_clip_features = torch.transpose(torch.stack(all_clip_features, axis=0), 0, 1)
    all_texts = np.array(all_texts).T
    h, w = all_clip_features.shape[:2]
    gendurations = torch.ones((h*w, 1), dtype=int) * params['num_frames']

    # generate the repr (joints3D/pose etc)
    model.eval()
    with torch.no_grad():
        generation = model.generate(all_clip_features, gendurations,
                                    is_amass=True,
                                    is_clip_features=True)
        generation['y'] = all_texts.reshape([-1])

    for key, val in generation.items():
        if len(generation[key].shape) == 1:
            generation[key] = val.reshape(h, w)
        else:
            generation[key] = val.reshape(h, w, *val.shape[1:])

    if os.path.isfile(params['input_file']):
        f_name = os.path.basename(params['input_file'].replace('.csv', ''))
    finalpath = os.path.join(folder, f'clip_edit_{f_name}_' + figname + ".gif")
    tmp_path = os.path.join(folder, f"clip_edit_subfigures_{figname}")
    os.makedirs(tmp_path, exist_ok=True)

    print("Generate the videos..")
    frames = generate_by_video({}, {}, generation,
                               lambda x: str(x), params, w, h, tmp_path, mode='edit')
    #shutil.rmtree(tmp_path)
    print(f"Writing video [{finalpath}]")
    imageio.mimsave(finalpath, frames, fps=params["fps"], loop=0)


def stack_images_sequence(visu):
    print("Stacking frames..")
    allframes = visu
    nframes, nspa, nats, h, w, pix = allframes.shape
    frames = []
    for frame_idx in tqdm(range(nframes)):
        columns = np.vstack(allframes[frame_idx].transpose(1, 2, 3, 4, 0)).transpose(3, 1, 0, 2)
        frame = np.concatenate(columns).transpose(1, 0, 2)
        frames.append(frame)
    return np.stack(frames)

def get_gpu_device():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    for gpu_idx, free_mem in enumerate(memory_free_values):
        #if free_mem > GPU_MINIMUM_MEMORY: #FIXME
            return gpu_idx
    #raise Exception('No GPU with required memory')
"""
def get_motion_text_mapping(datasets):
    print('Building text-motion mapping...')
    split_names = list(datasets.keys())
    collection_path = datasets[split_names[0]].datapath.replace('.pt', '_text_labels.txt')
    if len(split_names) > 1:
        assert split_names[0] in os.path.basename(collection_path)
        _base = os.path.basename(collection_path).replace(split_names[0], 'all')
        collection_path = os.path.join(os.path.dirname(collection_path), _base)
    cache_path = collection_path.replace('.txt', '.npy')

    # load if exists
    word = 'Loading' if os.path.isfile(cache_path) else 'Saving'
    print('{} list of text labels in current dataset to [{}]:'.format(word, collection_path))
    print('Look it up next time you want to retrieve new motions using textual labels.')

    if os.path.isfile(cache_path):
        return np.load(cache_path, allow_pickle=True)[None][0]

    motion_collection = {}
    for split_name, data in datasets.items():
        for i, d in tqdm(enumerate(data)):
            motion_collection[d['clip_text']] = motion_collection.get(d['clip_text'], []) + [(split_name, i)]

    with open(collection_path, 'w') as fw:
        text_labels = sorted(list(motion_collection.keys()))
        fw.write('\n'.join(text_labels) + '\n')
    np.save(cache_path, motion_collection)

    return motion_collection
"""
SEQ_LEN = 100

SEQ_LEN = 100

def normalize_text_label(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore").strip()
    return str(x).strip()

def retrieve_motions_raw(raw_dbs, motion_collection, texts, device, seq_len=SEQ_LEN):
    retrieved_motions = []

    for txt in texts:
        key = normalize_text_label(txt)
        split_name, seq_idx, sub_idx = motion_collection[key][0]

        db = raw_dbs[split_name]

        start = sub_idx * seq_len
        end = start + seq_len

        # raw AMASS data
        joints3d = np.asarray(db["joints3d"][seq_idx][start:end]).copy()   # (T, 24, 3)
        thetas = np.asarray(db["thetas"][seq_idx][start:end]).copy()       # (T, 72)

        # --- translation branch ---
        # same logic as Dataset._load():
        joints3d = joints3d - joints3d[0, 0, :]
        ret_tr = torch.from_numpy(joints3d[:, 0, :]).float()               # (T, 3)

        # --- pose branch ---
        pose = thetas.reshape(seq_len, 24, 3)                              # (T, 24, 3)
        pose = torch.from_numpy(pose).float()

        # rotvec -> rot6d
        ret = geometry.matrix_to_rotation_6d(
            geometry.axis_angle_to_matrix(pose)
        )                                                                  # (T, 24, 6)

        # add translation as extra joint
        padded_tr = torch.zeros((ret.shape[0], ret.shape[2]), dtype=ret.dtype)
        padded_tr[:, :3] = ret_tr
        ret = torch.cat((ret, padded_tr[:, None]), dim=1)                  # (T, 25, 6)

        # --- SAVE joints_vec here ---
        motions_np = ret.detach().cpu().numpy()      # (T, 25, 6)
        #print(os.getcwd())
        np.save(f"./exps/paper-model/output.npy", motions_np)    # save one file per text label

        # final model input layout
        ret = ret.permute(1, 2, 0).contiguous().float()                    # (25, 6, T)

        retrieved_motions.append(ret.unsqueeze(0).to(device))              # (1, 25, 6, T)

    return torch.cat(retrieved_motions, dim=0)                             # (B, 25, 6, T)

def encode_motions(model, motions, device):
    lengths = torch.ones(motions.shape[0], dtype=torch.long, device=device) * motions.shape[-1]
    return encode_motions_with_lengths(model, motions, lengths, device)


def encode_motions_with_lengths(model, motions, lengths, device):
    lengths = lengths.to(device=device, dtype=torch.long)
    mu = model.encoder({
        'x': motions,
        'y': torch.zeros(motions.shape[0], dtype=torch.long, device=device),
        'mask': model.lengths_to_mask(lengths)
    })["mu"]
    #np.save("motionclip_embeddings.npy", mu.detach().cpu().numpy())
    return mu
