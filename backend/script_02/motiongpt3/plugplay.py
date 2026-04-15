import imageio
##import gradio as gr
# import random
import torch
import sys
import time
# import cv2
import os
import numpy as np
import pickle
import io
import pytorch_lightning as pl
import moviepy.editor as mp
from pathlib import Path
from motGPT.data.build_data import build_data
from motGPT.models.build_model import build_model
from motGPT.config import parse_args
from scipy.spatial.transform import Rotation as RRR
import motGPT.render.matplot.plot_3d_global as plot_3d
from motGPT.render.pyrender.hybrik_loc2rot import HybrIKJointsToRotmat
from motGPT.render.pyrender.smpl_render import SMPLRender
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")) #for databaseaccess
import databaseaccess as db

conn, cursor = db.connect()
db.init(conn, cursor)
userid = None

##os.environ['DISPLAY'] = ':0.0'
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
motion_uploaded = {
        "feats": None,
        "joints": None,
        "motion_video": None,
        "motion_lengths": 0,
        "motion_token": None,
        "motion_token_string": '',
        "motion_token_length": 0,
        "motion_tokens_input": None,
        "file": None,
    }
data_stored = [{
    "user_input": None,
    "motion_uploaded": {
        "feats": None,
        "joints": None,
        "motion_video": None,
        "motion_lengths": 0,
        "motion_token": None,
        "motion_token_string": '',
        "motion_token_length": 0,
        "motion_tokens_input": None,
        "file": None,
        "motion_video_fname": None,
        "motion_joints": None,
        "motion_joints_fname": None
    },
    "model_input": None,
    "model_output": {
        "feats": None,
        "joints": None,
        "length": 0,
        "texts": None,
        "motion_video": None,
        "motion_video_fname": None,
        "motion_joints": None,
        "motion_joints_fname": None
    }
}]
task = ""
txt = ""
file = None

# Load model
cfg = parse_args(phase="webui")  # parse config file
cfg.FOLDER = 'cache'
output_dir = Path(cfg.FOLDER)
output_dir.mkdir(parents=True, exist_ok=True)
pl.seed_everything(cfg.SEED_VALUE)
if cfg.ACCELERATOR == "gpu":
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
datamodule = build_data(cfg, phase="test")
model = build_model(cfg, datamodule).eval()
state_dict = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
model.load_state_dict(state_dict)
model.to(device)

#audio_processor = WhisperProcessor.from_pretrained(cfg.model.whisper_path)
#audio_model = WhisperForConditionalGeneration.from_pretrained(cfg.model.whisper_path).to(device)
#forced_decoder_ids = audio_processor.get_decoder_prompt_ids(language="zh", task="translate")
#forced_decoder_ids_zh = audio_processor.get_decoder_prompt_ids(language="zh", task="translate")
#forced_decoder_ids_en = audio_processor.get_decoder_prompt_ids(language="en", task="translate")

# task = 't2m'
def motion_token_to_string(motion_token, lengths, codebook_size=512):
    motion_string = []
    for i in range(motion_token.shape[0]):
        motion_i = motion_token[i].cpu(
        ) if motion_token.device.type == 'cuda' else motion_token[i]
        motion_list = motion_i.tolist()[:lengths[i]]
        motion_string.append(
            (f'<motion_id_{codebook_size}>' +
             ''.join([f'<motion_id_{int(i)}>' for i in motion_list]) +
             f'<motion_id_{codebook_size + 1}>'))
    return motion_string

def render_motion(data, feats, method='fast'):
    fname = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(
        time.time())) + str(np.random.randint(10000, 99999))
    video_fname = fname + '.mp4'
    feats_fname = fname + '.npy'
    output_npy_path = os.path.join(output_dir, feats_fname)
    output_mp4_path = os.path.join(output_dir, video_fname)
    np.save(output_npy_path, feats)

    if method == 'slow':
        if len(data.shape) == 4:
            data = data[0]
        data = data - data[0, 0]
        pose_generator = HybrIKJointsToRotmat()
        pose = pose_generator(data)
        pose = np.concatenate([
            pose,
            np.stack([np.stack([np.eye(3)] * pose.shape[0], 0)] * 2, 1)
        ], 1)
        shape = [768, 768]
        render = SMPLRender(cfg.RENDER.SMPL_MODEL_PATH)

        r = RRR.from_rotvec(np.array([np.pi, 0.0, 0.0]))
        pose[:, 0] = np.matmul(r.as_matrix().reshape(1, 3, 3), pose[:, 0])
        vid = []
        aroot = data[:, 0].copy()
        aroot[:, 1] = -aroot[:, 1]
        aroot[:, 2] = -aroot[:, 2]
        params = dict(pred_shape=np.zeros([1, 10]),
                      pred_root=aroot,
                      pred_pose=pose)
        render.init_renderer([shape[0], shape[1], 3], params)
        for i in range(data.shape[0]):
            renderImg = render.render(i)
            vid.append(renderImg)

        # out = np.stack(vid, axis=0)
        out_video = mp.ImageSequenceClip(vid, fps=model.fps)
        out_video.write_videofile(output_mp4_path,fps=model.fps)
        del render

    elif method == 'fast':
        output_gif_path = output_mp4_path[:-4] + '.gif'
        if len(data.shape) == 3:
            data = data[None]
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        pose_vis = plot_3d.draw_to_batch(data, [''], [output_gif_path])
        out_video = mp.VideoFileClip(output_gif_path)
        out_video.write_videofile(output_mp4_path,codec="libx264",audio=False,ffmpeg_params=["-pix_fmt", "yuv420p"])
        del pose_vis

    print('render_motion')
    return output_mp4_path, video_fname, output_npy_path, feats_fname


def motion_feats_to_tokens(motion_feat, motion_encode_net, length=None):
    if length is None:
        length = motion_feat.shape[-2]
    dist = motion_encode_net.encode_dist(motion_feat.to(motion_feat.device), [length])
    z, _ = motion_encode_net.encode_dist2z(dist)
    motion_token_input = z.permute(1,0,2).mul_(motion_encode_net.mean_std_inv)
    return motion_token_input


def load_motion(motion_uploaded, method, dataset_id):
    #file = motion_uploaded['file']
    #file = os.path.join('cache', motion_uploaded['file'])
    feats = torch.tensor(db.change_tonpy(motion_uploaded['file'], dataset_id), device=model.device)
    if len(feats.shape) == 2:
        feats = feats[None]
    motion_feat = model.datamodule.normalize(feats)
    motion_length = feats.shape[0]

    # # Motion encoding
    # motion_tokens_input, _ = model.vae.encode(motion_feat.unsqueeze(0), motion_length)
    motion_tokens_input = motion_feats_to_tokens(motion_feat, model.vae, motion_length)


    # motion_token_string = model.lm.motion_token_to_string(
    #     motion_token, [motion_token.shape[1]])[0]
    # motion_token_length = motion_token.shape[1]

    # Motion rendered
    joints = model.datamodule.feats2joints(feats.cpu()).cpu().numpy()
    output_mp4_path, video_fname, output_npy_path, joints_fname = render_motion(
        joints,
        feats.to('cpu').numpy(), method)

    motion_uploaded.update({
        "feats": feats,
        "joints": joints,
        "motion_video": output_mp4_path,
        "motion_video_fname": video_fname,
        "motion_joints": output_npy_path,
        "motion_joints_fname": joints_fname,
        "motion_lengths": motion_length,
        "motion_tokens_input": motion_tokens_input,
        # "motion_token_string": motion_token_string,
        # "motion_token_length": motion_token_length,
    })

    print('load_motion')
    return motion_uploaded

def submit(motion_uploaded, data_stored, file, txt, task, dataset_id):
    method = 'fast' # no SMPL
    if file is not None:
        motion_uploaded['file'] = getattr(file, 'name', file)
        motion_uploaded = load_motion(motion_uploaded, method, dataset_id)
        txt = txt.replace(" <Motion_Placeholder>", "") + " <Motion_Placeholder>"
    data_stored = data_stored + [{
        'user_input': txt,
        'motion_uploaded': motion_uploaded.copy() if hasattr(motion_uploaded, 'copy') else dict(motion_uploaded),
        'model_input': None,
        'model_output': {
            'feats': None,
            'joints': None,
            'length': 0,
            'texts': None,
            'motion_video': None,
            'motion_video_fname': None,
            'motion_joints': None,
            'motion_joints_fname': None
        }
    }]
    while len(data_stored) > 20: #capping data_stored length to 21 for memory control
        data_stored.pop(0)
    return bot(motion_uploaded, data_stored, method, task)

def bot(motion_uploaded, data_stored, method, task='t2m'):
    motion_length = motion_uploaded["motion_lengths"]
    motion_tokens_input = motion_uploaded['motion_tokens_input']

    input = data_stored[-1]['user_input']
    prompt = model.lm.placeholder_fulfill(input, motion_length,
                                          model.lm.input_motion_holder_seq, "")
    data_stored[-1]['model_input'] = prompt
    batch = {
        "length": [motion_length],
        "text": [prompt],
        "motion_tokens_input": [motion_tokens_input] if motion_tokens_input is not None else None,
        "feats_ref": motion_uploaded['feats'],
    }

    # print('task', task)
    outputs = model(batch, task=task)
    if task in ['t2t', 'm2t']:
        out_texts = outputs["texts"][0]
        # print(out_texts)
        output_mp4_path = None
        video_fname = None
        output_npy_path = None
        joints_fname = None
        out_feats = None
        out_joints = None
        out_lengths = None
        final_hidden = outputs["semantic_embeddings_text"]   # or motion

        save_dict = {}

        for layer_idx, h in enumerate(final_hidden):
            save_dict[f"layer_{layer_idx}_hidden"] = h.detach().cpu().numpy()

        #final_hidden = outputs["semantic_embeddings_text"]
        #np.save(os.path.join(output_dir, "final_hidden_text.npy"), final_hidden.detach().cpu().numpy())
        # fname = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(
        # time.time())) + str(np.random.randint(10000, 99999))
        # np.savetxt(os.path.join(output_dir, fname+'.txt'), out_texts, fmt='%s')
    else:
        out_feats = outputs["feats"][0]
        out_lengths = outputs["length"][0]
        out_joints = outputs["joints"][:out_lengths].detach().cpu().numpy()
        out_texts = outputs["texts"][0].replace('<Motion_Placeholder>', "")
        if out_texts.strip() == "":
            out_texts = None
        output_mp4_path, video_fname, output_npy_path, joints_fname = render_motion(
            out_joints,
            out_feats.to('cpu').numpy(), method)
        final_hidden = outputs["semantic_embeddings_motion"]   # or motion
        final_positions = outputs["semantic_positions_text"] # or motion

        save_dict = {}

        for layer_idx, h in enumerate(final_hidden):
            save_dict[f"layer_{layer_idx}_hidden"] = h.detach().cpu().numpy()
            save_dict[f"layer_{layer_idx}_pos"] = final_positions[layer_idx].detach().cpu().numpy()
        #final_hidden = outputs["semantic_embeddings_motion"]
        #np.save(os.path.join(output_dir, "final_hidden_motion.npy"), final_hidden.detach().cpu().numpy())

    data_stored[-1]['model_output'].update({
        "feats": out_feats,
        "joints": out_joints,
        "length": out_lengths,
        "texts": out_texts,
        "motion_video": output_mp4_path,
        "motion_video_fname": video_fname,
        "motion_joints": output_npy_path,
        "motion_joints_fname": joints_fname
    })

    motion_uploaded = {
        "feats": None,
        "joints": None,
        "motion_video": None,
        "motion_lengths": 0,
        "motion_token": None,
        "motion_token_string": '',
        "motion_token_length": 0,
        "motion_tokens_input": None,
        "file": None,
    }
    return data_stored, save_dict
    """
    if '<Motion_Placeholder>' == out_texts.strip():
        response = 
            Video_Components.format(video_path=output_mp4_path,
                                    video_fname=video_fname,
                                    motion_path=output_npy_path,
                                    motion_fname=joints_fname)
        ]
    elif '<Motion_Placeholder>' in out_texts.strip():
        response = [
            Text_Components.format(
                msg=out_texts.split("<Motion_Placeholder>")[0]),
            Video_Components.format(video_path=output_mp4_path,
                                    video_fname=video_fname,
                                    motion_path=output_npy_path,
                                    motion_fname=joints_fname),
            Text_Components.format(
                msg=out_texts.split("<Motion_Placeholder>")[1]),
        ]
    else:
        response = f"<h3>{out_texts}</h3>"
    """
    #response = f"""<h3>{out_texts}</h3>""" # For simplicity, just return text. You can modify this to include video if needed.
    """
    history[-1][1] = "" 
    for character in response:
        history[-1][1] += character
        time.sleep(0.02)
        yield history, motion_uploaded, data_stored
    print('bot')
    """
    """
    txt_msg = txt.submit(
            add_text, [chatbot, txt, motion_uploaded, data_stored, method],
            [chatbot, txt, motion_uploaded, data_stored],
            queue=False).then(bot, [chatbot, motion_uploaded, data_stored, method, task],
                            [chatbot, motion_uploaded, data_stored])

    file_msg = btn.upload(add_file, [chatbot, btn, txt, motion_uploaded],
                            [chatbot, txt, motion_uploaded],
                            queue=False)
    """
print("MotionGPT3 is running...")
"""
while(True):
    inputy = input("User: ")
    # Test if the input is a .npy file path
    if '.npy' in inputy:
        file = inputy
    elif inputy == "submit":
        if file is not None and txt != "":
            inputy = input("t2m or m2t?: ")
            task = inputy
        elif file is not None and txt == "":
            task = "m2t"
        elif file is None and txt != "":
            task = "t2m"
        data_stored = submit(motion_uploaded, data_stored, file, txt, task)
        print(data_stored[-1])
        txt = ""
        task = ""
        file = None
    else:
        txt = inputy
"""
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    # First message is always userid
    if userid is None:
        if line.startswith("USERID:"):
            userid = int(line[7:])
            print("TEXT:ready", flush=True)  # signal frontend
        continue

    # Parse file and/or text
    file = None
    txt = ""
    inputy = None
    dataset_id = None
    skip_inference = False
    for part in line.split("|"):
        #if part.startswith("MODE:"):
        if part.startswith("FILE:"):
            file = part[5:]
        elif part.startswith("TEXT:"):
            txt = part[5:]
            try:
                score_value = int(txt)
                db.score(score_value, 2)  # setnum is the chatbox number
                skip_inference = True  # skip normal inference for this stdin line
                break
            except ValueError:
                pass
        elif part.startswith("TASK:"):
            inputy = part[5:]  # "t2m" or "m2t"
        elif part.startswith("DATASET:"):
            dataset_id = int(part[8:])

    if skip_inference:
        print("TEXT:Score recorded", flush=True)
        continue

    # Your inference logic here
    #print("TEXT:Processing...", flush=True)
    #if file and not os.path.exists("./cache/"+file):
    #    db.change_tonpy("./cache/"+file)
    
    if file is not None and txt != "":
        if txt.startswith("PRED:"):
            txt = ""
            task = "pred"
        else:
            task = inputy
    elif file is not None and txt == "":
        task = "m2t"
    elif file is None and txt != "":
        task = "t2m"
    data_stored, save_dict = submit(motion_uploaded, data_stored, file, txt, task, dataset_id)
    #print(data_stored[-1])
    if data_stored[-1]['model_output']['motion_video_fname']:
        timestamp = int(time.time())
        print(f"FILE:mp4:{os.path.abspath('./cache/'+data_stored[-1]['model_output']['motion_video_fname'])}?t={timestamp}", flush=True)
    if data_stored[-1]['model_output']['texts']:
        print(f"TEXT:{data_stored[-1]['model_output']['texts']}", flush=True)
    motion_joints_path = (
        os.path.abspath('./cache/' + data_stored[-1]['model_output']['motion_joints_fname'])
        if data_stored[-1]['model_output']['motion_joints_fname'] else None
    )
    db.add_history(
        data_stored[-1]['user_input'],
        data_stored[-1]['model_output']['texts'],
        data_stored[-1]['motion_uploaded']['file'],
        motion_joints_path,
        2, task, userid, save_dict
    )
    """ for steering
    history = pickle.dumps(save_dict)
    np.savez("./cache/output.npz", **np.load(io.BytesIO(history), allow_pickle=True))
    cache_dir = "./cache" #clearing cache except for needed file
    """
    for fname in os.listdir(cache_dir):
        fpath = os.path.abspath(os.path.join(cache_dir, fname))
        if os.path.isfile(fpath) and fname != data_stored[-1]['model_output']['motion_video_fname']:
            os.remove(fpath) #cache delete end line