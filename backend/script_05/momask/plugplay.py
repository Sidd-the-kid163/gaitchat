##import gradio as gr
# import random
import sys
import time
# import cv2
import os
import subprocess
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")) #for databaseaccess
import databaseaccess as db

os.environ["PYTHONPATH"] = f".:{os.environ.get('PYTHONPATH', '')}"

conn, cursor = db.connect()
db.init(conn, cursor)
userid = None
task = ""
txt = ""
file = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXEC = sys.executable
GENERATION_DIR = os.path.join(BASE_DIR, "generation", "exp1")
EDITING_DIR = os.path.join(BASE_DIR, "editing", "exp1")
TRIAL_NPY_PATH = os.path.join(GENERATION_DIR, "trial.npy")

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
    txt = ""
    file = None
    dataset_id = None
    skip_inference = False
    mode = None
    for part in line.split("|"):
        if part.startswith("MODE:"):
            mode = part[5:]  # "direct", "broadcast", or "inbetween"
        elif part.startswith("TEXT:"):
            txt = part[5:]
            try:
                score_value = int(txt)
                db.score(score_value, 5)  # setnum is the chatbox number
                skip_inference = True  # skip normal inference for this stdin line
                break
            except ValueError:
                pass
        elif part.startswith("FILE:"):
            file = part[5:]
        elif part.startswith("DATASET:"):
            dataset_id = int(part[8:])

    if skip_inference:
        print("TEXT:Score recorded", flush=True)
        continue
    # Your inference logic here
    #print("TEXT:Processing...", flush=True)
    #if file and not os.path.exists("./cache/"+file):
    #    db.change_tonpy("./cache/"+file)
    if mode == "inbetween":
        if file is not None:
            datanpy = db.change_tonpy(file, dataset_id)
            os.makedirs(os.path.dirname(TRIAL_NPY_PATH), exist_ok=True)
            np.save(TRIAL_NPY_PATH, datanpy)
            subprocess.run([
                PYTHON_EXEC, "edit_t2m.py", "--gpu_id", "0", "--ext", "exp1", "-msec", "0.25,0.75",
                "--source_motion", TRIAL_NPY_PATH, "--text_prompt", f"{txt}"
            ], check=True, cwd=BASE_DIR)
        else:
            subprocess.run([
                PYTHON_EXEC, "edit_t2m.py", "--gpu_id", "0", "--ext", "exp1", "-msec", "0.25,0.75",
                "--text_prompt", f"{txt}"
            ], check=True, cwd=BASE_DIR)
        output_root = EDITING_DIR
        history_task = 'inbetween'
    else:
        subprocess.run([
            PYTHON_EXEC, "gen_t2m.py", "--gpu_id", "0", "--ext", "exp1", "--text_prompt", f"{txt}"
        ], check=True, cwd=BASE_DIR)
        output_root = GENERATION_DIR
        history_task = 't2m'

    output_mp4 = os.path.join(output_root, "animations", "0", "output.mp4")
    output_npy = os.path.join(output_root, "joints", "0", "output.npy")

    if os.path.exists(output_mp4):
        timestamp = int(time.time())
        print(f"FILE:mp4:{os.path.abspath(output_mp4)}?t={timestamp}", flush=True)

        if mode == "inbetween":
            db.add_history(
                txt,
                None,  # model_input is not used in current implementation
                file if file is not None else None,
                output_npy,
                5,
                history_task,
                userid
            )
        else:
            db.add_history(
                txt,
                None,  # model_input is not used in current implementation
                None,
                output_npy,
                5,
                history_task,
                userid
            )