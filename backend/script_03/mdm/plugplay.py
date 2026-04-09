##import gradio as gr
# import random
import sys
import time
# import cv2
import os
import subprocess
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")) #for databaseaccess
import databaseaccess as db

os.environ["PYTHONPATH"] = f".:{os.environ.get('PYTHONPATH', '')}"
mdm_env = os.environ.copy() #debug
#mdm_env["CUDA_VISIBLE_DEVICES"] = ""

conn, cursor = db.connect()
db.init(conn, cursor)
userid = None
task = ""
txt = ""
file = None

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
    skip_inference = False
    mode = None
    for part in line.split("|"):
        if part.startswith("MODE:"):
            mode = part[5:]  # "direct", "broadcast", or "inbetween"
        if part.startswith("TEXT:"):
            txt = part[5:]
            try:
                score_value = int(txt)
                db.score(score_value, 3)  # setnum is the chatbox number
                skip_inference = True  # skip normal inference for this stdin line
                break
            except ValueError:
                pass
    
    if skip_inference:
        print("TEXT:Score recorded", flush=True)
        continue

    # Your inference logic here
    #print("TEXT:Processing...", flush=True)
    #if file and not os.path.exists("./cache/"+file):
    #    db.change_tonpy("./cache/"+file)
    if mode == "inbetween":
        subprocess.run([
        "python", "-m", "sample.edit", "--model_path", "./save/humanml_enc_512_50steps/model000750000.pt", "--edit_mode", "in_between", "--prefix_end", "0.25", "--suffix_start", "0.75", "--text_condition", f"{txt}", "--output_dir", "./cache", "--num_samples", "1", "--num_repetitions", "1"
        ], check=True)
    else:
        subprocess.run([
        "python", "-m", "sample.generate", "--model_path", "./save/humanml_enc_512_50steps/model000750000.pt", "--text_prompt", f"{txt}", "--output_dir", "./cache", "--num_samples", "1", "--num_repetitions", "1" #"--device", "-1"
        ], check=True) #,env=mdm_env)
    #print(data_stored[-1])
    if os.path.exists("./cache/samples_00_to_00.mp4"):
        timestamp = int(time.time())
        print(f"FILE:mp4:{os.path.abspath('./cache/samples_00_to_00.mp4')}?t={timestamp}", flush=True)
        if mode == "inbetween":
            db.add_history(
                txt,
                None, # model_input is not used in current implementation
                None, # motion_uploaded is not used in current implementation
                "./cache/results.npy",
                3, 'inbetween', userid
            )
        else:
            db.add_history(
                txt,
                None, # model_input is not used in current implementation
                None, # motion_uploaded is not used in current implementation
                "./cache/results.npy",
                3, 't2m', userid
            )