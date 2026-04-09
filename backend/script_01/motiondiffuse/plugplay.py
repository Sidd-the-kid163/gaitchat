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
    for part in line.split("|"):
        #if part.startswith("MODE:"):
        if part.startswith("TEXT:"):
            txt = part[5:]
            try:
                score_value = int(txt)
                db.score(score_value, 1)  # setnum is the chatbox number
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
    subprocess.run([
    "python", "-u", "./tools/visualization.py",
    "--opt_path", "./checkpoints/t2m/t2m_motiondiffuse/opt.txt",
    "--text", f"{txt}",
    "--motion_length", "60",
    "--result_path", "./cache/test_sample.mp4",
    "--npy_path", "./cache/test_sample.npy"
    ], check=True)
    #print(data_stored[-1])
    if os.path.exists("./cache/test_sample.mp4"):
        timestamp = int(time.time())
        print(f"FILE:mp4:{os.path.abspath('./cache/test_sample.mp4')}?t={timestamp}", flush=True)
        db.add_history(
            txt,
            None, # model_input is not used in current implementation
            None, # motion_uploaded is not used in current implementation
            "./cache/test_sample.npy",
            1, 't2m', userid
        )