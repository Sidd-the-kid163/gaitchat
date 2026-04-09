##import gradio as gr
# import random
import sys
import time
import csv
import re
import spacy
import numpy as np
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
nlp = spacy.load("en_core_web_sm")

def pick_word(text):
    doc = nlp(text)
    
    # Lists to store words by priority
    ing_verbs = []
    nouns = []
    verbs = []
    adj = []

    # Iterate through tokens and store matching ones
    for token in doc:
        if token.tag_ == "VBG":
            ing_verbs.append(token.text)
        elif token.pos_ == "NOUN":
            nouns.append(token.text)
        elif token.pos_ == "VERB":
            verbs.append(token.text)
        elif token.pos_ == "ADJ":
            adj.append(token.text)
    
    # Pick from the end
    if ing_verbs:
        return ing_verbs[-1]
    elif nouns and len(nouns) > 1:
        if adj and len(nouns) > 2:
            return adj[-1] + " " + nouns[(len(nouns)//2)-1]
        elif len(nouns) > 2:
            return nouns[(len(nouns)//2)-1]
        if adj:
            return adj[-1] + " " + nouns[-1]
        return nouns[-1]
    elif adj:
        return adj[-1]
    elif verbs:
        return verbs[-1]
    else:
        return None  # no match

def get_last_two_segments(text):
    # Split by standalone word 'and' (case-insensitive) and trim
    segments = [seg.strip() for seg in re.split(r"\band\b", text, flags=re.IGNORECASE) if seg.strip()]
    
    # Pick first two segments in original order
    if len(segments) >= 2:
        return segments[:2]
    else:
        return [None, None]  # no valid segments

def load_latent_dict(npz_path):
    with np.load(npz_path, allow_pickle=True) as latent_npz:
        return {key: latent_npz[key] for key in latent_npz.files}

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
    interp = False
    skip_inference = False
    for part in line.split("|"):
        if part.startswith("TEXT:"):
            txt = part[5:]
            try:
                score_value = int(txt)
                db.score(score_value, 6)  # setnum is the chatbox number
                skip_inference = True  # skip normal inference for this stdin line
                break
            except ValueError:
                pass
    if skip_inference:
        print("TEXT:Score recorded", flush=True)
        continue
    temp = get_last_two_segments(txt)
    if temp == [None, None]:
        txt = pick_word(txt)
    else:
        interp = True
    # Your inference logic here
    #print("TEXT:Processing...", flush=True)
    #if file and not os.path.exists("./cache/"+file):
    #    db.change_tonpy("./cache/"+file)
    save_dict = {}
    if not interp:
        new_content = f"""{txt}"""
        with open("./assets/paper_texts.txt", "w") as f:
            f.write(new_content)
        subprocess.run([
        "python", "-m", "src.visualize.text2motion", "./exps/paper-model/checkpoint_0100.pth.tar", "--input_file", "./assets/paper_texts.txt"
        ], check=True)
        save_dict = load_latent_dict('./exps/paper-model/output_latents_text2motion.npz')
        print(save_dict.keys())
    else:
        with open("./assets/paper_interps.csv", "r", newline="") as f:
            reader = list(csv.reader(f))
        if pick_word(temp[0]) and pick_word(temp[1]):
            reader[1] = [pick_word(temp[0]), pick_word(temp[1])]   
            with open("./assets/paper_interps.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(reader)
            subprocess.run([
            "python", "-m", "src.visualize.motion_interpolation", "./exps/paper-model/checkpoint_0100.pth.tar", "--input_file", "./assets/paper_interps.csv"
            ], check=True)
            save_dict = load_latent_dict('./exps/paper-model/output_latents_interpolation.npz')
            print(save_dict.keys())
        elif pick_word(temp[0]):
            new_content = f"""{pick_word(temp[0])}"""
            with open("./assets/paper_texts.txt", "w") as f:
                f.write(new_content)
            subprocess.run([
            "python", "-m", "src.visualize.text2motion", "./exps/paper-model/checkpoint_0100.pth.tar", "--input_file", "./assets/paper_texts.txt"
            ], check=True)
            save_dict = load_latent_dict('./exps/paper-model/output_latents_text2motion.npz')
            print(save_dict.keys())
        elif pick_word(temp[1]):
            new_content = f"""{pick_word(temp[1])}"""
            with open("./assets/paper_texts.txt", "w") as f:
                f.write(new_content)
            subprocess.run([
            "python", "-m", "src.visualize.text2motion", "./exps/paper-model/checkpoint_0100.pth.tar", "--input_file", "./assets/paper_texts.txt"
            ], check=True)
            save_dict = load_latent_dict('./exps/paper-model/output_latents_text2motion.npz')
            print(save_dict.keys())
    #print(data_stored[-1])
    if os.path.exists("./exps/paper-model/output.gif"):
        timestamp = int(time.time())
        print(f"FILE:gif:{os.path.abspath('./exps/paper-model/output.gif')}?t={timestamp}", flush=True)
        db.add_history(
            txt,
            None, # model_input is not used in current implementation
            None, # motion_uploaded is not used in current implementation
            "./exps/paper-model/output.npy",
            6, 't2m', userid, save_dict
        )