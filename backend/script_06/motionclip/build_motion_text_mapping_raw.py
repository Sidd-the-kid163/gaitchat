#!/usr/bin/env python3
import argparse
import os
import joblib
import numpy as np
from tqdm import tqdm

SEQ_LEN = 100
CLIP_LABEL_TEXT = "text_raw_labels"


def chunk_to_clip_text(txt_arr):
    """
    Replicates exactly what dataset.py does:
        text_labels = " and ".join(list(np.unique(text_labels)))
    np.unique sorts alphabetically, so order doesn't matter — it's deterministic.
    """
    # Strip each element the same way dataset.py would receive them
    cleaned = []
    for elem in txt_arr:
        if isinstance(elem, bytes):
            s = elem.decode("utf-8", errors="ignore").strip()
        else:
            s = str(elem).strip()
        cleaned.append(s)
    cleaned = np.array(cleaned)
    return " and ".join(list(np.unique(cleaned)))


def build_mapping_from_db_file(db_path, split_name, seq_len=SEQ_LEN, clip_label_text=CLIP_LABEL_TEXT):
    print(f"Loading {db_path} ...")
    db = joblib.load(db_path, mmap_mode="r")
    if clip_label_text not in db:
        raise KeyError(f"{clip_label_text} not found in {db_path}. Available keys: {list(db.keys())}")

    motion_collection = {}
    n_sequences = len(db["thetas"])

    for seq_idx in tqdm(range(n_sequences), desc=f"Scanning {split_name}"):
        n_frames = db["thetas"][seq_idx].shape[0]
        n_sub_seq = n_frames // seq_len
        if n_sub_seq == 0:
            continue

        n_frames_in_use = n_sub_seq * seq_len
        text_chunks = np.split(db[clip_label_text][seq_idx][:n_frames_in_use], n_sub_seq)

        for sub_idx, txt_arr in enumerate(text_chunks):
            # Use same logic as dataset.py __getitem__
            txt = chunk_to_clip_text(txt_arr)
            motion_collection.setdefault(txt, []).append((split_name, seq_idx, sub_idx))

    return motion_collection


def merge_mappings(*maps):
    merged = {}
    for mp in maps:
        for k, v in mp.items():
            merged.setdefault(k, []).extend(v)
    return merged


def save_outputs(mapping, txt_path):
    npy_path = txt_path.replace(".txt", ".npy")
    with open(txt_path, "w", encoding="utf-8") as fw:
        labels = sorted(mapping.keys())
        fw.write("\n".join(labels) + "\n")
    np.save(npy_path, mapping, allow_pickle=True)
    print(f"Saved text labels  : {txt_path}")
    print(f"Saved mapping cache: {npy_path}")
    print(f"Unique labels      : {len(mapping)}")


def debug_key(mapping, key):
    if key in mapping:
        print(f"\n✓ Found exact match: '{key}' -> {len(mapping[key])} entries")
        return
    print(f"\n✗ Key not found: '{key}'")
    close = [k for k in mapping if any(w in k for w in key.split())]
    if close:
        print(f"  Possible partial matches ({len(close)} found, showing first 10):")
        for k in close[:10]:
            print(f"    '{k}'")


def main():
    parser = argparse.ArgumentParser(description="Build MotionCLIP text-to-motion mapping cache")
    parser.add_argument("--split", choices=["train", "vald", "all"], default="all")
    parser.add_argument("--base", default="data/amass_db/amass_30fps_db.pt")
    parser.add_argument("--debug_key", default=None,
                        help="After building, check if this key exists (quote multi-word keys)")
    args = parser.parse_args()

    if args.split == "all":
        train_path = args.base.replace("_db.pt", "_train.pt")
        vald_path  = args.base.replace("_db.pt", "_vald.pt")
        train_map  = build_mapping_from_db_file(train_path, "train")
        vald_map   = build_mapping_from_db_file(vald_path,  "vald")
        mapping    = merge_mappings(train_map, vald_map)
        txt_path   = args.base.replace("_db.pt", "_all_text_labels.txt")
    else:
        db_path  = args.base.replace("_db.pt", f"_{args.split}.pt")
        mapping  = build_mapping_from_db_file(db_path, args.split)
        txt_path = args.base.replace("_db.pt", f"_{args.split}_text_labels.txt")

    save_outputs(mapping, txt_path)

    if args.debug_key:
        debug_key(mapping, args.debug_key)


if __name__ == "__main__":
    main()