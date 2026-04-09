#!/usr/bin/env python3
import joblib
import numpy as np

DB_PATH = "data/amass_db/amass_30fps_train.pt"  # adjust if needed
TARGET = "drink from mug and transition"

db = joblib.load(DB_PATH, mmap_mode="r")
labels = db["text_raw_labels"]

print(f"Total sequences: {len(labels)}")
print(f"\n--- First 3 sequences, first element ---")
for i in range(min(3, len(labels))):
    raw = labels[i]
    print(f"  seq[{i}] type={type(raw)}, shape={getattr(raw, 'shape', 'N/A')}")
    elem = raw.flat[0] if isinstance(raw, np.ndarray) else raw
    print(f"  seq[{i}] first elem type={type(elem)}, repr={repr(elem)}")

print(f"\n--- Searching for '{TARGET}' ---")
found = False
for i in range(len(labels)):
    raw = labels[i]
    if isinstance(raw, np.ndarray) and raw.size > 0:
        elem = raw.flat[0]
        if isinstance(elem, bytes):
            s = elem.decode("utf-8", errors="ignore").strip()
        else:
            s = str(elem).strip()
    elif isinstance(raw, bytes):
        s = raw.decode("utf-8", errors="ignore").strip()
    else:
        s = str(raw).strip()

    if TARGET.lower() in s.lower() or s.lower() in TARGET.lower():
        print(f"  CLOSE MATCH at seq[{i}]: repr={repr(s)}")
        found = True

if not found:
    print(f"  Not found at all. Showing 20 sample labels:")
    seen = set()
    count = 0
    for i in range(len(labels)):
        raw = labels[i]
        elem = raw.flat[0] if isinstance(raw, np.ndarray) and raw.size > 0 else raw
        if isinstance(elem, bytes):
            s = elem.decode("utf-8", errors="ignore").strip()
        else:
            s = str(elem).strip()
        if s not in seen:
            seen.add(s)
            print(f"  [{count}] {repr(s)}")
            count += 1
            if count >= 20:
                break