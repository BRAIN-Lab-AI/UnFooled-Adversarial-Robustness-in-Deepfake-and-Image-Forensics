# flake8: noqa
from __future__ import annotations
import math, os, sys, re, json, time, random, pathlib, logging
import numpy as np
try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except Exception:
    # allow import without torch on docs/build
    torch = None
    nn = object
    F = None
    Dataset = object
    DataLoader = object
##############################################################################
# DATASETS / LOADERS
##############################################################################


class DiskDataset(Dataset):
    def __init__(self, fake_dir, real_dir, split="train", seed=2025, val_frac=0.15, test_frac=0.15):
        paths = [(os.path.join(fake_dir, p), 1) for p in os.listdir(fake_dir)]
        paths += [(os.path.join(real_dir, p), 0) for p in os.listdir(real_dir)]
        paths = [(p,l) for p,l in paths if p.lower().endswith((".jpg",".jpeg",".png"))]
        rng = random.Random(seed); rng.shuffle(paths)
        n = len(paths); val_n = int(n*val_frac); test_n = int(n*test_frac)
        if split == "train": self.samples = paths[:n-val_n-test_n]
        elif split == "val": self.samples = paths[n-val_n-test_n:n-test_n]
        else: self.samples = paths[n-test_n:]
        self.split = split

NUM_WORKERS = 0; PIN_MEMORY=True; PERSISTENT=False
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True,
                          persistent_workers=PERSISTENT)
val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                          persistent_workers=PERSISTENT)
test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                          persistent_workers=PERSISTENT)
