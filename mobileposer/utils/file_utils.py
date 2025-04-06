import os
import re
import glob
import datetime
from typing import Any, Optional, Iterable

from mobileposer.config import paths, amass, datasets


def make_dir(path: str):
    if not os.path.exists(path):
        os.mkdir(path)

def get_datestring():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

def get_dir_number(path: str):
    return max([int(d) for d in os.listdir(path) if d.isdigit() and os.path.isdir(os.path.join(path, d))] + [0]) + 1

def get_file_number(path: str):
    return len(glob.glob(f"{path}/*"))

def get_best_checkpoint(path: str):
    pattern = re.compile(r"epoch=\d+-validation_step_loss=([0-9.]+).ckpt")
    files = [f for f in os.listdir(path) if pattern.search(f)]
    best_ckpt = min(files, key=lambda x: float(pattern.search(x).group(1))) if files else None
    return best_ckpt
