import os
import torch
import torch.nn as nn
import lightning as L
from torch.nn import functional as F
import numpy as np
from argparse import ArgumentParser

from mobileposer.config import *
from mobileposer.utils.model_utils import *
from mobileposer.viewer import Viewer


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('--model', type=str, default=paths.weights_file)
    args.add_argument('--dataset', type=str, default='dip')
    args.add_argument('--combo', type=str, default='lw_rp')
    args.add_argument('--with-tran', action='store_true')
    args.add_argument('--seq-num', type=int, default=1)
    args = args.parse_args()

    # check for valid combo
    combos = amass.combos.keys()
    if args.combo not in combos:
        raise ValueError(f"Invalid combo: {args.combo}. Must be one of {combos}")

    # view dataset
    v = Viewer(dataset=args.dataset, seq_num=args.seq_num, combo=args.combo)
    v.view(with_tran=args.with_tran)

