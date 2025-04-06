import math
import numpy as np
import torch
torch.set_printoptions(sci_mode=False)
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from typing import List
import lightning as L
from lightning.pytorch import seed_everything
from tqdm import tqdm 

from mobileposer.config import *
from mobileposer.constants import MODULES
from mobileposer.data import PoseDataModule


# faster training
torch.set_float32_matmul_precision('medium')

def print_model_size(model):
    num_params = sum(p.numel() for p in model.parameters())
    print(f"# Parameters: {num_params:,}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--module", default="poser")
    args = parser.parse_args()

    # Set seed for reproducible results
    seed_everything(42, workers=True)

    # Initialize DataModule
    datamodule = PoseDataModule()

    # Initialize model from scratch
    model = MODULES[args.module]()

    # Print number of parameters
    print_model_size(model)

    # Initialize a PyTorch Lightning Trainer
    trainer = L.Trainer(
        overfit_batches=1,
        gradient_clip_val=1,
        max_epochs=100,
        devices=[0],
        accelerator=train_hypers.accelerator,
        enable_checkpointing=False,
        logger=False,
        deterministic=True
    )

	# Train model
    trainer.fit(model, datamodule=datamodule)
