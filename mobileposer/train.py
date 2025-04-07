import os
import math
import numpy as np
import torch
torch.set_printoptions(sci_mode=False)
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch import seed_everything
from argparse import ArgumentParser
from pathlib import Path
from typing import List
from tqdm import tqdm 
import wandb

from mobileposer.constants import MODULES
from mobileposer.data import PoseDataModule
from mobileposer.utils.file_utils import (
    get_datestring, 
    make_dir, 
    get_dir_number, 
    get_best_checkpoint
)
from mobileposer.config import paths, train_hypers, finetune_hypers


# set precision for Tensor cores
torch.set_float32_matmul_precision('medium')


class TrainingManager:
    """Manage training of MobilePoser modules."""
    def __init__(self, finetune: str=None, fast_dev_run: bool=False):
        self.finetune = finetune
        self.fast_dev_run = fast_dev_run
        self.hypers = finetune_hypers if finetune else train_hypers

    def _setup_wandb_logger(self, save_path: Path):
        wandb_logger = WandbLogger(
            project=save_path.name, 
            name=get_datestring(),
            save_dir=save_path
        ) 
        return wandb_logger

    def _setup_callbacks(self, save_path):
        checkpoint_callback = ModelCheckpoint(
                monitor="validation_step_loss",
                save_top_k=3,
                mode="min",
                verbose=False,
                dirpath=save_path, 
                save_weights_only=True,
                filename="{epoch}-{validation_step_loss:.4f}"
                )
        return checkpoint_callback

    def _setup_trainer(self, module_path: Path):
        print("Module Path: ", module_path.name, module_path)
        logger = self._setup_wandb_logger(module_path) 
        checkpoint_callback = self._setup_callbacks(module_path)
        trainer = L.Trainer(
                fast_dev_run=self.fast_dev_run,
                min_epochs=self.hypers.num_epochs,
                max_epochs=self.hypers.num_epochs,
                devices=[self.hypers.device], 
                accelerator=self.hypers.accelerator,
                logger=logger,
                callbacks=[checkpoint_callback],
                deterministic=True
                )
        return trainer

    def train_module(self, model: L.LightningModule, module_name: str, checkpoint_path: Path):
        # set the appropriate hyperparameters
        model.hypers = self.hypers 

        # create directory for module
        module_path = checkpoint_path / module_name
        make_dir(module_path)
        datamodule = PoseDataModule(finetune=self.finetune)
        trainer = self._setup_trainer(module_path)

        print()
        print("-" * 50)
        print(f"Training Module: {module_name}")
        print("-" * 50)
        print()

        try:
            trainer.fit(model, datamodule=datamodule)
        finally:
            wandb.finish()
            del model
            torch.cuda.empty_cache()


def get_checkpoint_path(finetune: str, init_from: str):
    if finetune:
        # finetune from a checkpoint
        parts = init_from.split(os.path.sep)
        checkpoint_path = Path(os.path.join(parts[0], parts[1]))
        finetune_dir = f"finetuned_{finetune}"
        checkpoint_path = checkpoint_path / finetune_dir
    else:
        # make directory for trained models
        dir_name = get_dir_number(paths.checkpoint) 
        checkpoint_path = paths.checkpoint / str(dir_name)
    
    make_dir(checkpoint_path)
    return Path(checkpoint_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--module", default=None)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--finetune", type=str, default=None)
    parser.add_argument("--init-from", nargs="?", default="scratch", type=str)
    args = parser.parse_args()

    # set seed for reproducible results
    seed_everything(42, workers=True)

    # create checkpoint directory, if missing
    paths.checkpoint.mkdir(exist_ok=True)

    # initialize training manager
    checkpoint_path = get_checkpoint_path(args.finetune, args.init_from)
    training_manager = TrainingManager(
        finetune=args.finetune,
        fast_dev_run=args.fast_dev_run
    )

    # train single module
    if args.module:
        if args.module not in MODULES.keys():
            raise ValueError(f"Module {args.module} not found.")

        model_dir = Path(args.init_from)
        module = MODULES[args.module]
        model = module() # init model from scratch

        if args.finetune: 
            model_path = get_best_checkpoint(model_dir)
            model = module.from_pretrained(model_path=os.path.join(model_dir, model_path)) # load pre-trained model

        training_manager.train_module(model, args.module, checkpoint_path)
    else:
        # train all modules
        for module_name, module in MODULES.items():
            training_manager.train_module(module(), module_name, checkpoint_path)
