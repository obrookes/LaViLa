import os
import ast
import copy
import math
import torch
import pickle
import einops
import itertools
import argparse
import torchmetrics
import pandas as pd
import numpy as np
from typing import Any, Callable
from einops import rearrange
from torch import nn, Tensor
from torch.nn import Transformer
from typing import Optional
import torch.nn.functional as F
import pytorch_lightning as pl
from flash.video import VideoClassificationData
from pytorch_lightning.loggers import WandbLogger
from flash.core.utilities.imports import requires
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DistributedSampler, RandomSampler
from torchvision.transforms import Compose, Resize
from torchvision.transforms.v2 import (
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomVerticalFlip,
    TrivialAugmentWide,
    RandAugment,
)
from flash.core.data.io.input import DataKeys
from flash.core.data.transforms import ApplyToKeys
from flash.core.optimizers import LinearWarmupCosineAnnealingLR
from train_lavila import BaselineVLMClassifier
from pytorchvideo.data import LabeledVideoDataset, UniformClipSampler
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.utilities.classification import TargetFormatter
from pytorchvideo.transforms import create_video_transform
from dataclasses import dataclass

torch.set_float32_matmul_precision("high")


@dataclass
class CustomTargetFormatter(TargetFormatter):
    def format(self, target: Any) -> Any:
        return target


@requires("video")
@dataclass
class CustomTransform(InputTransform):
    def train_per_batch_transform(self) -> Callable:
        train_transform = Compose(
            [
                RandomHorizontalFlip(p=0.2),
                RandomVerticalFlip(p=0.2),
                RandAugment(num_ops=4, magnitude=7),
                Normalize(mean=[0.5017, 0.5159, 0.5168], std=[0.2814, 0.2778, 0.2635]),
            ]
        )
        return ApplyToKeys(
            DataKeys.INPUT,
            train_transform,
        )

    def val_per_batch_transform(self) -> Callable:
        val_transform = Compose(
            [
                Normalize(mean=[0.5017, 0.5159, 0.5168], std=[0.2814, 0.2778, 0.2635]),
            ]
        )
        return ApplyToKeys(
            DataKeys.INPUT,
            val_transform,
        )


class FinetuneLavila(pl.LightningModule):
    def __init__(
        self,
        lr,
        num_classes,
        num_frames,
        warmup_epochs,
        max_epochs,
        original_weights,
        ckpt,
        finetune_method,
    ):
        super().__init__()
        self.lr = lr
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.finetune_method = finetune_method

        # Initialize model
        self.model = BaselineVLMClassifier.load_from_checkpoint(
            ckpt,
            lr=lr,
            num_classes=num_classes,
            num_frames=num_frames,
            path_to_ckpt=original_weights,  # vclm_openai_timesformer_base_gpt2_base.pt_ego4d.jobid_319630.ep_0002.md5sum_68a71f.pth
            freeze_lm=True,
            freeze_visual_spatial=False,
            freeze_visual_temporal=False,
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
        )
        self.model = self.model.model.visual
        self.fc = nn.Linear(768, num_classes)

        # freeze all layers except last if linear
        if self.finetune_method == "linear":
            for param in self.model.parameters():
                param.requires_grad = False

        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.lr = lr

        self.train_map = torchmetrics.AveragePrecision(
            task="multilabel", average="macro", num_labels=num_classes
        )

        self.train_none_map = torchmetrics.AveragePrecision(
            task="multilabel", average=None, num_labels=num_classes
        )

        self.val_map = torchmetrics.AveragePrecision(
            task="multilabel", average="macro", num_labels=num_classes
        )

        self.val_none_map = torchmetrics.AveragePrecision(
            task="multilabel", average=None, num_labels=num_classes
        )

        self.test_map = torchmetrics.AveragePrecision(
            task="multilabel", average="macro", num_labels=num_classes
        )

        self.test_none_map = torchmetrics.AveragePrecision(
            task="multilabel", average=None, num_labels=num_classes
        )

        self.behaviours = [
            "camera_reaction",
            "tool_use",
            "object_carrying",
            "bipedal",
            "feeding",
            "chimp_carrying",
            "vocalisation",
            "climbing",
            "aggression",
            "travel",
            "sex",
            "piloerection",
            "social_interaction",
            "grooming",
            "display",
            "cross_species_interaction",
            "resting",
            "playing",
        ]

    def decode(self, encoded_label, labels):
        indices = np.where(np.array(encoded_label) == 1)
        return [labels[i] for i in indices[0]]

    def forward(self, x):
        x = rearrange(x, "b f c h w -> b c f h w")
        x = self.model(x)
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["input"], batch["target"]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.train_map(logits, y)
        self.train_none_map(logits, y)
        return loss

    def training_epoch_end(self, outputs):
        self.log(
            "train_map",
            self.train_map,
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )

        loss = torch.stack([x["loss"] for x in outputs]).mean()

        self.log(
            "loss",
            loss,
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=False,
            sync_dist=True,
        )

        ap_dict = {}
        average_precision = self.train_none_map.compute()
        for i, label in enumerate(self.behaviours):
            ap_dict[f"train_{label}"] = average_precision[i]
        self.log_dict(ap_dict, sync_dist=True)

    # Validation Loop #

    def validation_step(self, batch, batch_idx):
        x, y = batch["input"], batch["target"]
        logits = self(x)
        self.val_map(logits, y)
        self.val_none_map(logits, y)

    def validation_epoch_end(self, outputs):
        # Log val map acc per epoch
        self.log(
            "val_map",
            self.val_map,
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )

        ap_dict = {}
        average_precision = self.val_none_map.compute()
        for i, label in enumerate(self.behaviours):
            ap_dict[f"val_{label}"] = average_precision[i]
        self.log_dict(ap_dict, sync_dist=True)

    # Test loop
    def test_step(self, batch, batch_idx):
        x, y = batch["input"], batch["target"]
        logits = self(x)
        self.test_map(logits, y)
        self.test_none_map(logits, y)

    def test_epoch_end(self, outputs):
        self.log(
            "test_map",
            self.test_map,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        ap_dict = {}
        average_precision = self.test_none_map.compute()
        for i, label in enumerate(self.behaviours):
            ap_dict[f"test_{label}"] = average_precision[i]
        self.log_dict(ap_dict, sync_dist=True)

    def configure_optimizers(self):
        # Optimiser
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=self.warmup_epochs, max_epochs=self.max_epochs
        )
        return [optimizer], [scheduler]


def main():
    parser = argparse.ArgumentParser()

    # Data args
    parser.add_argument("--path_to_dataset", type=str, required=True)

    # Model args
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--original_weights", type=str, required=True)
    parser.add_argument("--finetune_method", type=str, required=True)
    parser.add_argument("--sequence_length", type=int, required=True)
    parser.add_argument("--num_classes", type=int, required=False, default=18)
    parser.add_argument("--lr", type=float, required=False, default=1e-5)
    parser.add_argument("--warmup_epochs", type=int, required=False, default=5)

    # Trainer args
    parser.add_argument("--num_nodes", type=int, required=False, default=1)
    parser.add_argument("--accelerator", type=str, required=False, default="gpu")
    parser.add_argument("--strategy", type=str, required=False, default="ddp")
    parser.add_argument("--devices", type=int, required=True)

    parser.add_argument("--max_epochs", type=int, required=False, default=48)
    parser.add_argument(
        "--accumulate_grad_batches", type=int, required=False, default=1
    )

    parser.add_argument("--fast_dev_run", type=int, required=False, default=False)
    parser.add_argument(
        "--limit_train_batches", type=float, required=False, default=1.0
    )
    parser.add_argument("--limit_val_batches", type=float, required=False, default=1.0)

    # Data module args
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_workers", type=int, required=True)

    parser.add_argument("--ckpt_name", type=str, default=None)

    args = parser.parse_args()

    ckpt_name = f"_{args.ckpt_name}" if args.ckpt_name else ""

    pl.seed_everything(42, workers=True)
    wand_logger = WandbLogger(
        offline=True,
        name=f"lavila_finetune_{args.sequence_length}_{args.lr}{ckpt_name}",
        id=f"lavila_finetune_{args.sequence_length}_{args.lr}{ckpt_name}",
    )

    torch.autograd.set_detect_anomaly(True)

    # Initialize the trainer
    trainer = pl.Trainer(
        num_nodes=args.num_nodes,
        accelerator=args.accelerator,
        strategy=args.strategy,
        devices=args.devices,
        accumulate_grad_batches=args.accumulate_grad_batches,
        logger=wand_logger,
        fast_dev_run=args.fast_dev_run,
        max_epochs=args.max_epochs,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        callbacks=[
            ModelCheckpoint(
                dirpath=f"./checkpoints/lavila_finetune_{args.sequence_length}_{args.lr}",
                monitor="val_map",
                mode="max",
            )
        ],
    )

    model = FinetuneLavila(
        lr=args.lr,
        num_classes=args.num_classes,
        num_frames=args.sequence_length,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        original_weights=args.original_weights,
        ckpt=args.ckpt,
        finetune_method=args.finetune_method,
    )

    # Load tensor dataset
    print("Loading dataset...")
    with open(args.path_to_dataset, "rb") as f:
        dataset = pickle.load(f)
    print("Loaded!")

    train_dataset = dataset["train"]
    val_dataset = dataset["val"]
    test_dataset = dataset["test"]

    transform = CustomTransform()
    formatter = CustomTargetFormatter()

    datamodule = VideoClassificationData.from_tensors(
        train_data=train_dataset["tensor"],
        train_targets=train_dataset["label"],
        val_data=val_dataset["tensor"],
        val_targets=val_dataset["label"],
        test_data=test_dataset["tensor"],
        test_targets=test_dataset["label"],
        transform=transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_formatter=formatter,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
