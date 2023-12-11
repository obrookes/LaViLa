import ast
import torch
import pickle
import argparse
import torchmetrics
import numpy as np
from torchmetrics import Metric
from dataclasses import dataclass
from flash.core.utilities.imports import requires
from collections import OrderedDict
from typing import Any, Callable
from torch import nn, Tensor
import torch.nn.functional as F
import pytorch_lightning as pl
from flash.video import VideoClassificationData
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
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
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.utilities.classification import TargetFormatter
from lavila.models.models import VCLM_OPENAI_TIMESFORMER_BASE_GPT2
from lavila.models.tokenizer import MyGPT2Tokenizer


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


class CaptionLoss(nn.Module):
    def __init__(self, pad_id=0, tokenizer=None):
        super().__init__()
        self.pad_id = pad_id
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id

    def forward(self, outputs):
        logits = outputs["text_tokens_logits"]
        labels = outputs["labels"]
        return F.cross_entropy(
            logits, labels, ignore_index=self.pad_id, reduction="mean"
        )


class Accuracy(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.pad_id = 0

    def update(self, outputs: dict) -> None:
        logits = outputs["text_tokens_logits"]
        labels = outputs["labels"]
        for i in range(logits.size(0)):
            pred = torch.argmax(logits[i], dim=0)
            nopad = labels[i].ne(self.pad_id)
            self.correct += (pred.eq(labels[i]) & nopad).sum()
            self.total += nopad.sum()

    def compute(self) -> Tensor:
        return self.correct.float() / self.total


class BaselineVLMClassifier(pl.LightningModule):
    def __init__(
        self,
        lr,
        num_classes,
        num_frames,
        path_to_ckpt,
        freeze_lm,
        freeze_visual_spatial,
        freeze_visual_temporal,
    ):
        super().__init__()
        self.lr = lr
        self.num_classes = num_classes

        # Initialise LaVila
        self.model = VCLM_OPENAI_TIMESFORMER_BASE_GPT2(
            text_use_cls_token=False,
            project_embed_dim=256,
            gated_xattn=True,
            timesformer_gated_xattn=False,
            freeze_lm_vclm=freeze_lm,
            freeze_visual_vclm=freeze_visual_spatial,
            freeze_visual_vclm_temporal=freeze_visual_temporal,
            num_frames=num_frames,
            drop_path_rate=0.0,
        )
        self.tokenizer = MyGPT2Tokenizer("gpt2")

        ckpt = torch.load(path_to_ckpt, map_location="cpu")
        state_dict = OrderedDict()
        for k, v in ckpt["state_dict"].items():
            state_dict[k.replace("module.", "")] = v

        # Load pretrained weights + drop size mismatches
        model_state_dict = self.model.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(
                        f"Skip loading parameter: {k}, "
                        f"required shape: {model_state_dict[k].shape}, "
                        f"loaded shape: {state_dict[k].shape}"
                    )
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                print(f"Dropping parameter {k}")

        self.loss_fn = CaptionLoss(tokenizer=self.tokenizer)

        # Metrics
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def forward(self, x_v, x_t):
        from einops import rearrange

        tokens = self.tokenizer(x_t)
        outputs = self.model(
            image=rearrange(x_v, "b t c w h -> b c t w h"), text=tokens
        )
        return outputs

    def get_inputs(self, batch):
        x_v, x_t, y = batch["input"], batch["target"]["desc"], batch["target"]["label"]
        return x_v, x_t, y

    def training_step(self, batch, batch_idx):
        x_v, x_t, y = self.get_inputs(batch)
        outputs = self(x_v, x_t)
        loss = self.loss_fn(outputs)
        self.train_acc(outputs)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log(
            "loss",
            loss,
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train_acc",
            self.train_acc.compute(),
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=False,
            sync_dist=True,
        )

    # Validation Loop #
    def validation_step(self, batch, batch_idx):
        x_v, x_t, y = self.get_inputs(batch)
        outputs = self(x_v, x_t)
        loss = self.loss_fn(outputs)
        self.val_acc(outputs)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log(
            "val_loss",
            loss,
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=False,
            sync_dist=True,
        )

        self.log(
            "val_acc",
            self.val_acc.compute(),
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )

    # Test loop
    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        # Optimiser
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=10, max_epochs=50
        )
        return [optimizer], [scheduler]


def main():
    parser = argparse.ArgumentParser()

    # Data args
    parser.add_argument("--path_to_dataset", type=str, required=True)
    parser.add_argument("--path_to_ckpt", type=str, required=True)
    parser.add_argument("--desc_key", type=str, required=True)

    # Model args
    parser.add_argument("--model_name", type=str, default="modelo")
    parser.add_argument("--sequence_length", type=int, required=True)
    parser.add_argument("--num_classes", type=int, required=False, default=18)
    parser.add_argument("--lr", type=float, required=False, default=1e-5)
    parser.add_argument("--freeze_lm", type=int, required=True)
    parser.add_argument("--freeze_visual_spatial", type=int, required=True)
    parser.add_argument("--freeze_visual_temporal", type=int, required=True)

    # Trainer args
    parser.add_argument("--num_nodes", type=int, required=False, default=1)
    parser.add_argument("--accelerator", type=str, required=False, default="gpu")
    parser.add_argument("--strategy", type=str, required=False, default="ddp")
    parser.add_argument("--devices", type=int, required=True)

    parser.add_argument("--max_epochs", type=int, required=False, default=100)
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

    args = parser.parse_args()

    pl.seed_everything(42, workers=True)
    wand_logger = WandbLogger(
        offline=True,
        name=f"{args.model_name}_{args.sequence_length}f_{args.lr}",
        id=f"{args.model_name}_{args.sequence_length}f_{args.lr}",
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
                dirpath=f"./checkpoints/{args.model_name}_{args.sequence_length}f_{args.lr}",
                monitor="loss",
                mode="min",
            )
        ],
    )

    model = BaselineVLMClassifier(
        lr=args.lr,
        num_classes=args.num_classes,
        num_frames=args.sequence_length,
        path_to_ckpt=args.path_to_ckpt,
        freeze_lm=args.freeze_lm,
        freeze_visual_spatial=args.freeze_visual_spatial,
        freeze_visual_temporal=args.freeze_visual_temporal,
    )

    # Load tensor dataset
    print("Loading dataset...")
    with open(args.path_to_dataset, "rb") as f:
        dataset = pickle.load(f)
    print("Loaded!")

    train_dataset = dataset["train"]
    val_dataset = dataset["val"]
    test_dataset = dataset["test"]

    train_targets = [
        {"label": x[0], "desc": x[1]}
        for x in zip(train_dataset["label"], train_dataset[args.desc_key])
    ]

    val_targets = [
        {"label": x[0], "desc": x[1]}
        for x in zip(val_dataset["label"], val_dataset[args.desc_key])
    ]

    test_targets = [
        {"label": x[0], "desc": x[1]}
        for x in zip(test_dataset["label"], test_dataset[args.desc_key])
    ]

    transform = CustomTransform()
    formatter = CustomTargetFormatter()

    datamodule = VideoClassificationData.from_tensors(
        train_data=train_dataset["tensor"],
        train_targets=train_targets,
        val_data=val_dataset["tensor"],
        val_targets=val_targets,
        test_data=test_dataset["tensor"],
        test_targets=test_targets,
        transform=transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_formatter=formatter,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
