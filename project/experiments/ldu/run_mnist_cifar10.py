import glob
import math
import os
import statistics
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.optim as optim
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader

from project.datasets.maf.cifar10 import CIFAR10
from project.datasets.maf.mnist import MNIST
from project.models.ldu import LDUFlow


class ImageData(pl.LightningDataModule):
    def __init__(self, dataset, batch_size):
        super(ImageData, self).__init__()

        self.dataset = dataset
        self.batch_size = batch_size
        self.dims, self.data, self.datasets = None, None, None

    def setup(self, stage=None):
        if self.dataset == "mnist":
            self.data = MNIST(logit=True, dequantize=True)
            self.dims = (1, 28, 28)
        elif self.dataset == "cifar10":
            self.data = CIFAR10(logit=True, flip=True, dequantize=True)
            self.dims = (3, 32, 32)
        else:
            raise ValueError("Invalid dataset. ")

        self.datasets = {
            "train": TensorDataset(torch.from_numpy(self.data.trn.x).reshape(self.data.trn.N, *self.dims)),
            "val": TensorDataset(torch.from_numpy(self.data.val.x).reshape(self.data.val.N, *self.dims)),
            "test": TensorDataset(torch.from_numpy(self.data.tst.x).reshape(self.data.tst.N, *self.dims))
        }

    def train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=4, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.datasets["val"], batch_size=self.batch_size,
                          num_workers=4, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=4, shuffle=False, pin_memory=True)


class SinusoidalFlowForImageData(pl.LightningModule):
    def __init__(self, hparams):
        super(SinusoidalFlowForImageData, self).__init__()

        self.save_hyperparameters(hparams)
        self.sinusoidal_flow = LDUFlow(self.hparams.c, self.hparams.h, self.hparams.w,
                                       self.hparams.embed_dim, self.hparams.conditioner,
                                       self.hparams.num_layers, self.hparams.num_d_trans,
                                       affine=self.hparams.affine,
                                       hid_dims=self.hparams.hid_dims,  # dropout=self.hparams.dropout,
                                       attn_size=self.hparams.attn_size,
                                       num_fmaps=self.hparams.num_fmaps, num_blocks=self.hparams.num_blocks)

        self.chw = self.hparams.c * self.hparams.h * self.hparams.w

    def training_step(self, batch, batch_idx):
        _, log_probs = self.sinusoidal_flow(batch[0])
        loss = -torch.mean(log_probs)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        _, log_probs = self.sinusoidal_flow(x)

        bit_per_dim = -log_probs / (self.chw * math.log(2)) - math.log2(1 - 2 * self.hparams.alpha) + 8
        bit_per_dim += torch.mean(torch.log2(torch.sigmoid(x)) + torch.log2(1 - torch.sigmoid(x)), dim=(-3, -2, -1))

        self.log("val/loss", -torch.mean(log_probs))
        self.log("val/bpd", bit_per_dim)

    def test_step(self, batch, batch_idx):
        x = batch[0]
        _, log_probs = self.sinusoidal_flow(x)

        bit_per_dim = -log_probs / (self.chw * math.log(2)) - math.log2(1 - 2 * self.hparams.alpha) + 8
        bit_per_dim += torch.mean(torch.log2(torch.sigmoid(x)) + torch.log2(1 - torch.sigmoid(x)), dim=(-3, -2, -1))

        self.log("test/loss", -torch.mean(log_probs))
        self.log("test/bpd", bit_per_dim)

    def configure_optimizers(self):
        if self.hparams.adamw:
            optimiser = optim.AdamW(self.sinusoidal_flow.parameters(),
                                    lr=self.hparams.lr,
                                    weight_decay=self.hparams.weight_decay)
        else:
            optimiser = optim.Adam(self.sinusoidal_flow.parameters(),
                                   lr=self.hparams.lr,
                                   weight_decay=self.hparams.weight_decay)

        ret_dict = {"optimizer": optimiser}

        if self.hparams.anneal_lr:

            # compute total training steps
            if self.hparams.max_steps is not None:
                train_steps = self.hparams.max_steps
            else:
                num_devices = max(1, 0 if self.hparams.gpus is None else self.hparams.gpus) * self.hparams.num_nodes
                num_batches = len(self.trainer.datamodule.train_dataloader()) // num_devices
                train_steps = (self.hparams.max_epochs * num_batches) // self.hparams.accumulate_grad_batches

            ret_dict["lr_scheduler"] = {"scheduler": CosineAnnealingLR(optimiser, train_steps), "interval": "step"}

        elif self.hparams.lr_decay is not None:
            ret_dict["lr_scheduler"] = {
                "scheduler": ReduceLROnPlateau(optimiser, mode="min", factor=self.hparams.lr_decay,
                                               patience=2, min_lr=1e-4),
                "monitor": "val_bpd"
            }

        return ret_dict

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # overall architecture
        parser.add_argument("--embed_dim", default=4, type=int, help="Number of units in a sinusoidal transformer")
        parser.add_argument("--conditioner", default="cnn", type=str, help="The conditioner to use to create shifts")
        parser.add_argument("--num_layers", default=12, type=int, help="Number of LDU transformers to use")
        parser.add_argument("--num_d_trans", default=4, type=int, help="Number of sinusoidal transformers in one LDU")
        parser.add_argument("--affine", action="store_true", help="Whether to insert affine transformers")

        # conditioner-specific
        parser.add_argument("--hid_dims", default=(1024,), type=int, nargs="+", help="Sizes of the hidden layers")
        parser.add_argument("--dropout", default=0.0, type=float, help="Dropout probability")
        parser.add_argument("--attn_size", default=4, type=int, help="Size of the attention vectors")
        parser.add_argument("--num_fmaps", default=16, type=int, help="Base number of feature maps to use in CNN")
        parser.add_argument("--num_blocks", default=4, type=int, help="Number of residual blocks in CNN")

        # optimisation
        parser.add_argument("--adamw", action="store_true", help="Whether to use AdamW optimiser")
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay rate")
        parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
        parser.add_argument("--lr_decay", default=None, type=float, help="Learning rate decay")
        parser.add_argument("--anneal_lr", action="store_true", help="Whether to cosine-anneal LR")

        return parser


def main(args):
    dm = ImageData(args.dataset, args.batch_size)
    dm.setup()
    args.c, args.h, args.w = dm.size()
    args.alpha = dm.data.alpha
    args.hid_dims = tuple(args.hid_dims)

    if args.resume is not None:
        args.num_runs = 1

    test_losses = []
    test_bpds = []

    if args.test_checkpoints:
        checkpoint_path = os.path.join(os.path.dirname(__file__), f"checkpoints/{args.dataset}")
        for checkpoint in glob.iglob(f"{checkpoint_path}/**/*.ckpt", recursive=True):
            print(f"Testing: {checkpoint}")
            model = SinusoidalFlowForImageData.load_from_checkpoint(checkpoint)
            trainer = pl.Trainer(resume_from_checkpoint=checkpoint, gpus=args.gpus)
            test_results = trainer.test(model, datamodule=dm)[0]
            test_loss, test_bpd = test_results["test/loss"], test_results["test/bpd"]
            test_losses.append(test_loss)
            test_bpds.append(test_bpd)
    else:
        for run in range(args.num_runs):
            print(f"Run {run}")

            model = SinusoidalFlowForImageData(args)

            callbacks = [ModelCheckpoint(monitor="val/loss")]
            if (args.lr_decay is not None) or args.anneal_lr:
                callbacks.append(LearningRateMonitor())

            if args.resume is None:
                trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
            else:
                trainer = pl.Trainer(resume_from_checkpoint=args.resume, gpus=args.gpus,
                                     max_epochs=args.max_epochs, max_steps=args.max_steps)

            trainer.fit(model, datamodule=dm)
            test_results = trainer.test(datamodule=dm)[0]
            test_loss, test_bpd = test_results["test/loss"], test_results["test/bpd"]
            test_losses.append(test_loss)
            test_bpds.append(test_bpd)

            print(f"Run {run} test loss = {test_loss}")
            print(f"Run {run} test bpd = {test_bpd}")

    print(f"mean of test_loss over {args.num_runs} runs = {statistics.mean(test_losses)}")
    print(f"std of test_loss over {args.num_runs} runs = {statistics.pstdev(test_losses)}")

    print(f"mean of test_bpd over {args.num_runs} runs = {statistics.mean(test_bpds)}")
    print(f"std of test_bpd over {args.num_runs} runs = {statistics.pstdev(test_bpds)}")


if __name__ == "__main__":
    pl.seed_everything(42)

    parser = ArgumentParser(add_help=False)

    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_runs", default=1, type=int, help="Number of runs to perform")
    parser.add_argument("--resume", default=None, type=str, help="The checkpoint to resume training from")
    parser.add_argument("--test_checkpoints", action="store_true", help="Whether to test checkpoints")

    # add args from trainer
    parser = pl.Trainer.add_argparse_args(parser)

    # give the module a chance to add own params
    # good practice to define LightningModule specific params in the module
    parser = SinusoidalFlowForImageData.add_model_specific_args(parser)

    # parse params
    args = parser.parse_args()

    main(args)
