import math
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.optim as optim
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader

from project.models.sinusoidal import SinusoidalFlow


# Ref: https://github.com/CW-Huang/torchkit/blob/master/torchkit/model_maf_toy.py
class GridGaussian2D(pl.LightningDataModule):
    def __init__(self, modes_per_dim, batch_size, num_train):
        super(GridGaussian2D, self).__init__()

        self.modes_per_dim = modes_per_dim
        self.batch_size = batch_size
        self.num_train = num_train
        self.dims, self.data, self.datasets = None, None, None
        self.low, self.high = None, None

    def setup(self, stage=None):
        side = torch.linspace(-5, 5, self.modes_per_dim)
        xx, yy = torch.meshgrid([side, side])
        means = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=-1)  # (modes_per_dim, 2)
        covs = torch.eye(2).expand(self.modes_per_dim ** 2, -1, -1) / (
                self.modes_per_dim * math.log(self.modes_per_dim))

        mix = D.Categorical(torch.ones(self.modes_per_dim ** 2) / self.modes_per_dim ** 2)
        comp = D.MultivariateNormal(means, covs)

        self.data = D.MixtureSameFamily(mix, comp)
        self.low, self.high = -10.0, 10.0

        total = self.num_train + 10000
        train, val, test = torch.split(self.data.sample((total, 1, 1)), (self.num_train, 5000, 5000), dim=0)

        self.dims = val[0].size()
        self.datasets = {
            "train": TensorDataset(train),
            "val": TensorDataset(val),
            "test": TensorDataset(test)
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


class SinusoidalFlowForGridGaussian2D(pl.LightningModule):
    def __init__(self, hparams):
        super(SinusoidalFlowForGridGaussian2D, self).__init__()

        self.save_hyperparameters(hparams)
        self.sinusoidal_flow = SinusoidalFlow(self.hparams.c, self.hparams.h, self.hparams.w,
                                              self.hparams.embed_dim, self.hparams.conditioner, self.hparams.num_layers,
                                              affine=self.hparams.affine)

    def training_step(self, batch, batch_idx):
        _, log_probs = self.sinusoidal_flow(batch[0])
        loss = -torch.mean(log_probs)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        _, log_probs = self.sinusoidal_flow(batch[0])
        loss = -torch.mean(log_probs)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        _, log_probs = self.sinusoidal_flow(batch[0])
        loss = -torch.mean(log_probs)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimiser = optim.Adam(self.sinusoidal_flow.parameters(), lr=1e-3)
        return {
            "optimizer": optimiser,
            "lr_scheduler": ReduceLROnPlateau(optimiser, factor=0.5, patience=5, verbose=True),
            "monitor": "val_loss"
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--modes_per_dim", default=5, type=int, help="Number of mixture components per dim")
        parser.add_argument("--batch_size", default=128, type=int)
        parser.add_argument("--num_train", default=60000, type=int, help="Number of training points")

        parser.add_argument("--embed_dim", default=4, type=int, help="Number of sinusoidal units in a layer")
        parser.add_argument("--conditioner", default="ind", type=str, help="The conditioner to use")
        parser.add_argument("--num_layers", default=8, type=int, help="Number of sinusoidal transformers to use")
        parser.add_argument("--affine", action="store_true", help="Whether to insert affine transformers")
        parser.add_argument("--gpu", action="store_true", help="Whether to use GPU for training")

        return parser


@torch.no_grad()
def plot_for_pub(device, true_model, model, low=-5, high=5, npts=300):
    side = torch.linspace(low, high, npts)
    xx, yy = torch.meshgrid([side, side])
    x = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=-1).to(device)

    model.eval()
    z, log_probs = model(x.reshape(-1, 1, 1, 2))
    log_probs = log_probs.reshape(npts, npts).cpu()

    x = x.cpu()
    true_log_probs = true_model.log_prob(x).reshape(npts, npts)

    kl_div = torch.sum(true_log_probs.exp() * (true_log_probs - log_probs)).item()

    fig, axes = plt.subplots(2, 1, figsize=(6, 12), squeeze=False, constrained_layout=True)

    axes[0, 0].set_aspect("equal")
    axes[0, 0].pcolormesh(xx, yy, true_log_probs.exp(), shading="auto")
    axes[0, 0].axis("off")

    axes[1, 0].set_aspect("equal")
    axes[1, 0].pcolormesh(xx, yy, log_probs.exp(), shading="auto")
    axes[1, 0].axis("off")

    fig.suptitle(f"KL Div = {kl_div:.4f}", fontsize=25)
    plt.show()


def main(args):
    """
    nn.init.uniform_(self.in_proj_weight, -3, 3)
    nn.init.uniform_(self.in_proj_bias, -3, 3)
    10x10 grid: embed_dim=4, conditioner="ind", num_layers=12, max_epochs=40; 90000 training points
    5x5 grid: embed_dim=4, conditioner="ind", num_layers=8, max_epochs=50; 60000 training points
    2x2 grid: embed_dim=4, conditioner="ind", num_layers=4, max_epochs=30; 30000 training points
    """
    dm = GridGaussian2D(args.modes_per_dim, args.batch_size, args.num_train)
    dm.setup()
    args.c, args.h, args.w = dm.size()

    model = SinusoidalFlowForGridGaussian2D(args)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback], gpus=int(args.gpu))
    trainer.fit(model, datamodule=dm)
    trainer.test(datamodule=dm)
    model = SinusoidalFlowForGridGaussian2D.load_from_checkpoint(checkpoint_callback.best_model_path).eval()

    plot_for_pub(model.device, dm.data, model.sinusoidal_flow, dm.low, dm.high)


if __name__ == "__main__":
    pl.seed_everything(42)

    parser = ArgumentParser(add_help=False)

    # add args from trainer
    parser = pl.Trainer.add_argparse_args(parser)

    # give the module a chance to add own params
    # good practice to define LightningModule specific params in the module
    parser = SinusoidalFlowForGridGaussian2D.add_model_specific_args(parser)

    # parse params
    args = parser.parse_args()

    main(args)
