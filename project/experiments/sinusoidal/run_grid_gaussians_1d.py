import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.optim as optim
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader

from project.models.affine import AffineFlow
from project.models.sinusoidal import SinusoidalFlow


class GridGaussian1D(pl.LightningDataModule):
    def __init__(self, num_modes, batch_size, num_train):
        super(GridGaussian1D, self).__init__()

        self.modes_per_dim = num_modes
        self.batch_size = batch_size
        self.num_train = num_train
        self.dims, self.data, self.datasets = None, None, None
        self.low, self.high = None, None

    def setup(self, stage=None):
        means = torch.linspace(-40, 40, self.modes_per_dim)
        stds = torch.ones(self.modes_per_dim)

        mix = D.Categorical(torch.ones(self.modes_per_dim) / self.modes_per_dim)
        comp = D.Normal(means, stds)

        self.data = D.MixtureSameFamily(mix, comp)
        self.low, self.high = -50.0, 50.0

        total = self.num_train + 10000
        train, val, test = torch.split(self.data.sample((total, 1, 1, 1)), (self.num_train, 5000, 5000), dim=0)

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


class SinusoidalFlowForGridGaussian1D(pl.LightningModule):
    def __init__(self, hparams):
        super(SinusoidalFlowForGridGaussian1D, self).__init__()

        self.save_hyperparameters(hparams)
        self.sinusoidal_flow = SinusoidalFlow(self.hparams.c, self.hparams.h, self.hparams.w,
                                              self.hparams.embed_dim, self.hparams.conditioner, self.hparams.num_layers,
                                              affine=self.hparams.affine)
        # self.sinusoidal_flow = AffineFlow(self.hparams.c, self.hparams.h, self.hparams.w, self.hparams.conditioner, 2)

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

        parser.add_argument("--num_modes", default=7, type=int, help="Number of mixture components")
        parser.add_argument("--batch_size", default=128, type=int)
        parser.add_argument("--num_train", default=30000, type=int, help="Number of training points")

        parser.add_argument("--embed_dim", default=16, type=int, help="Number of parallel sinusoidal units")
        parser.add_argument("--conditioner", default="ind", type=str, help="The conditioner to use")
        parser.add_argument("--num_layers", default=8, type=int, help="Number of sinusoidal transformers")
        parser.add_argument("--affine", action="store_true", help="Whether to insert affine transformers")
        parser.add_argument("--gpu", action="store_true", help="Whether to use GPU for training")

        return parser


@torch.no_grad()
def plot_for_pub(device, true_model, model, low=-5, high=5, npts=300):
    model.eval()

    x = torch.linspace(low, high, npts).reshape(-1, 1)
    z, log_probs = model(x.reshape(-1, 1, 1, 1).to(device))
    z = z.reshape(-1, 1).cpu()
    log_probs = log_probs.reshape(-1, 1).cpu()

    true_log_probs = true_model.log_prob(x)
    kl_div = torch.sum(true_log_probs.exp() * (true_log_probs - log_probs)).item()

    true_z = model.base_dist.icdf(true_model.cdf(x).to(device))
    true_z_low = model.base_dist.mean - 3 * model.base_dist.stddev
    true_z_high = model.base_dist.mean + 3 * model.base_dist.stddev
    z_mask = (true_z_low <= true_z) & (true_z <= true_z_high)
    rmse = torch.mean((true_z[z_mask] - z[z_mask]) ** 2).sqrt().item()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), squeeze=False)

    axes[0, 0].plot(x[:, 0], true_log_probs[:, 0].exp(), color="#1A85FF", label="True")
    axes[0, 0].plot(x[:, 0], log_probs[:, 0].exp(), color="#D41159", linestyle="--", label="Fitted")

    axes[0, 0].set_title(f"KL Div = {kl_div:.4f}", fontsize=15)
    axes[0, 0].set_xlabel(r"$x$", fontsize=15)
    axes[0, 0].set_ylabel(r"$p(x)$", fontsize=15)
    axes[0, 0].tick_params(axis="both", which="major", labelsize=15)
    axes[0, 0].tick_params(axis="both", which="minor", labelsize=15)
    axes[0, 0].set_ylim(top=0.07)
    axes[0, 0].legend(fontsize=12)

    axes[0, 1].plot(x[:, 0], true_z[:, 0], color="#1A85FF", label="True")
    axes[0, 1].plot(x[:, 0], z[:, 0], color="#D41159", linestyle="--", label="Fitted")
    axes[0, 1].axhline(true_z_low, color="k")
    axes[0, 1].axhline(true_z_high, color="k")
    axes[0, 1].set_title(f"RMSE = {rmse:.4f}", fontsize=15)
    axes[0, 1].set_xlabel(r"$x$", fontsize=15)
    axes[0, 1].set_ylabel(r"$z$", fontsize=15)
    axes[0, 1].tick_params(axis="both", which="major", labelsize=15)
    axes[0, 1].tick_params(axis="both", which="minor", labelsize=15)
    axes[0, 1].legend(fontsize=12)

    plt.show()


def main(args):
    dm = GridGaussian1D(args.num_modes, args.batch_size, args.num_train)
    dm.setup()
    args.c, args.h, args.w = dm.size()

    model = SinusoidalFlowForGridGaussian1D(args)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback], gpus=int(args.gpu))
    trainer.fit(model, datamodule=dm)
    trainer.test(datamodule=dm)
    model = SinusoidalFlowForGridGaussian1D.load_from_checkpoint(checkpoint_callback.best_model_path).eval()

    plot_for_pub(model.device, dm.data, model.sinusoidal_flow, dm.low, dm.high, npts=1000)


if __name__ == "__main__":
    pl.seed_everything(42)

    parser = ArgumentParser(add_help=False)

    # add args from trainer
    parser = pl.Trainer.add_argparse_args(parser)

    # give the module a chance to add own params
    # good practice to define LightningModule specific params in the module
    parser = SinusoidalFlowForGridGaussian1D.add_model_specific_args(parser)

    # parse params
    args = parser.parse_args()

    main(args)
