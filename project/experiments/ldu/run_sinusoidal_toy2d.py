import pytorch_lightning as pl
import torch
import torch.optim as optim
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau

from project.data import ToyDensityData2D
from project.models.ldu import LDUFlow
from project.visualisation import plot_kl_div_2d


class SinusoidalFlowForToy2D(pl.LightningModule):
    def __init__(self, hparams):
        super(SinusoidalFlowForToy2D, self).__init__()

        self.save_hyperparameters(hparams)
        self.sinusoidal_flow = LDUFlow(self.hparams.c, self.hparams.h, self.hparams.w,
                                       self.hparams.embed_dim, self.hparams.conditioner,
                                       self.hparams.num_layers, self.hparams.num_d_trans,
                                       affine=self.hparams.affine, hid_dims=(32,))

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

        parser.add_argument("--embed_dim", default=4, type=int, help="Number of units in a sinusoidal transformer")
        parser.add_argument("--conditioner", default="msk", type=str, help="The conditioner to use to create shifts")
        parser.add_argument("--num_layers", default=3, type=int, help="Number of LDU transformers to use")
        parser.add_argument("--num_d_trans", default=4, type=int, help="Number of sinusoidal transformers in one LDU")
        parser.add_argument("--affine", action="store_true", help="Whether to insert affine transformers")

        return parser


def main(args):
    dm = ToyDensityData2D(args.dataset, args.batch_size)
    dm.setup()
    args.c, args.h, args.w = dm.size()

    model = SinusoidalFlowForToy2D(args)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule=dm)
    trainer.test(datamodule=dm)
    model = SinusoidalFlowForToy2D.load_from_checkpoint(checkpoint_callback.best_model_path).eval()

    plot_kl_div_2d(model.device, dm.data, model.sinusoidal_flow, dm.low, dm.high)


if __name__ == "__main__":
    pl.seed_everything(42)

    parser = ArgumentParser(add_help=False)

    parser.add_argument("--dataset", required=True, type=str, help="The toy dataset to use")
    parser.add_argument("--batch_size", default=128, type=int)

    # add args from trainer
    parser = pl.Trainer.add_argparse_args(parser)

    # give the module a chance to add own params
    # good practice to define LightningModule specific params in the module
    parser = SinusoidalFlowForToy2D.add_model_specific_args(parser)

    # parse params
    args = parser.parse_args()

    main(args)
