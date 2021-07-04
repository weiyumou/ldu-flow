import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.optim as optim
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau

from project.data import ToyShapesData2D
from project.models.ldu import LDUFlow


class SinusoidalFlowForShapes2D(pl.LightningModule):
    def __init__(self, hparams):
        super(SinusoidalFlowForShapes2D, self).__init__()

        self.save_hyperparameters(hparams)

        self.sinusoidal_flow = LDUFlow(self.hparams.c, self.hparams.h, self.hparams.w,
                                       self.hparams.embed_dim, self.hparams.conditioner,
                                       self.hparams.num_layers, self.hparams.num_d_trans,
                                       affine=self.hparams.affine, hid_dims=(100,))

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
        parser.add_argument("--num_layers", default=16, type=int, help="Number of LDU transformers to use")
        parser.add_argument("--num_d_trans", default=4, type=int, help="Number of sinusoidal transformers in one LDU")
        parser.add_argument("--affine", action="store_true", help="Whether to insert affine transformers")

        return parser


@torch.no_grad()
def plot_for_pub(device, true_model, model, low=-5, high=5, npts=300):
    fig, axes = plt.subplots(3, 1, figsize=(4, 12), squeeze=False, constrained_layout=True)

    # plot true samples
    true_samples = true_model.sample(npts * npts)
    axes[0, 0].set_aspect("equal")
    axes[0, 0].hist2d(true_samples[:, 0], true_samples[:, 1], range=[[low, high], [low, high]], bins=npts)
    axes[0, 0].axis("off")

    # plot predicted density
    side = torch.linspace(low, high, npts)
    xx, yy = torch.meshgrid([side, side])
    x = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=-1).to(device)
    model.eval()
    _, log_probs = model(x.reshape(-1, 1, 1, 2))
    log_probs = log_probs.reshape(npts, npts).cpu()
    axes[1, 0].set_aspect("equal")
    axes[1, 0].pcolormesh(xx, yy, log_probs.exp(), shading="auto")
    axes[1, 0].axis("off")

    pred_samples = model.sample(npts * npts, max_iter=100).reshape(-1, 2).numpy()
    axes[2, 0].set_aspect("equal")
    axes[2, 0].hist2d(pred_samples[:, 0], pred_samples[:, 1], range=[[low, high], [low, high]], bins=npts)
    axes[2, 0].axis("off")

    plt.show()


def main(args):
    """
    --num_train
    90000
    --embed_dim
    4
    --conditioner
    msk
    --num_layers
    16
    --num_d_trans
    4
    --max_epochs
    70
    """
    for dataset in ["swissroll", "moons", "pinwheel", "circles", "8gaussians", "2spirals", "checkerboard"]:
        dm = ToyShapesData2D(dataset, args.batch_size, args.num_train)
        dm.setup()
        args.c, args.h, args.w = dm.size()

        model = SinusoidalFlowForShapes2D(args)

        checkpoint_callback = ModelCheckpoint(monitor="val_loss")
        trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
        trainer.fit(model, datamodule=dm)
        # trainer.test(datamodule=dm)
        model = SinusoidalFlowForShapes2D.load_from_checkpoint(checkpoint_callback.best_model_path).eval()

        plot_for_pub(model.device, dm.data, model.sinusoidal_flow, dm.low, dm.high)


if __name__ == "__main__":
    pl.seed_everything(42)

    parser = ArgumentParser(add_help=False)

    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_train", default=90000, type=int, help="Number of training points")

    # add args from trainer
    parser = pl.Trainer.add_argparse_args(parser)

    # give the module a chance to add own params
    # good practice to define LightningModule specific params in the module
    parser = SinusoidalFlowForShapes2D.add_model_specific_args(parser)

    # parse params
    args = parser.parse_args()

    main(args)
