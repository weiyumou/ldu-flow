from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from torchvision.utils import make_grid

from project.experiments.ldu.run_mnist_cifar10 import ImageData, SinusoidalFlowForImageData


def display_samples(dm, samples):
    images = make_grid(samples, padding=0).permute([1, 2, 0]).numpy()
    if images.shape[-1] == 1:
        images = images[..., 0]

    images = dm.data.inv_logit_transform(images)

    plt.imshow(images, interpolation="nearest")
    plt.axis("off")
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{dm.dataset}.png")


def display_errors(means, stds):
    means, stds = torch.tensor(means), torch.tensor(stds)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(len(means)), means, color="#1A85FF")
    ax.fill_between(range(len(means)), means - stds, means + stds, color="#1A85FF", alpha=0.2)
    ax.set_xlabel("# Iterations")
    ax.set_ylabel(r"Normalised $\ell_{2}$ Rec. Error")
    ax.set_yscale("log")
    # plt.show()
    plt.savefig("recon.png")


def display_recons(dm, x, xp):
    images = torch.cat([x, xp], dim=0).cpu()
    images = make_grid(images, nrow=10, padding=0).permute([1, 2, 0]).numpy()
    if images.shape[-1] == 1:
        images = images[..., 0]

    images = dm.data.inv_logit_transform(images)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(images, interpolation="nearest")
    ax.axis("off")
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{dm.dataset}_recon.png")


def main(args):
    dm = ImageData(args.dataset, args.batch_size)
    dm.setup()
    args.c, args.h, args.w = dm.size()
    args.alpha = dm.data.alpha

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    model = SinusoidalFlowForImageData.load_from_checkpoint(args.checkpoint).to(device)
    model.eval()

    if args.sample:
        samples = model.sinusoidal_flow.sample(args.sample_size, temp=args.temp,
                                               rtol=args.rtol, atol=args.atol, max_iter=args.max_iter)
        display_samples(dm, samples)

    if args.recon:
        x = next(iter(dm.test_dataloader()))[0].to(device)
        with torch.no_grad():
            z = model.sinusoidal_flow(x)[0]

        # generate error curve
        means, stds = [], []
        for max_iter in range(1, args.max_iter + 1):
            xp = model.sinusoidal_flow.inv_transform(z, rtol=args.rtol, atol=args.atol, max_iter=max_iter)
            norm_l2_err = torch.mean((x - xp) ** 2, dim=(-3, -2, -1)).cpu()
            means.append(norm_l2_err.mean().item())
            stds.append(norm_l2_err.std().item())
            print(f"max_iter = {max_iter}: {means[-1]}")

        display_errors(means, stds)

        # compare real images with reconstructions
        xp = model.sinusoidal_flow.inv_transform(z[:10], rtol=args.rtol, atol=args.atol, max_iter=100)
        display_recons(dm, x[:10], xp)


if __name__ == "__main__":
    pl.seed_everything(42)

    parser = ArgumentParser(add_help=False)

    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--checkpoint", required=True, type=str, help="Relative path to the model to be inverted")
    parser.add_argument("--sample_size", default=16, type=int, help="Number of samples to generate")
    parser.add_argument("--temp", default=1.0, type=float, help="Temperature used to sample")
    parser.add_argument("--rtol", default=1e-5, type=float, help="Relative tolerance")
    parser.add_argument("--atol", default=1e-8, type=float, help="Absolute tolerance")
    parser.add_argument("--max_iter", default=100, type=int, help="Maximum number of iterations per layer")
    parser.add_argument("--gpu", action="store_true", help="Whether to use GPU")
    parser.add_argument("--sample", action="store_true", help="Whether to generate samples")
    parser.add_argument("--recon", action="store_true", help="Whether to compute reconstruction errors")

    # parse params
    args = parser.parse_args()

    main(args)
