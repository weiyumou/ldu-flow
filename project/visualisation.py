import matplotlib.pyplot as plt
import torch


def plot_2dflow(model, ax, low=-4, high=4, npts=300, title=r"$p(x_{1}, x_{2})$"):
    device = model.device if hasattr(model, "device") else "cpu"
    side = torch.linspace(low, high, npts)
    xx, yy = torch.meshgrid([side, side])
    x = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=-1).reshape(-1, 1, 1, 2).to(device)

    with torch.no_grad():
        probs = model.log_prob(x).reshape(npts, npts).exp().cpu()

    ax.set_aspect("equal")
    ax.pcolormesh(xx, yy, probs, shading="auto")
    cs = ax.contour(xx, yy, probs, colors="black")
    ax.clabel(cs)

    # ax.set_xlim(low, high)
    # ax.set_ylim(low, high)

    # ax.invert_yaxis()
    # ax.get_xaxis().set_ticks([])
    # ax.get_yaxis().set_ticks([])
    ax.set_title(title)


def plot_marginals(model, ax, i, c, h, w, level=0.0, low=-4, high=4, npts=300):
    device = model.device if hasattr(model, "device") else "cpu"
    r = torch.linspace(low, high, npts, device=device)
    x = torch.full((npts, c * h * w), level, device=device)
    x[:, i] = r

    with torch.no_grad():
        log_probs = model.log_prob(x.reshape(npts, c, h, w))

    ax.plot(r.cpu().flatten(), log_probs.exp().cpu().flatten())
    ax.set_xlim(low, high)
    title = f"$p(x_{i + 1})$"
    ax.set_title(title)


@torch.no_grad()
def plt_samples_2d(device, true_model, model, low=-5, high=5, npts=300):
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

    # Generate samples
    pred_samples = model.sample(10, max_iter=40).reshape(-1, 2).numpy()
    axes[2, 0].set_aspect("equal")
    axes[2, 0].hist2d(pred_samples[:, 0], pred_samples[:, 1], range=[[low, high], [low, high]], bins=npts)
    axes[2, 0].axis("off")

    plt.show()


@torch.no_grad()
def plot_kl_div_2d(device, true_model, model, low=-5, high=5, npts=300, contour=True):
    side = torch.linspace(low, high, npts)
    xx, yy = torch.meshgrid([side, side])
    x = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=-1).to(device)

    model.eval()
    z, log_probs = model(x.reshape(-1, 1, 1, 2))
    log_probs = log_probs.reshape(npts, npts).cpu()

    x = x.cpu()
    true_log_probs = true_model.log_prob(x).reshape(npts, npts)

    kl_div = torch.sum(true_log_probs.exp() * (true_log_probs - log_probs)).item()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), squeeze=False)

    axes[0, 0].set_aspect("equal")
    axes[0, 0].pcolormesh(xx, yy, true_log_probs.exp(), shading="auto")
    if contour:
        cs = axes[0, 0].contour(xx, yy, true_log_probs.exp(), colors="black")
        axes[0, 0].clabel(cs)
    axes[0, 0].set_title("True")

    axes[0, 1].set_aspect("equal")
    axes[0, 1].pcolormesh(xx, yy, log_probs.exp(), shading="auto")
    if contour:
        cs = axes[0, 1].contour(xx, yy, log_probs.exp(), colors="black")
        axes[0, 1].clabel(cs)
    axes[0, 1].set_title("Predicted")

    fig.suptitle(f"KL Div = {kl_div:.4f}")
    plt.show()


@torch.no_grad()
def plot_kl_div_1d(device, true_model, model, low=-5, high=5, npts=300):
    model.eval()

    x = torch.linspace(low, high, npts).reshape(-1, 1).to(device)

    z, log_probs = model(x.reshape(-1, 1, 1, 1))
    z = z.reshape(-1, 1).cpu()
    log_probs = log_probs.reshape(-1, 1).cpu()
    x = x.cpu()

    true_log_probs = true_model.log_prob(x)
    true_z = model.base_dist.icdf(true_model.cdf(x))
    true_z_low = model.base_dist.mean - 3 * model.base_dist.stddev
    true_z_high = model.base_dist.mean + 3 * model.base_dist.stddev
    z_mask = (true_z_low <= true_z) & (true_z <= true_z_high)

    kl_div = torch.sum(true_log_probs.exp() * (true_log_probs - log_probs)).item()
    rmse = torch.mean((true_z[z_mask] - z[z_mask]) ** 2).sqrt().item()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), squeeze=False)

    axes[0, 0].plot(x[:, 0], true_log_probs[:, 0].exp(), color="b", label="True")
    axes[0, 0].plot(x[:, 0], log_probs[:, 0].exp(), color="r", linestyle="--", label="Fitted")

    # axes[0, 0].set_title("Density Estimation")
    axes[0, 0].set_title(f"KL Div = {kl_div:.4f}")
    axes[0, 0].set_xlabel(r"$x$")
    axes[0, 0].set_ylabel(r"$p(x)$")
    axes[0, 0].legend()

    axes[0, 1].plot(x[:, 0], true_z[:, 0], color="b", label="True")
    axes[0, 1].plot(x[:, 0], z[:, 0], color="r", linestyle="--", label="Fitted")
    axes[0, 1].axhline(true_z_low, color="k")
    axes[0, 1].axhline(true_z_high, color="k")
    # axes[0, 1].set_title("Transformation Estimation")
    axes[0, 1].set_title(f"RMSE = {rmse:.4f}")
    axes[0, 1].set_xlabel(r"$x$")
    axes[0, 1].set_ylabel(r"$z$")
    axes[0, 1].legend()

    # modes = model.calc_modes(torch.randn(1, 1))
    # for item in modes:
    #     if item:
    #         for mode in item[0].tolist():
    #             axes[0, 0].axvline(mode, color="k", linestyle=":")
    #             print(mode)

    # fig.suptitle(f"KL Div = {kl_div:.4f}")

    plt.show()
