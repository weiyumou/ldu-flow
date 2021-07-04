import torch
import torch.nn as nn
from torch.distributions import Normal

from project.models.affine import AffineTransformer, IdentityTransformer, validate_cond_var, FlipTransformer
from project.models.normaliser import DiffPointwiseActNorm


class IndependentConditioner(nn.Module):
    """
        Computes "input-agnostic" parameters
    """

    def __init__(self, c: int, h: int, w: int, embed_dim: int):
        super(IndependentConditioner, self).__init__()

        self.in_proj_weight = nn.Parameter(torch.empty(c, h, w, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty_like(self.in_proj_weight))
        self.out_proj_weight = nn.Parameter(torch.empty_like(self.in_proj_weight))
        self.out_proj_bias = nn.Parameter(torch.empty(c, h, w))
        self.residual_weight = nn.Parameter(torch.empty_like(self.out_proj_bias))

        self.softplus = nn.Softplus()
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.in_proj_weight, -3, 3)
        nn.init.uniform_(self.in_proj_bias, -3, 3)
        nn.init.zeros_(self.out_proj_weight)
        nn.init.zeros_(self.out_proj_bias)
        nn.init.zeros_(self.residual_weight)

    def forward(self, x):
        in_proj_weight = self.softplus(self.in_proj_weight)
        return in_proj_weight, self.in_proj_bias, self.out_proj_weight, self.out_proj_bias, self.residual_weight


class MLPConditioner(nn.Module):
    """
    Computes parameters based on some external cond_var
    """

    def __init__(self, c: int, h: int, w: int, embed_dim: int, in_dim=1, hid_dims=tuple(), **kwargs):
        super(MLPConditioner, self).__init__()

        assert isinstance(hid_dims, tuple), "hid_dims must be a tuple. "

        flow_dim = c * h * w
        out_dim = 3 * flow_dim * embed_dim + 2 * flow_dim
        hid_dims += (out_dim,)

        mlp = nn.ModuleList([nn.Linear(in_features=in_dim, out_features=hid_dims[0])])
        for i in range(1, len(hid_dims)):
            mlp.extend([nn.ReLU(),
                        nn.Linear(in_features=hid_dims[i - 1], out_features=hid_dims[i])])
        self.mlp = nn.Sequential(*mlp)
        self.chw = c, h, w
        self.flow_dim = flow_dim
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor):  # (*, D)
        batch_dims = x.size()[:-1]
        out = self.mlp(x)

        weights = out[..., :-2 * self.flow_dim].reshape(*batch_dims, *self.chw, 3, -1)  # (*, C, H, W, 3, E)
        in_proj_weight, in_proj_bias, out_proj_weight = torch.unbind(weights, dim=-2)
        in_proj_weight = self.softplus(in_proj_weight)

        out_proj_bias, residual_weight = torch.unbind(out[..., -2 * self.flow_dim:].reshape(*batch_dims, *self.chw, 2),
                                                      dim=-1)  # (*, C, H, W)

        return in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias, residual_weight


class SinusoidalTransformer(nn.Module):
    def __init__(self, c: int, h: int, w: int, embed_dim: int, conditioner: str, **kwargs):
        super(SinusoidalTransformer, self).__init__()

        if conditioner == "ind":
            self.conditioner = IndependentConditioner(c, h, w, embed_dim)
        elif conditioner == "mlp":
            self.conditioner = MLPConditioner(c, h, w, embed_dim, **kwargs)
        else:
            self.conditioner = None
            raise ValueError("Invalid conditioner. ")

        self._inverting = False

    def forward(self, x: torch.Tensor, cond_var: torch.Tensor = None):
        """
        :param x: input variable of shape (*, C, H, W)
        :param cond_var: the conditional variable of shape (*, D) or x itself
        :return: the transformed variable of shape (*, C, H, W) and the log determinant of shape (*, )
        """
        assert validate_cond_var(x, cond_var), "Your cond_var is not compatible in shape with x. "

        cond_var = x if cond_var is None else cond_var

        in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias, residual_weight = self.conditioner(cond_var)
        mix_weights = torch.softmax(out_proj_weight, dim=-1)
        residual_weight = torch.tanh(residual_weight)

        u = in_proj_weight * x.unsqueeze(dim=-1) + in_proj_bias
        y = (torch.sin(2 * in_proj_bias) - torch.sin(2 * u)) / (2 * in_proj_weight)
        y = x + residual_weight * torch.sum(mix_weights * y, dim=-1) + out_proj_bias

        log_dets = None
        if not self._inverting:
            dy_dx = -residual_weight * torch.sum(mix_weights * torch.cos(2 * u), dim=-1)  # (*, C, H, W)
            log_dets = torch.log1p(dy_dx).sum(dim=(-3, -2, -1))

        return y, log_dets

    @torch.no_grad()
    def inv_transform(self, z, cond_var=None, rtol=1e-5, atol=1e-8, max_iter=100):
        assert not self.training, "inv_transform() cannot be called during training. "
        assert validate_cond_var(z, cond_var), "Your cond_var is not compatible in shape with z. "

        self._inverting = True

        num_iter = 1
        x_k = z
        x_kp1 = z - (self(x_k, cond_var)[0] - x_k)
        while num_iter < max_iter and not torch.allclose(x_kp1, x_k, rtol=rtol, atol=atol):
            x_k = x_kp1
            x_kp1 = z - (self(x_k, cond_var)[0] - x_k)
            num_iter += 1

        self._inverting = False

        return x_kp1


class SinusoidalFlow(nn.Module):
    def __init__(self, c: int, h: int, w: int, embed_dim: int,
                 conditioner: str, num_layers: int, affine=False, **kwargs):
        super(SinusoidalFlow, self).__init__()

        self.transformers = nn.ModuleList([SinusoidalTransformer(c, h, w, embed_dim, conditioner, **kwargs)])
        for _ in range(num_layers - 1):
            self.transformers.extend([
                AffineTransformer(c, h, w, conditioner="ind") if affine else IdentityTransformer(),
                FlipTransformer() if conditioner != "ind" else IdentityTransformer(),
                SinusoidalTransformer(c, h, w, embed_dim, conditioner, **kwargs)
            ])

        self.norm = DiffPointwiseActNorm(c, h, w)
        self.base_dist = Normal(0.0, 1.0)
        self.chw = c, h, w

    def forward(self, x, cond_var=None):
        sum_log_dets = []

        for layer in self.transformers:
            x, log_dets = layer(x, cond_var)
            sum_log_dets.append(log_dets)

        z, log_dets = self.norm(x)
        sum_log_dets.append(log_dets)

        sum_log_dens = torch.sum(self.base_dist.log_prob(z), dim=(-3, -2, -1))
        sum_log_dets = torch.stack(sum_log_dets, dim=0).sum(dim=0)
        log_joint_dens = sum_log_dens + sum_log_dets

        return z, log_joint_dens

    @torch.no_grad()
    def inv_transform(self, z, cond_var=None, rtol=1e-5, atol=1e-8, max_iter=100):
        assert not self.training, "inv_transform() cannot be called during training. "

        z = self.norm.inv_transform(z, cond_var)
        for layer in reversed(self.transformers):
            z = layer.inv_transform(z, cond_var, rtol, atol, max_iter)
        return z

    @torch.no_grad()
    def sample(self, sample_size, temp=1.0, cond_var=None, batch_size=128, rtol=1e-10, atol=1e-16, max_iter=100):
        assert not self.training, "sample() cannot be called during training. "

        device = self.norm.log_std.device
        self.double()

        samples = []
        z = self.base_dist.sample((sample_size, *self.chw)) * temp
        for batch in torch.split(z, batch_size, dim=0):
            batch = batch.to(device, torch.double)
            y = self.inv_transform(batch, cond_var, rtol, atol, max_iter)
            samples.append(y.cpu().float())
        samples = torch.cat(samples, dim=0)

        self.float()
        return samples
