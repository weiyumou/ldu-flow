from typing import Tuple

import torch
from torch import nn as nn
from torch.distributions import Normal

from project.models.normaliser import DiffPointwiseActNorm


def validate_cond_var(x: torch.Tensor, cond_var: torch.Tensor = None):
    if cond_var is None:
        return True

    l1 = len(cond_var.size()[:-1])
    l2 = len(x.size()[:-3])

    return (l1 <= l2) and cond_var.size()[:-1] == x.size()[:-3][-l1:]


class IdentityTransformer(nn.Module):
    """
    Executes an identity transformation. log_det = 0.
    """

    def __init__(self):
        super(IdentityTransformer, self).__init__()

        self.register_buffer("log_det", torch.zeros(1))

    def forward(self, x: torch.Tensor, cond_var=None):  # (*, C, H, W)
        return x, self.log_det.expand_as(x)

    @torch.no_grad()
    def inv_transform(self, z: torch.Tensor, cond_var=None, rtol=1e-5, atol=1e-8, max_iter=5000):  # (*, C, H, W)
        assert not self.training, "inv_transform() cannot be called during training. "
        return z


class FlipTransformer(nn.Module):
    """
    Flips the input. log_det = 0.
    """

    def __init__(self):
        super(FlipTransformer, self).__init__()

        self.register_buffer("log_det", torch.zeros(1))

    def forward(self, x: torch.Tensor, cond_var=None):  # (*, C, H, W)
        return torch.flip(x, [-3, -2, -1]), self.log_det.expand_as(x)

    @torch.no_grad()
    def inv_transform(self, z: torch.Tensor, cond_var=None, rtol=1e-5, atol=1e-8, max_iter=5000):  # (*, C, H, W)
        assert not self.training, "inv_transform() cannot be called during training. "
        return torch.flip(z, [-3, -2, -1])


class IndependentConditioner(nn.Module):
    """
    Computes "input-agnostic" parameters
    """

    def __init__(self, c: int, h: int, w: int):
        super(IndependentConditioner, self).__init__()

        self.log_std = nn.Parameter(torch.empty(c, h, w))
        self.mean = nn.Parameter(torch.empty_like(self.log_std))

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.log_std)
        nn.init.zeros_(self.mean)

    def forward(self, x: torch.Tensor):  # (*, D)
        return self.log_std, self.mean


class MLPConditioner(nn.Module):
    """
    Computes parameters based on some external cond_var
    """

    def __init__(self, c: int, h: int, w: int, in_dim=1, hid_dims=tuple(), use_ind_std=False):
        super(MLPConditioner, self).__init__()

        assert isinstance(hid_dims, tuple), "hid_dims must be a tuple. "

        out_dim = ((not use_ind_std) + 1) * c * h * w
        hid_dims += (out_dim,)

        mlp = nn.ModuleList([nn.Linear(in_features=in_dim, out_features=hid_dims[0])])
        for i in range(1, len(hid_dims)):
            mlp.extend([nn.ReLU(),
                        nn.Linear(in_features=hid_dims[i - 1], out_features=hid_dims[i])])
        self.mlp = nn.Sequential(*mlp)
        self.log_std = nn.Parameter(torch.zeros(c, h, w)) if use_ind_std else None
        self.chw = c, h, w

    def forward(self, x: torch.Tensor):  # (*, D)
        out = self.mlp(x).reshape(*x.size()[:-1], -1, *self.chw)  # (*, -1, C, H, W)

        if self.log_std is None:
            log_std, mean = torch.unbind(out, dim=-4)  # (*, C, H, W)
        else:
            log_std, mean = self.log_std, out.squeeze(dim=-4)  # (C, H, W) (*, C, H, W)

        return log_std, mean


class MaskedLinear(nn.Linear):
    def __init__(self, flow_dim, in_features, out_features, mask_type=None, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)

        if mask_type == "input":
            in_degrees = torch.arange(in_features) % flow_dim
        else:
            in_degrees = torch.arange(in_features) % (flow_dim - 1)

        if mask_type == "output":
            out_degrees = torch.arange(out_features) % flow_dim - 1
        else:
            out_degrees = torch.arange(out_features) % (flow_dim - 1)

        mask = torch.ge(out_degrees.unsqueeze(-1), in_degrees.unsqueeze(0)).float()
        self.register_buffer("mask", mask)
        self.mask_type = mask_type

    def forward(self, x):
        return nn.functional.linear(x, self.weight * self.mask, self.bias)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, mask_type=\"{self.mask_type}\""


class MaskedMLPConditioner(nn.Module):
    """
    Computes parameters autoregressively based on the input
    """

    def __init__(self, c: int, h: int, w: int, hid_dims=(4,)):
        super(MaskedMLPConditioner, self).__init__()

        assert isinstance(hid_dims, tuple), "hid_dims must be a tuple. "

        flow_dim = c * h * w
        out_dim = 2 * flow_dim
        hid_dims += (out_dim,)

        mlp = nn.ModuleList([MaskedLinear(flow_dim, in_features=flow_dim, out_features=hid_dims[0], mask_type="input")])
        for i in range(1, len(hid_dims)):
            mask_type = None if i < len(hid_dims) - 1 else "output"
            mlp.extend([nn.ReLU(),
                        MaskedLinear(flow_dim, in_features=hid_dims[i - 1], out_features=hid_dims[i],
                                     mask_type=mask_type)])
        self.mlp = nn.Sequential(*mlp)
        self.chw = c, h, w

    def forward(self, x: torch.Tensor):  # (*, C, H, W)
        x = torch.flatten(x, start_dim=-3)  # (*, CHW)
        batch_dims = x.size()[:-1]
        out = self.mlp(x).reshape(*batch_dims, 2, *self.chw)  # (*, 2, C, H, W)
        log_std, mean = torch.unbind(out, dim=-4)

        return log_std, mean


class AffineTransformer(nn.Module):
    def __init__(self, c: int, h: int, w: int, conditioner: str, **kwargs):
        super(AffineTransformer, self).__init__()

        if conditioner == "ind":
            self.conditioner = IndependentConditioner(c, h, w)
        elif conditioner == "mlp":
            self.conditioner = MLPConditioner(c, h, w, **kwargs)
        elif conditioner == "msk":
            self.conditioner = MaskedMLPConditioner(c, h, w, **kwargs)
        else:
            self.conditioner = None
            raise ValueError("Invalid conditioner. ")

    def forward(self, x: torch.Tensor, cond_var: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: input variable of shape (*, C, H, W)
        :param cond_var: condition variable of shape (*, D) or x itself
        :return: the transformed variable z and the log determinant log_dets
        """
        assert validate_cond_var(x, cond_var), "Your cond_var is not compatible in shape with x. "
        cond_var = x if cond_var is None else cond_var

        log_std, mean = self.conditioner(cond_var)
        z = torch.exp(-log_std) * (x - mean)
        log_dets = -torch.sum(log_std, dim=(-3, -2, -1)).expand(*x.size()[:-3])
        return z, log_dets

    @torch.no_grad()
    def inv_transform(self, z: torch.Tensor, cond_var: torch.Tensor = None, rtol=1e-5, atol=1e-8, max_iter=5000):
        """
        :param z: transformed variable of shape (*, C, H, W)
        :param cond_var: condition variable of shape (*, D) or z itself
        :return: the input variable
        :param rtol: ignored
        :param atol: ignored
        :param max_iter: ignored
        """
        assert not self.training, "inv_transform() cannot be called during training. "
        assert validate_cond_var(z, cond_var), "Your cond_var is not compatible in shape with z. "

        cond_var = z if cond_var is None else cond_var
        log_std, mean = self.conditioner(cond_var)

        return torch.exp(log_std) * z + mean  # (*, C, H, W)


class AffineFlow(nn.Module):
    def __init__(self, c: int, h: int, w: int, conditioner: str, num_layers: int, use_norm=False, **kwargs):
        super(AffineFlow, self).__init__()

        self.transformers = nn.ModuleList([AffineTransformer(c, h, w, conditioner, **kwargs)])
        for _ in range(num_layers - 1):
            self.transformers.extend([
                DiffPointwiseActNorm(c, h, w) if use_norm else IdentityTransformer(),
                FlipTransformer(),
                AffineTransformer(c, h, w, conditioner, **kwargs),
            ])

        self.norm = DiffPointwiseActNorm(c, h, w)
        self.base_dist = Normal(0.0, 1.0)

    def forward(self, x: torch.Tensor, cond_var: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: input variable of shape (*, C, H, W)
        :param cond_var: the conditional variable of shape (*, D)
        :return: the transformed variable and the joint density
        """
        sum_log_dets = []

        for layer in self.transformers:
            x, log_dets = layer(x, cond_var)
            sum_log_dets.append(log_dets)

        z, log_dets = self.norm(x)  # (*, C, H, W)
        sum_log_dets.append(log_dets)

        sum_log_dens = torch.sum(self.base_dist.log_prob(z), dim=(-3, -2, -1))
        sum_log_dets = torch.stack(sum_log_dets, dim=0).sum(dim=0)
        log_joint_dens = sum_log_dens + sum_log_dets

        return z, log_joint_dens

    @torch.no_grad()
    def inv_transform(self, z: torch.Tensor, cond_var: torch.Tensor = None) -> torch.Tensor:
        """
        :param z: transformed variable of shape (*, C, H, W)
        :param cond_var: the conditional variable of shape (*, D)
        :return: the input variable of shape (*, C, H, W)
        """
        assert not self.training, "inv_transform() cannot be called during training. "

        z = self.norm.inv_transform(z, cond_var)
        for layer in reversed(self.transformers):
            z = layer.inv_transform(z, cond_var)
        return z
