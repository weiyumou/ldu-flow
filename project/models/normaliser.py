import torch
import torch.nn as nn


class PointwiseBatchNorm(nn.Module):
    def __init__(self, c, h, w, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(PointwiseBatchNorm, self).__init__()

        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.empty(c, h, w))
            self.bias = nn.Parameter(torch.empty(c, h, w))
            self._init_weights()

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(c, h, w))
            self.register_buffer("running_var", torch.ones(c, h, w))
            self.register_buffer("batch_count", torch.tensor([0], dtype=torch.long))

    def _init_weights(self):
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, cond_var=None):
        """
        :param x: input variable of shape (*, C, H, W)
        :param cond_var: ignored
        :return: the normalised output
        """
        with torch.no_grad():
            batch_mean = torch.mean(x, dim=tuple(range(-x.ndim, -3)))
            batch_var_biased = torch.var(x, dim=tuple(range(-x.ndim, -3)), unbiased=False)
            batch_var_unbiased = torch.var(x, dim=tuple(range(-x.ndim, -3)), unbiased=True)

        y = (x - batch_mean) / torch.sqrt(batch_var_biased + self.eps)

        if self.track_running_stats:
            if self.training:
                self.batch_count += 1
                momentum = self.momentum if self.momentum is not None else 1.0 / self.batch_count.item()
                self.running_mean = (1 - momentum) * self.running_mean + momentum * batch_mean
                self.running_var = (1 - momentum) * self.running_var + momentum * batch_var_unbiased
            else:
                y = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        return torch.exp(self.weight) * y + self.bias if self.affine else y

    @torch.no_grad()
    def inv_transform(self, z, cond_var=None):
        """
        :param z: normalised variable of shape (*, C, H, W)
        :param cond_var: ignored
        :return: the input variable of shape (*, C, H, W)
        """
        assert not self.training, "inv_transform() cannot be called during training. "
        assert self.track_running_stats, "inv_transform() cannot be called if running stats weren't tracked. "

        if self.affine:
            z = torch.exp(-self.weight) * (z - self.bias)
        return torch.sqrt(self.running_var + self.eps) * z + self.running_mean


class DiffPointwiseBatchNorm(PointwiseBatchNorm):
    def forward(self, x, cond_var=None):
        """
        :param x: input variable of shape (*, C, H, W)
        :param cond_var: ignored
        :return: the normalised output
        """
        z = super().forward(x)

        if self.track_running_stats and not self.training:
            log_dets = -torch.log(self.running_var + self.eps) / 2
        else:
            log_dets = -torch.log(torch.var(x.detach(), dim=tuple(range(-x.ndim, -3)), unbiased=False) + self.eps) / 2

        if self.affine:
            log_dets = torch.sum(self.weight + log_dets)

        log_dets = log_dets.expand(*x.size()[:-3])
        return z, log_dets


class PointwiseActNorm(nn.Module):
    def __init__(self, c, h, w, eps=1e-5):
        super(PointwiseActNorm, self).__init__()

        self.mean = nn.Parameter(torch.zeros(c, h, w))
        self.log_std = nn.Parameter(torch.zeros(c, h, w))
        self.eps = eps
        self.register_buffer("initialised", torch.tensor(False, dtype=torch.bool))

    def forward(self, x, cond_var=None):
        """
        :param x: input variable of shape (*, C, H, W)
        :param cond_var: ignored
        :return: the normalised output
        """
        if not self.initialised and self.training:
            with torch.no_grad():
                batch_mean = torch.mean(x, dim=tuple(range(-x.ndim, -3)))
                batch_var = torch.var(x, dim=tuple(range(-x.ndim, -3)), unbiased=False)

            self.mean.data = batch_mean
            self.log_std.data = torch.log(batch_var + self.eps) / 2
            self.initialised = ~self.initialised

        return torch.exp(-self.log_std) * (x - self.mean)

    @torch.no_grad()
    def inv_transform(self, z, cond_var=None):
        """
        :param z: normalised variable of shape (*, C, H, W)
        :param cond_var: ignored
        :return: the input variable of shape (*, C, H, W)
        """
        assert not self.training, "inv_transform() cannot be called during training. "

        return torch.exp(self.log_std) * z + self.mean


class DiffPointwiseActNorm(PointwiseActNorm):
    def forward(self, x, cond_var=None):
        """
        :param x: input variable of shape (*, C, H, W)
        :param cond_var: ignored
        :return: the normalised output
        """
        z = super().forward(x)
        log_dets = -torch.sum(self.log_std).expand(*x.size()[:-3])
        return z, log_dets


def get_normaliser(normaliser, c, h, w, **kwargs):
    if normaliser == "batchnorm":
        return PointwiseBatchNorm(c, h, w, **kwargs)
    elif normaliser == "actnorm":
        return PointwiseActNorm(c, h, w, **kwargs)
    elif normaliser == "diffbatchnorm":
        return DiffPointwiseBatchNorm(c, h, w, **kwargs)
    elif normaliser == "diffactnorm":
        return DiffPointwiseActNorm(c, h, w, **kwargs)
    return nn.Identity()


def get_use_norm(normaliser):
    return normaliser in ["batchnorm", "actnorm", "diffbatchnorm", "diffactnorm"]
