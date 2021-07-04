import torch
from torch import nn as nn
from typing import Tuple

from project.models.affine import MaskedLinear, FlipTransformer, IdentityTransformer
from project.models.pixelcnn import PixelCNN
from project.models.sinusoidal import SinusoidalTransformer, SinusoidalFlow


class MaskedMLPConditioner(nn.Module):
    def __init__(self, c: int, h: int, w: int, hid_dims=(4,), dropout=0.0, **kwargs):
        super(MaskedMLPConditioner, self).__init__()

        assert isinstance(hid_dims, tuple), "hid_dims must be a tuple. "

        flow_dim = c * h * w
        out_dim = flow_dim
        hid_dims += (out_dim,)

        mlp = nn.ModuleList([MaskedLinear(flow_dim, in_features=flow_dim, out_features=hid_dims[0], mask_type="input")])
        for i in range(1, len(hid_dims)):
            mask_type = None if i < len(hid_dims) - 1 else "output"
            mlp.extend([
                nn.Dropout(dropout),
                nn.ReLU(),
                MaskedLinear(flow_dim, in_features=hid_dims[i - 1], out_features=hid_dims[i], mask_type=mask_type)
            ])
        self.mlp = nn.Sequential(*mlp)
        self.chw = c, h, w

    def forward(self, x: torch.Tensor):  # (*, C, H, W)
        x = torch.flatten(x, start_dim=-3)  # (*, CHW)
        batch_dims = x.size()[:-1]
        shift = self.mlp(x).reshape(*batch_dims, *self.chw)  # (*, C, H, W)

        return shift


class LinearAttentionConditioner(nn.Module):

    def __init__(self, c, h, w, attn_size=4, **kwargs):
        super(LinearAttentionConditioner, self).__init__()

        self.in_proj_weight = nn.Parameter(torch.empty(3, c, h, w, attn_size))
        self.in_proj_bias = nn.Parameter(torch.empty_like(self.in_proj_weight))

        self.hid_proj_weight = nn.Parameter(torch.empty(attn_size, c * h * w, attn_size))
        self.hid_proj_bias = nn.Parameter(torch.empty(attn_size, c * h * w))

        self.out_proj_weight = nn.Parameter(torch.empty(attn_size, c * h * w))
        self.out_proj_bias = nn.Parameter(torch.empty(c * h * w))

        self.softplus = nn.Softplus()
        self.act_fn = nn.Tanh()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)

        nn.init.xavier_uniform_(self.hid_proj_weight)
        nn.init.zeros_(self.hid_proj_bias)

        nn.init.xavier_uniform_(self.out_proj_weight)
        nn.init.zeros_(self.out_proj_bias)

    def forward(self, x: torch.Tensor):  # (N, C, H, W)
        n, c, h, w = x.size()
        x = x.unsqueeze(dim=1).unsqueeze(dim=-1)  # (N, 1, C, H, W, 1)

        q, k, v = torch.chunk(self.in_proj_weight * x + self.in_proj_bias, 3, dim=1)
        q = q.reshape(q.size(0), -1, q.size(-1))
        k = k.reshape(k.size(0), -1, k.size(-1))
        v = v.reshape(v.size(0), -1, v.size(-1))

        q, k = self.softplus(q), self.softplus(k)
        s = torch.einsum("nsi,nsj->nsij", k, v)  # batch outer product
        s = torch.cumsum(s, dim=1)  # (N, S, E, E)
        z = torch.cumsum(k, dim=1)

        numer = torch.einsum("nsi,nsij->nsj", q, s)  # (N, S, E)
        denom = torch.einsum("nsi,nsi->ns", q, z).unsqueeze(dim=-1)  # (N, S, 1)
        y = numer / denom  # (N, S, E)

        # Strictly autoregressive
        y = torch.cat([torch.ones_like(y[:, [0]]), y[:, :-1]], dim=1)

        y = torch.sum(self.hid_proj_weight * y.unsqueeze(dim=1), dim=-1) + self.hid_proj_bias
        y = torch.sum(self.out_proj_weight * self.act_fn(y), dim=1) + self.out_proj_bias
        shift = y.reshape(n, c, h, w)

        return shift


class PixelCNNConditioner(nn.Module):
    def __init__(self, c: int, h: int, w: int, num_fmaps=16, num_blocks=4, **kwargs):
        super(PixelCNNConditioner, self).__init__()

        self.pixel_cnn = PixelCNN(c, c, num_fmaps, num_blocks, h, w, normaliser="actnorm")

    def forward(self, x: torch.Tensor):  # (N, C, H, W)
        shift = self.pixel_cnn(x)  # (N, C, H, W)

        return shift


class LowerShiftTransformer(nn.Module):
    def __init__(self, c: int, h: int, w: int, conditioner: str, **kwargs):
        super(LowerShiftTransformer, self).__init__()

        if conditioner == "msk":
            self.conditioner = MaskedMLPConditioner(c, h, w, **kwargs)
        elif conditioner == "atn":
            self.conditioner = LinearAttentionConditioner(c, h, w, **kwargs)
        elif conditioner == "cnn":
            self.conditioner = PixelCNNConditioner(c, h, w, **kwargs)
        else:
            self.conditioner = None
            raise ValueError(f"Invalid conditioner '{conditioner}'. ")

    def forward(self, x: torch.Tensor, cond_var: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: input variable of shape (*, C, H, W)
        :param cond_var: ignored
        :return: the transformed variable z and the log determinant log_dets
        """

        shift = self.conditioner(x)
        z = x + shift

        return z, x.new_zeros(x.size()[:-3])

    @torch.no_grad()
    def inv_transform(self, z, cond_var=None, rtol=1e-5, atol=1e-8, max_iter=100):
        assert not self.training, "inv_transform() cannot be called during training. "

        num_iter = 1
        x_k = z
        x_kp1 = z - (self(x_k, cond_var)[0] - x_k)
        while num_iter < max_iter and not torch.allclose(x_kp1, x_k, rtol=rtol, atol=atol):
            x_k = x_kp1
            x_kp1 = z - (self(x_k, cond_var)[0] - x_k)
            num_iter += 1

        return x_kp1


class UpperShiftTransformer(LowerShiftTransformer):

    def forward(self, x: torch.Tensor, cond_var: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: input variable of shape (*, C, H, W)
        :param cond_var: ignored
        :return: the transformed variable z and the log determinant log_dets
        """

        x = torch.flip(x, dims=(-3, -2, -1))
        shift = self.conditioner(x)
        z = torch.flip(x + shift, dims=(-3, -2, -1))

        return z, x.new_zeros(x.size()[:-3])


class LDUTransformer(nn.Module):
    def __init__(self, c: int, h: int, w: int, embed_dim: int, conditioner: str, num_d_trans: int, **kwargs):
        super(LDUTransformer, self).__init__()

        self.u_shift = UpperShiftTransformer(c, h, w, conditioner, **kwargs)
        self.d_scale = nn.ModuleList(
            [SinusoidalTransformer(c, h, w, embed_dim, "ind", **kwargs) for _ in range(num_d_trans)]
        )
        self.l_shift = LowerShiftTransformer(c, h, w, conditioner, **kwargs)

    def forward(self, x: torch.Tensor, cond_var: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: input variable of shape (*, C, H, W)
        :param cond_var: condition variable of shape (*, D) or x itself
        :return: the transformed variable z and the log determinant log_dets
        """

        # U-transform
        y, _ = self.u_shift(x)

        # D-transform
        sum_log_dets = []
        for layer in self.d_scale:
            y, log_dets = layer(y, cond_var)
            sum_log_dets.append(log_dets)
        log_dets = torch.stack(sum_log_dets, dim=0).sum(dim=0)

        # L-transform
        z, _ = self.l_shift(y)

        return z, log_dets

    @torch.no_grad()
    def inv_transform(self, z, cond_var=None, rtol=1e-5, atol=1e-8, max_iter=100):
        assert not self.training, "inv_transform() cannot be called during training. "

        # inverse L-transform
        y = self.l_shift.inv_transform(z, cond_var, rtol, atol, max_iter)

        # inverse D-transform
        for layer in reversed(self.d_scale):
            y = layer.inv_transform(y, cond_var, rtol, atol, max_iter)

        # inverse U-transform
        x = self.u_shift.inv_transform(y, cond_var, rtol, atol, max_iter)

        return x


class LDUFlow(SinusoidalFlow):
    def __init__(self, c: int, h: int, w: int, embed_dim: int,
                 conditioner: str, num_layers: int, num_d_trans: int, affine=False, **kwargs):
        super(LDUFlow, self).__init__(c, h, w, embed_dim, "ind", num_layers, affine, **kwargs)

        for i in range(len(self.transformers)):
            if isinstance(self.transformers[i], SinusoidalTransformer):
                self.transformers[i] = LDUTransformer(c, h, w, embed_dim, conditioner, num_d_trans, **kwargs)
            elif isinstance(self.transformers[i], FlipTransformer):
                self.transformers[i] = IdentityTransformer()
