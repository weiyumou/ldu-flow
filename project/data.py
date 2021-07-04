import math

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.datasets import make_swiss_roll, make_circles, make_moons
from torch import distributions as D
from torch.utils.data import TensorDataset, DataLoader


class ToyDensityData1D(pl.LightningDataModule):
    def __init__(self, dataset, batch_size):
        super(ToyDensityData1D, self).__init__()

        self.dataset = dataset
        self.batch_size = batch_size
        self.dims, self.data, self.datasets = None, None, None
        self.low, self.high = None, None

    def setup(self, stage=None):
        if self.dataset == "gmm_2modes":
            mix = D.Categorical(torch.tensor([0.5, 0.2, 0.3]))
            comp = D.Normal(torch.tensor([-2.0, 1.0, 4.0]), torch.tensor([0.5, 2, 1.0]).sqrt())
            self.data = D.MixtureSameFamily(mix, comp)
            self.low, self.high = -5, 8
        elif self.dataset == "gmm_3modes":
            mix = D.Categorical(torch.tensor([0.29, 0.28, 0.43]))
            comp = D.Normal(torch.tensor([-2.75, -0.5, 3.64]), torch.tensor([0.06, 0.25, 1.63]).sqrt())
            self.data = D.MixtureSameFamily(mix, comp)
            self.low, self.high = -5, 8
        elif self.dataset == "sos_3equalmodes":
            mix = D.Categorical(torch.tensor([1.0, 1.0, 1.0]) / 3)  # low, high = -15, 10
            comp = D.Normal(torch.tensor([-5.0, 0.0, 5.0]), torch.tensor([1.0, 1.0, 1.0]).sqrt())
            self.data = D.MixtureSameFamily(mix, comp)
            self.low, self.high = -15, 10
        elif self.dataset == "sos_3unequalmodes":
            mix = D.Categorical(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]) / 5)
            comp = D.Normal(torch.tensor([-5.0, -2.0, 0.0, 2.0, 5.0]), torch.tensor([1.5, 2.0, 1.0, 2.0, 1.0]).sqrt())
            self.data = D.MixtureSameFamily(mix, comp)
            self.low, self.high = -15, 10
        elif self.dataset == "sos_2separated":
            mix = D.Categorical(torch.tensor([0.5, 0.5]))
            comp = D.Normal(torch.tensor([-10.0, 10.0]), torch.tensor([1.0, 1.0]).sqrt())
            self.data = D.MixtureSameFamily(mix, comp)
            self.low, self.high = -25, 25
        elif self.dataset == "sos_2moreseparated":
            mix = D.Categorical(torch.tensor([0.5, 0.5]))
            comp = D.Normal(torch.tensor([-20.0, 20.0]), torch.tensor([1.0, 1.0]).sqrt())
            self.data = D.MixtureSameFamily(mix, comp)
            self.low, self.high = -25, 25
        elif self.dataset == "sos_3separated":  # use larger embed_dim (e.g., 8) for well-separated cases
            mix = D.Categorical(torch.tensor([1.0, 1.0, 1.0]) / 3)
            comp = D.Normal(torch.tensor([-20.0, -5.0, 15.0]), torch.tensor([1.0, 1.0, 1.0]).sqrt())
            self.data = D.MixtureSameFamily(mix, comp)
            self.low, self.high = -25, 25
        elif self.dataset == "exponential":
            self.data = D.Exponential(1.0)
            self.low, self.high = 0, 4
        else:
            raise ValueError("Unknown dataset. ")

        train, val, test = torch.split(self.data.sample((30000, 1, 1, 1)), (20000, 5000, 5000), dim=0)

        self.dims = train[0].size()
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


class CondGaussian2D:

    def __init__(self):
        self.sigma_1 = 1.0
        self.mu_2 = 0.0
        self.sigma_2 = 2.0

    def sample(self, size):
        x2 = torch.normal(self.mu_2, self.sigma_2, size=size)
        x1 = torch.normal(x2 ** 2 / 4, self.sigma_1)
        return torch.stack([x1, x2], dim=-1)  # (*size, 2)

    def log_prob(self, x):  # (*, 2)
        mu_1 = x[..., 1] ** 2 / 4
        log_p2 = -math.log(2 * math.pi) / 2 - math.log(self.sigma_2) - ((x[..., 1] - self.mu_2) / self.sigma_2) ** 2 / 2
        log_p1 = -math.log(2 * math.pi) / 2 - math.log(self.sigma_1) - ((x[..., 0] - mu_1) / self.sigma_1) ** 2 / 2
        return log_p1 + log_p2


class GaussianMixture2D:
    def __init__(self):
        mix = D.Categorical(torch.tensor([0.4, 0.6]))
        comp = D.MultivariateNormal(torch.stack([torch.tensor([0.0, 2.0]), torch.tensor([0.0, -2.0])]),
                                    torch.stack([torch.diag(torch.ones(2, )), torch.tensor([[8.4, 2.0], [2.0, 1.7]])]))
        self.gmm = D.MixtureSameFamily(mix, comp)

    def sample(self, sample_shape=torch.Size([])):
        return self.gmm.sample(sample_shape)

    def log_prob(self, x):
        return self.gmm.log_prob(x)


class ToyDensityData2D(pl.LightningDataModule):
    def __init__(self, dataset, batch_size):
        super(ToyDensityData2D, self).__init__()

        self.dataset = dataset
        self.batch_size = batch_size
        self.dims, self.data, self.datasets = None, None, None
        self.low, self.high = None, None

    def setup(self, stage=None):
        if self.dataset == "diag_normal":
            self.data = D.MultivariateNormal(torch.tensor([0.0, 0.0]), torch.diag(torch.tensor([4.0, 4.0])))
            self.low, self.high = -5, 5
        elif self.dataset == "cond_normal":
            self.data = CondGaussian2D()
            self.low, self.high = -4.5, 5.5
        elif self.dataset == "gmm_2separated":
            mix = D.Categorical(torch.tensor([0.5, 0.5]))
            comp = D.MultivariateNormal(torch.stack([torch.tensor([-2.5, 0.0]), torch.tensor([2.5, 0.0])]),
                                        torch.stack([torch.diag(torch.ones(2, )), torch.diag(torch.ones(2, ))]))

            self.data = D.MixtureSameFamily(mix, comp)
            self.low, self.high = -5.0, 5.0
        elif self.dataset == "4gaussians":
            modes_per_dim = 2
            side = torch.linspace(-5, 5, modes_per_dim)
            xx, yy = torch.meshgrid([side, side])
            means = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=-1)  # (modes_per_dim, 2)
            covs = torch.eye(2).expand(modes_per_dim ** 2, -1, -1) / (2 * modes_per_dim)
            mix = D.Categorical(torch.ones(modes_per_dim ** 2) / modes_per_dim ** 2)
            comp = D.MultivariateNormal(means, covs)

            self.data = D.MixtureSameFamily(mix, comp)
            self.low, self.high = -8.0, 8.0
        else:
            raise ValueError("Unknown dataset. ")

        train, val, test = torch.split(self.data.sample((30000, 1, 1)), (20000, 5000, 5000), dim=0)

        self.dims = train[0].size()
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


# Ref: https://github.com/AWehenkel/UMNN/blob/master/lib/toy_data.py
class Shapes2D:
    def __init__(self, dataset):
        self.dataset = dataset

    def sample(self, sample_size):
        if self.dataset == "swissroll":
            data = make_swiss_roll(n_samples=sample_size, noise=1.0)[0]
            data = data.astype("float32")[:, [0, 2]]
            data /= 5

        elif self.dataset == "circles":
            data = make_circles(n_samples=sample_size, factor=.5, noise=0.08)[0]
            data = data.astype("float32")
            data *= 3

        elif self.dataset == "moons":
            data = make_moons(n_samples=sample_size, noise=0.1)[0]
            data = data.astype("float32")
            data = data * 2 + np.array([-1, -0.2])
            data = data.astype("float32")

        elif self.dataset == "8gaussians":
            scale = 4.
            centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                       (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                             1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
            centers = [(scale * x, scale * y) for x, y in centers]

            data = []
            for i in range(sample_size):
                point = np.random.randn(2) * 0.5
                idx = np.random.randint(8)
                center = centers[idx]
                point[0] += center[0]
                point[1] += center[1]
                data.append(point)
            data = np.array(data, dtype="float32")
            data /= 1.414

        elif self.dataset == "pinwheel":
            radial_std = 0.3
            tangential_std = 0.1
            num_classes = 5
            num_per_class = sample_size // 5
            rate = 0.25
            rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

            features = np.random.randn(num_classes * num_per_class, 2) \
                       * np.array([radial_std, tangential_std])
            features[:, 0] += 1.
            labels = np.repeat(np.arange(num_classes), num_per_class)

            angles = rads[labels] + rate * np.exp(features[:, 0])
            rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
            rotations = np.reshape(rotations.T, (-1, 2, 2))

            data = 2 * np.random.permutation(np.einsum("ti,tij->tj", features, rotations)).astype("float32")

        elif self.dataset == "2spirals":
            n = np.sqrt(np.random.rand(sample_size // 2, 1)) * 540 * (2 * np.pi) / 360
            d1x = -np.cos(n) * n + np.random.rand(sample_size // 2, 1) * 0.5
            d1y = np.sin(n) * n + np.random.rand(sample_size // 2, 1) * 0.5
            x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
            x += np.random.randn(*x.shape) * 0.1
            data = x.astype("float32")

        elif self.dataset == "checkerboard":
            x1 = np.random.rand(sample_size) * 4 - 2
            x2_ = np.random.rand(sample_size) - np.random.randint(0, 2, sample_size) * 2
            x2 = x2_ + (np.floor(x1) % 2)
            data = np.concatenate([x1[:, None], x2[:, None]], 1).astype("float32") * 2

        elif self.dataset == "line":
            x = np.random.rand(sample_size)
            x = x * 5 - 2.5
            y = x
            data = np.stack((x, y), 1).astype("float32")

        elif self.dataset == "line-noisy":
            x = np.random.rand(sample_size)
            x = x * 5 - 2.5
            y = x + np.random.randn(sample_size)
            data = np.stack((x, y), 1).astype("float32")

        elif self.dataset == "cos":
            x = np.random.rand(sample_size) * 5 - 2.5
            y = np.sin(x) * 2.5
            data = np.stack((x, y), 1).astype("float32")
        else:
            raise ValueError("Incorrect dataset name. ")

        return data


class ToyShapesData2D(pl.LightningDataModule):
    def __init__(self, dataset, batch_size, num_train):
        super(ToyShapesData2D, self).__init__()

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_train = num_train
        self.dims, self.data, self.datasets = None, None, None
        self.low, self.high = None, None

    def setup(self, stage=None):
        self.data = Shapes2D(self.dataset)
        self.low, self.high = -5.0, 5.0

        total = self.num_train + 10000
        train, val, test = torch.split(torch.from_numpy(self.data.sample(total)).reshape(-1, 1, 1, 2),
                                       (self.num_train, 5000, 5000), dim=0)

        self.dims = val[0].size()
        self.datasets = {
            "train": TensorDataset(train),
            "val": TensorDataset(val),
            "test": TensorDataset(test)
        }

    def train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=1, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.datasets["val"], batch_size=self.batch_size,
                          num_workers=1, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=1, shuffle=False, pin_memory=True)
