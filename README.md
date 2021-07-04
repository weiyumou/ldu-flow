---

<div align="center"> 

# Sinusoidal Flow

[![Paper](http://img.shields.io/badge/paper-link-B31B1B.svg)](https://www.dropbox.com/s/ztfoz2bpupbcndy/Sinusoidal_Flow_preprint.pdf?dl=0)

[comment]: <> ([![Conference]&#40;http://img.shields.io/badge/ACML-2021-4b44ce.svg&#41;]&#40;http://www.acml-conf.org/2021/&#41;)


[comment]: <> (![CI testing]&#40;https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push&#41;)


</div>

## Description

Normalising flows offer a flexible way of modelling continuous probability distributions. We consider expressiveness,
fast inversion and exact Jacobian determinant as three desirable properties a normalising flow should possess. However,
few flow models have been able to strike a good balance among all these properties. Realising that the integral of a
convex sum of sinusoidal functions squared leads to a bijective residual transformation, we propose Sinusoidal Flow, a
new type of normalising flows that inherits the expressive power and triangular Jacobian from fully autoregressive flows
while guaranteed by Banach fixed-point theorem to remain fast invertible and thereby obviate the need for sequential
inversion typically required in fully autoregressive flows. Experiments show that our Sinusoidal Flow is not only able
to model complex distributions, but can also be reliably inverted to generate realistic-looking samples even with many
layers of transformations stacked.

## How to run

First, install dependencies (Python 3.8 or above required)

```bash
# clone project   
git clone https://github.com/acml21-190/sinusoidal-flow.git

# install project   
cd sinusoidal-flow
pip install -e .
 ```   

Next, download the data files for UCI, MNIST and CIFAR-10
experiments [here](https://zenodo.org/record/1161203#.YCQJN5NKhTa), and place each data folder
under `project/datasets/maf/data`. For example, the data file for the POWER dataset should be accessible
from `project/datasets/maf/data/power/data.npy`.

### Reproducibility Check

You may do a quick reproducibility check for the results in Table 4 by testing checkpoints. First, download all the
checkpoints [here](https://1drv.ms/u/s!AqHACUDhxs0pgScZ7uErZBQxOl0w?e=3mVzEa) and place each checkpoint folder
corresponding to each dataset under `project/experiments/ldu/checkpoints`. For example, the ".ckpt" checkpoint file for
MNIST should be accessible under `project/experiments/ldu/checkpoints/mnist/version_21654323/checkpoints`.

Then you may either test the checkpoint for MNIST by running

```bash
python -m project.experiments.ldu.run_mnist_cifar10 --dataset mnist --test_checkpoints --gpus 0
```

or the checkpoint for CIFAR-10 by running

```bash
python -m project.experiments.ldu.run_mnist_cifar10 --dataset cifar10 --test_checkpoints --gpus 0
```

Use `--gpus 1` instead to test checkpoints on a single GPU.

[comment]: <> (### Model Training)

[comment]: <> (You may also run full training for the modules below. Add `--gpu` option to train on a single GPU. Please refer to each)

[comment]: <> (script for more command-line options.)

[comment]: <> ( ```bash)

[comment]: <> (# run_grid_gaussians_1d.py - reproduce parts of Figure 2 )

[comment]: <> (python -m project.experiments.run_grid_gaussians_1d --max_epochs 50)

[comment]: <> (# run_grid_gaussians_2d.py - reproduce parts of Figure 3)

[comment]: <> (python -m project.experiments.run_grid_gaussians_2d --modes_per_dim 5 --num_train 60000 --embed_dim 8 --conditioner ind --num_layers 8 --max_epochs 50)

[comment]: <> (# run_shapes_2d.py - reproduce parts of Figure 4)

[comment]: <> (python -m project.experiments.run_shapes_2d --num_train 90000 --embed_dim 4 --conditioner msk --num_layers 12 --max_epochs 50)

[comment]: <> (# run_uci.py - reproduce Table 2)

[comment]: <> (python -m project.experiments.run_uci --dataset power --embed_dim 4 --num_layers 16 --hid_dims 100 100 --weight_decay 0.0 --lr_decay 0.97 --max_epochs 100)

[comment]: <> (# run_mnist_cifar10.py - train models for MNIST)

[comment]: <> (python -m project.experiments.run_mnist_cifar10 --dataset mnist --conditioner atn --embed_dim 4 --num_layers 16 --multiplier 4 --weight_decay 0.0 --lr_decay 0.99 --batch_size 100 --max_epochs 230 --num_runs 5)

[comment]: <> (```)

[comment]: <> (## Imports)

[comment]: <> (This project is setup as a package which means you can now easily import any file into any other file like so:)

[comment]: <> (```python)

[comment]: <> (from project.datasets.mnist import mnist)

[comment]: <> (from project.lit_classifier_main import LitClassifier)

[comment]: <> (from pytorch_lightning import Trainer)

[comment]: <> (# model)

[comment]: <> (model = LitClassifier&#40;&#41;)

[comment]: <> (# data)

[comment]: <> (train, val, test = mnist&#40;&#41;)

[comment]: <> (# train)

[comment]: <> (trainer = Trainer&#40;&#41;)

[comment]: <> (trainer.fit&#40;model, train, val&#41;)

[comment]: <> (# test using the best model!)

[comment]: <> (trainer.test&#40;test_dataloaders=test&#41;)

[comment]: <> (```)

[comment]: <> (### Citation)

[comment]: <> (```)

[comment]: <> (@article{YourName,)

[comment]: <> (  title={Your Title},)

[comment]: <> (  author={Your team},)

[comment]: <> (  journal={Location},)

[comment]: <> (  year={Year})

[comment]: <> (})

[comment]: <> (```   )
