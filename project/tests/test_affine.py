import unittest

import torch
from torch.autograd.functional import jacobian

from project.models.affine import IndependentConditioner, AffineTransformer, AffineFlow


class IndependentConditionerTestCase(unittest.TestCase):

    def test_ind(self):
        c, h, w = 1, 1, 1
        m, n, d = 11, 7, 3
        x = torch.randn(m, n, d, requires_grad=True)
        model = IndependentConditioner(c, h, w)

        log_std, mean = model(x)
        dlog_std = torch.autograd.grad(log_std.sum(), x, create_graph=True, allow_unused=True)[0]
        dmean = torch.autograd.grad(mean.sum(), x, allow_unused=True)[0]

        self.assertIsInstance(log_std, torch.nn.Parameter)
        self.assertIsInstance(mean, torch.nn.Parameter)
        self.assertEqual(log_std.size(), (c, h, w))
        self.assertEqual(mean.size(), (c, h, w))
        self.assertIsNone(dlog_std, msg="log_std shouldn't depend on x. ")
        self.assertIsNone(dmean, msg="mean shouldn't depend on x. ")


class AffineTransformerTestCase(unittest.TestCase):

    def test_ind(self):
        c, h, w = 2, 3, 4
        m, n = 11, 7
        y = torch.randn(m, n, c, h, w, requires_grad=True)

        model = AffineTransformer(c, h, w, conditioner="ind")

        # Test transform
        z, log_dets = model(y)
        dz_dy = torch.autograd.grad(z.sum(), y)[0].log().sum(dim=(-3, -2, -1))

        self.assertIsInstance(z, torch.Tensor)
        self.assertIsInstance(log_dets, torch.Tensor)
        self.assertEqual(z.size(), y.size())
        self.assertEqual(log_dets.size(), dz_dy.size())
        self.assertEqual(log_dets.size(), (m, n))
        self.assertTrue(torch.isclose(log_dets, dz_dy).all())

        # Test inverse transform
        model.eval()
        yp = model.inv_transform(z)
        self.assertIsInstance(yp, torch.Tensor)
        self.assertEqual(yp.size(), y.size())
        self.assertTrue(torch.isclose(yp, y).all())

    def test_msk(self):
        c, h, w = 2, 3, 4
        n, d = 7, 3
        hid_dims = (36, 36)

        y = torch.randn(n, c, h, w, requires_grad=True) * 4

        model = AffineTransformer(c, h, w, conditioner="msk", hid_dims=hid_dims).train()

        # Test transform
        def f(a):
            return model(a)[0]

        jacob = jacobian(f, y)

        jacobs = []
        for b in range(n):
            jacobs.append(jacob[b, :, :, :, b, :, :, :].reshape(c * h * w, -1))
        jacobs = torch.stack(jacobs, dim=0)
        dz_dy = torch.logdet(jacobs)

        z, log_dets = model(y)

        self.assertIsInstance(z, torch.Tensor, f"z should be a torch.Tensor. ")
        self.assertIsInstance(log_dets, torch.Tensor, f"log_dets should be a torch.Tensor. ")
        self.assertEqual(z.size(), y.size(), f"z and y should have equal size. ")
        self.assertEqual(log_dets.size(), dz_dy.size(), f"log_dets and dz_dy should have equal size. ")
        self.assertEqual(log_dets.size(), (n,), f"log_dets should have size {(n,)}")
        self.assertTrue(torch.allclose(log_dets, dz_dy, rtol=1e-4, atol=1e-6), f"log_dets and dz_dy should be equal. ")


class AffineFlowTestCase(unittest.TestCase):

    def test_ind(self):
        c, h, w = 2, 3, 4
        m, n = 11, 7
        num_layers = 3

        for use_norm in [True, False]:
            y = torch.randn(m, n, c, h, w, requires_grad=True)

            model = AffineFlow(c, h, w, conditioner="ind", num_layers=num_layers, use_norm=use_norm)

            # Test transform
            z, log_joint_dens = model(y)
            log_dets = log_joint_dens - torch.sum(model.base_dist.log_prob(z), dim=(-3, -2, -1))
            dz_dy = torch.autograd.grad(z.sum(), y)[0].log().sum(dim=(-3, -2, -1))

            self.assertIsInstance(z, torch.Tensor,
                                  f"When use_norm = {use_norm}, z should be a torch.Tensor. ")
            self.assertIsInstance(log_dets, torch.Tensor,
                                  f"When use_norm = {use_norm}, log_dets should be a torch.Tensor. ")
            self.assertEqual(z.size(), y.size(),
                             f"When use_norm = {use_norm}, z and y should have equal size. ")
            self.assertEqual(log_dets.size(), dz_dy.size(),
                             f"When use_norm = {use_norm}, log_dets and dz_dy should have equal size. ")
            self.assertEqual(log_dets.size(), (m, n),
                             f"When use_norm = {use_norm}, log_dets should have size {(m, n)}")
            self.assertTrue(torch.isclose(log_dets, dz_dy, atol=1e-5).all(),
                            f"When use_norm = {use_norm}, log_dets and dz_dy should be equal. ")

            # Test inverse transform
            model.eval()
            y = torch.randn(m, n, c, h, w)  # use a fresh y to test the normalisation layers
            with torch.no_grad():
                z, _ = model(y)
            yp = model.inv_transform(z)

            self.assertIsInstance(yp, torch.Tensor,
                                  f"When use_norm = {use_norm}, yp should be a torch.Tensor. ")
            self.assertEqual(yp.size(), y.size(),
                             f"When use_norm = {use_norm}, yp and y should have equal size. ")
            self.assertTrue(torch.isclose(yp, y, atol=1e-5).all(),
                            f"When and use_norm = {use_norm}, yp and y should be equal. ")

    def test_msk(self):
        c, h, w = 2, 3, 4
        n = 7
        num_layers = 3

        for use_norm in [True, False]:
            y = torch.randn(n, c, h, w, requires_grad=True)

            model = AffineFlow(c, h, w, conditioner="msk", num_layers=num_layers, use_norm=use_norm).train()

            # Test transform
            def f(a):
                return model(a)[0]

            jacob = jacobian(f, y)

            jacobs = []
            for b in range(n):
                jacobs.append(jacob[b, :, :, :, b, :, :, :].reshape(c * h * w, -1))
            jacobs = torch.stack(jacobs, dim=0)
            dz_dy = torch.logdet(jacobs)

            z, log_joint_dens = model(y)
            log_dets = log_joint_dens - torch.sum(model.base_dist.log_prob(z), dim=(-3, -2, -1))

            self.assertIsInstance(z, torch.Tensor,
                                  f"When use_norm = {use_norm}, z should be a torch.Tensor. ")
            self.assertIsInstance(log_dets, torch.Tensor,
                                  f"When use_norm = {use_norm}, log_dets should be a torch.Tensor. ")
            self.assertEqual(z.size(), y.size(),
                             f"When use_norm = {use_norm}, z and y should have equal size. ")
            self.assertEqual(log_dets.size(), dz_dy.size(),
                             f"When use_norm = {use_norm}, log_dets and dz_dy should have equal size. ")
            self.assertEqual(log_dets.size(), (n,),
                             f"When use_norm = {use_norm}, log_dets should have size {(n,)}")
            self.assertTrue(torch.isclose(log_dets, dz_dy, atol=1e-5).all(),
                            f"When use_norm = {use_norm}, log_dets and dz_dy should be equal. ")
