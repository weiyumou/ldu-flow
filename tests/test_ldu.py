import torch
import unittest

from project.models.ldu import LDUTransformer, LDUFlow


class LDUTransformerTestCase(unittest.TestCase):

    def test_mlp(self):
        """
        Remember to change the 'conditioner' in SinusoidalTransformer to 'mlp' in 'd_scale'.
        """
        c, h, w = 2, 3, 4
        embed_dim = 6
        n, d = 7, 3
        hid_dims = (36, 36)
        attn_size = 4
        num_fmaps, num_blocks = 16, 4
        num_d_trans = 4

        for conditioner in ["msk", "atn", "cnn"]:
            x = torch.randn(n, d)
            y = torch.randn(n, c, h, w, requires_grad=True) * 4

            model = LDUTransformer(c, h, w, embed_dim,
                                   conditioner=conditioner, num_d_trans=num_d_trans,
                                   in_dim=d, hid_dims=hid_dims,
                                   attn_size=attn_size,
                                   num_fmaps=num_fmaps, num_blocks=num_blocks).train()

            # Test transform
            from torch.autograd.functional import jacobian

            def f(a):
                return model(a, cond_var=x)[0]

            jacob = jacobian(f, y)

            jacobs = []
            for b in range(n):
                jacobs.append(jacob[b, :, :, :, b, :, :, :].reshape(c * h * w, -1))
            jacobs = torch.stack(jacobs, dim=0)
            dz_dy = torch.logdet(jacobs)

            z, log_dets = model(y, cond_var=x)

            self.assertIsInstance(z, torch.Tensor, f"When conditioner = '{conditioner}', z should be a torch.Tensor. ")
            self.assertIsInstance(log_dets, torch.Tensor,
                                  f"When conditioner = '{conditioner}', log_dets should be a torch.Tensor. ")
            self.assertEqual(z.size(), y.size(),
                             f"When conditioner = '{conditioner}', z and y should have equal size. ")
            self.assertEqual(log_dets.size(), dz_dy.size(),
                             f"When conditioner = '{conditioner}', log_dets and dz_dy should have equal size. ")
            self.assertEqual(log_dets.size(), (n,),
                             f"When conditioner = '{conditioner}', log_dets should have size {(n,)}")
            self.assertTrue(torch.allclose(log_dets, dz_dy, rtol=1e-4, atol=1e-6),
                            f"When conditioner = '{conditioner}', log_dets and dz_dy should be equal. ")

            # Test inverse transform
            model.eval()
            yp = model.inv_transform(z, cond_var=x)
            self.assertIsInstance(yp, torch.Tensor,
                                  f"When conditioner = '{conditioner}', yp should be a torch.Tensor. ")
            self.assertEqual(yp.size(), y.size(),
                             f"When conditioner = '{conditioner}', yp and y should have equal size. ")
            self.assertTrue(torch.allclose(yp, y, rtol=1e-4),
                            f"When conditioner = '{conditioner}', yp and y should be equal. ")


class LDUFlowTestCase(unittest.TestCase):

    def test_mlp(self):
        """
        Remember to change the 'conditioner' in SinusoidalTransformer to 'mlp' in 'd_scale'.
        """
        c, h, w = 2, 3, 4
        embed_dim = 6
        n, d = 7, 3
        hid_dims = (16, 16)
        attn_size = 4
        num_fmaps, num_blocks = 16, 2
        num_layers, num_d_trans = 8, 3

        for conditioner in ["msk", "atn", "cnn"]:
            x = torch.randn(n, d)
            y = torch.randn(n, c, h, w, requires_grad=True) * 4

            model = LDUFlow(c, h, w, embed_dim,
                            conditioner=conditioner, num_layers=num_layers, num_d_trans=num_d_trans,
                            in_dim=d, hid_dims=hid_dims,
                            attn_size=attn_size,
                            num_fmaps=num_fmaps, num_blocks=num_blocks).train()

            # Test transform
            from torch.autograd.functional import jacobian

            def f(a):
                return model(a, cond_var=x)[0]

            jacob = jacobian(f, y)

            jacobs = []
            for b in range(n):
                jacobs.append(jacob[b, :, :, :, b, :, :, :].reshape(c * h * w, -1))
            jacobs = torch.stack(jacobs, dim=0)
            dz_dy = torch.logdet(jacobs)

            z, log_joint_dens = model(y, cond_var=x)
            log_dets = log_joint_dens - torch.sum(model.base_dist.log_prob(z), dim=(-3, -2, -1))

            self.assertIsInstance(z, torch.Tensor, f"When conditioner = '{conditioner}', z should be a torch.Tensor. ")
            self.assertIsInstance(log_dets, torch.Tensor,
                                  f"When conditioner = '{conditioner}', log_dets should be a torch.Tensor. ")
            self.assertEqual(z.size(), y.size(),
                             f"When conditioner = '{conditioner}', z and y should have equal size. ")
            self.assertEqual(log_dets.size(), dz_dy.size(),
                             f"When conditioner = '{conditioner}', log_dets and dz_dy should have equal size. ")
            self.assertEqual(log_dets.size(), (n,),
                             f"When conditioner = '{conditioner}', log_dets should have size {(n,)}")
            self.assertTrue(torch.allclose(log_dets, dz_dy, rtol=1e-4, atol=1e-6),
                            f"When conditioner = '{conditioner}', log_dets and dz_dy should be equal. ")

            # Test inverse transform
            model.eval()

            yp = model.inv_transform(z, cond_var=x)

            self.assertIsInstance(yp, torch.Tensor,
                                  f"When conditioner = '{conditioner}', yp should be a torch.Tensor. ")
            self.assertEqual(yp.size(), y.size(),
                             f"When conditioner = '{conditioner}', yp and y should have equal size. ")
            self.assertTrue(torch.allclose(yp, y, rtol=1e-3),
                            f"When conditioner = '{conditioner}', yp and y should be equal. ")
