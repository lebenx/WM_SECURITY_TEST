"""Regression checks for camera/time indexing and scalar patch normalization."""
import unittest
import torch
from worldagen_forecast_layout import primary_forecasts, restore_patch_statistics


class ForecastLayoutTests(unittest.TestCase):
    def test_primary_time_order_with_distinct_camera_sentinels(self):
        world = torch.zeros(6, 2, 196, 768)
        for b in range(3):
            for t in range(2):
                for view in range(2):
                    world[b * 2 + t, view] = 100 * b + 10 * t + view
        primary = primary_forecasts(world, 3)
        self.assertEqual(primary[:, :, 0, 0].tolist(), [[0, 10], [100, 110], [200, 210]])
        # The historical [:, 0] after reshape instead returns [0, 1]: two views.
        self.assertNotEqual(primary.tolist(), world.reshape(3, 2, 2, 196, 768)[:, 0].tolist())

    def test_scalar_patch_roundtrip_with_different_channel_means(self):
        torch.manual_seed(7)
        image = torch.randn(3, 224, 224) + torch.tensor([0., 3., 7.])[:, None, None]
        grid = image.reshape(3, 14, 16, 14, 16).permute(1, 3, 2, 4, 0)
        patches = grid.reshape(196, 768)
        normalized = (patches - patches.mean(-1, keepdim=True)) / (patches.var(-1, keepdim=True) + 1e-6).sqrt()
        restored = restore_patch_statistics(normalized, image)
        torch.testing.assert_close(restored, grid, atol=2e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
