import tempfile
import unittest
from pathlib import Path

import torch

from mllm.local_resampler import LocalResampler


class LocalResamplerTest(unittest.TestCase):
    def test_returns_target_grid(self):
        resampler = LocalResampler(hidden_size=8)
        tokens = torch.randn(6 * 10, 8)

        output = resampler(tokens, (6, 10), (3, 5))

        self.assertEqual(output.shape, (15, 8))

    def test_initialization_is_close_to_spatial_selection(self):
        resampler = LocalResampler(hidden_size=1, position_scale=50.0)
        tokens = torch.arange(4 * 6, dtype=torch.float32).reshape(-1, 1)

        output = resampler(tokens, (4, 6), (2, 3))

        expected = torch.tensor([[7.0], [9.0], [11.0], [19.0], [21.0], [23.0]])
        self.assertTrue(torch.allclose(output, expected, atol=1e-4))

    def test_content_scorer_receives_gradients(self):
        resampler = LocalResampler(hidden_size=4)
        tokens = torch.randn(4 * 4, 4)

        output = resampler(tokens, (4, 4), (2, 2))
        output.square().mean().backward()

        final_layer = resampler.content_score[-1]
        self.assertIsNotNone(final_layer.weight.grad)
        self.assertGreater(final_layer.weight.grad.abs().sum().item(), 0)

    def test_saves_and_loads(self):
        resampler = LocalResampler(hidden_size=4, neighbors=3)
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            resampler.save_pretrained(directory)
            loaded = LocalResampler.from_pretrained(directory, torch.device("cpu"))

        self.assertEqual(loaded.hidden_size, 4)
        self.assertEqual(loaded.neighbors, 3)
        for expected, actual in zip(
            resampler.parameters(), loaded.parameters(), strict=True
        ):
            self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
