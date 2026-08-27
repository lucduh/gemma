import unittest

import torch

from mllm.gemma4_manual import (
    compact_image_placeholders,
    spatial_average_pool,
    spatial_select,
)


class SpatialAveragePoolTest(unittest.TestCase):
    def test_pools_rectangular_grid(self):
        tokens = torch.arange(28 * 38 * 2, dtype=torch.float32).reshape(28 * 38, 2)

        pooled = spatial_average_pool(tokens, (28, 38), (14, 19))

        self.assertEqual(pooled.shape, (14 * 19, 2))

    def test_pools_odd_grid(self):
        tokens = torch.arange(33 * 33, dtype=torch.float32).reshape(33 * 33, 1)

        pooled = spatial_average_pool(tokens, (33, 33), (16, 16))

        self.assertEqual(pooled.shape, (16 * 16, 1))

    def test_rejects_incorrect_source_grid(self):
        with self.assertRaisesRegex(ValueError, "does not match source grid"):
            spatial_average_pool(torch.zeros(10, 4), (2, 4), (1, 2))


class SpatialSelectTest(unittest.TestCase):
    def test_selects_region_centers(self):
        tokens = torch.arange(4 * 6, dtype=torch.float32).reshape(4 * 6, 1)

        selected = spatial_select(tokens, (4, 6), (2, 3))

        self.assertEqual(selected[:, 0].tolist(), [7, 9, 11, 19, 21, 23])

    def test_preserves_tokens_for_identity_selection(self):
        tokens = torch.arange(3 * 5 * 2, dtype=torch.float32).reshape(3 * 5, 2)

        selected = spatial_select(tokens, (3, 5), (3, 5))

        self.assertTrue(torch.equal(selected, tokens))

    def test_rejects_larger_target_grid(self):
        with self.assertRaisesRegex(ValueError, "cannot be larger"):
            spatial_select(torch.zeros(8, 4), (2, 4), (3, 4))


class CompactImagePlaceholdersTest(unittest.TestCase):
    def test_compacts_all_sequence_tensors(self):
        inputs = {
            "input_ids": torch.tensor([[1, 9, 9, 9, 9, 2]]),
            "attention_mask": torch.ones(1, 6, dtype=torch.long),
            "mm_token_type_ids": torch.tensor([[0, 1, 1, 1, 1, 0]]),
            "pixel_values": torch.zeros(1, 10, 3),
        }

        compacted = compact_image_placeholders(inputs, image_token_id=9, target_count=2)

        self.assertEqual(compacted["input_ids"].tolist(), [[1, 9, 9, 2]])
        self.assertEqual(compacted["attention_mask"].shape, (1, 4))
        self.assertEqual(compacted["mm_token_type_ids"].tolist(), [[0, 1, 1, 0]])
        self.assertEqual(compacted["pixel_values"].shape, (1, 10, 3))

    def test_rejects_multiple_image_blocks(self):
        inputs = {"input_ids": torch.tensor([[1, 9, 2, 9, 3]])}

        with self.assertRaisesRegex(ValueError, "one contiguous"):
            compact_image_placeholders(inputs, image_token_id=9, target_count=1)


if __name__ == "__main__":
    unittest.main()
