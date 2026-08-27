import unittest

import torch

from scripts.train_lora_pooling import batch_examples


class BatchExamplesTest(unittest.TestCase):
    def make_example(self, length):
        return {
            "inputs_embeds": torch.ones(1, length, 3),
            "per_layer_inputs": torch.ones(1, length, 2, 4),
            "attention_mask": torch.ones(1, length, dtype=torch.long),
            "mm_token_type_ids": torch.ones(1, length, dtype=torch.long),
            "labels": torch.arange(length).reshape(1, length),
        }

    def test_right_pads_examples(self):
        batch = batch_examples([self.make_example(2), self.make_example(4)], "right")

        self.assertEqual(batch["inputs_embeds"].shape, (2, 4, 3))
        self.assertEqual(batch["per_layer_inputs"].shape, (2, 4, 2, 4))
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 0, 0], [1, 1, 1, 1]])
        self.assertEqual(batch["labels"].tolist(), [[0, 1, -100, -100], [0, 1, 2, 3]])

    def test_left_pads_examples(self):
        batch = batch_examples([self.make_example(2), self.make_example(3)], "left")

        self.assertEqual(batch["attention_mask"].tolist(), [[0, 1, 1], [1, 1, 1]])
        self.assertEqual(batch["labels"].tolist(), [[-100, 0, 1], [0, 1, 2]])


if __name__ == "__main__":
    unittest.main()
