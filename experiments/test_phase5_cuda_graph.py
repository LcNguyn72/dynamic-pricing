"""Semantic check for CUDA Graph optimizer and dropout state across resume."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np


class CudaGraphResumeTests(unittest.TestCase):
    def test_interruption_preserves_graph_dropout_and_partial_batch_updates(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA PyTorch runtime unavailable")
        from experiments import phase5_cuda_graph as graph
        from experiments.phase5_training import TENSOR_NAMES

        device = torch.device("cuda")
        settings = {"learning_rate": 1e-3, "best_epoch": 2, "batch_size": 64}
        config = {"history_dim": 9, "current_dim": 8}
        identity = {"fixture": "cuda-resume"}
        rng = np.random.default_rng(17)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name, data in zip(TENSOR_NAMES, (
                rng.normal(size=(66, 12, 9)).astype(np.float32),
                rng.normal(size=(66, 8)).astype(np.float32),
                np.zeros(66, dtype=np.int64),
                rng.normal(size=66).astype(np.float32),
            ), strict=True):
                np.save(root / f"store_{name}.npy", data)
            shards = {"store": {"n": 66}}

            def start(output):
                output.mkdir(exist_ok=True)
                return graph.fit_graph("song_transformer", 42, output, settings, device,
                                       config, root, shards, identity)

            step = graph.GraphStep.step

            def checked_step(instance, batch):
                self.assertFalse(torch.backends.cuda.mem_efficient_sdp_enabled())
                self.assertTrue(torch.backends.cuda.math_sdp_enabled())
                return step(instance, batch)

            with patch.object(graph, "PROGRESS_BATCH", 1):
                with patch.object(graph.GraphStep, "step", checked_step):
                    expected, _ = start(root / "continuous")
                for interrupt_before in (2, 3):
                    with self.subTest(interrupt_before=interrupt_before):
                        step = graph.GraphStep.step
                        counter = 0

                        def interrupted(instance, batch):
                            nonlocal counter
                            counter += 1
                            if counter == interrupt_before:
                                raise RuntimeError("simulated interruption")
                            return step(instance, batch)

                        output = root / f"resumed_{interrupt_before}"
                        with patch.object(graph.GraphStep, "step", interrupted):
                            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                                start(output)
                        actual, fit = start(output)
                        self.assertEqual(fit["optimizer_steps"], 4)
                        for key, value in expected.state_dict().items():
                            self.assertTrue(torch.equal(actual.state_dict()[key], value), msg=key)
