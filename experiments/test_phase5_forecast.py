import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


class ArimaRefitTests(unittest.TestCase):
    def test_sparse_fallback_advances_only_after_observing_that_week(self):
        from experiments.phase5_forecast import arima_series_predictions

        fit = np.array([8.0, np.nan, 12.0])
        test = np.array([57.0, np.nan, 900.0, 2.0])
        predicted, fallback = arima_series_predictions(fit, test)
        np.testing.assert_array_equal(predicted, [12, 57, 57, 900])
        np.testing.assert_array_equal(fallback, [True, True, True, True])

    def test_failed_fit_and_nonfinite_predictions_use_same_rolling_fallback(self):
        from experiments.phase5_forecast import arima_series_predictions

        with patch("statsmodels.tsa.statespace.sarimax.SARIMAX") as sarimax:
            fitted = Mock(mle_retvals={"converged": False})
            sarimax.return_value.fit.return_value = fitted
            train = np.arange(1, 13, dtype=float)
            test = np.array([40.0, 800.0])
            predicted, fallback = arima_series_predictions(train, test)
            np.testing.assert_array_equal(predicted, [12, 40])
            np.testing.assert_array_equal(fallback, [True, True])
            fitted.mle_retvals["converged"] = True
            fitted.extend.return_value.get_prediction.return_value.predicted_mean = [4.0, np.nan]
            predicted, fallback = arima_series_predictions(train, test)
            np.testing.assert_array_equal(predicted, [4, 40])
            np.testing.assert_array_equal(fallback, [False, True])

    def test_batch_kalman_extension_matches_walk_forward_without_current_week(self):
        from experiments.phase5_forecast import arima_series_predictions
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        train = np.random.default_rng(62).normal(20, 2, 150)
        test = np.array([100.0, np.nan, 10.0, 21.0, 19.0])
        fitted = SARIMAX(train, order=(1, 0, 1), trend="c",
                         enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        self.assertTrue(fitted.mle_retvals["converged"])
        expected = []
        for observation in test:
            expected.append(max(0, float(fitted.forecast(1)[0])))
            fitted = fitted.append([observation], refit=False)
        actual, used = arima_series_predictions(train, test)
        self.assertFalse(used.any())
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)


class NeuralResumeTests(unittest.TestCase):
    def test_resume_restores_optimizer_and_rng_after_saved_batch(self):
        import torch
        from experiments import phase5_training as training

        original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, original_threads)
        config = {"architecture": "lstm", "lookback": 12, "history_dim": 1,
                  "auxiliary_dim": 1, "hidden_dim": 4}
        settings = {"learning_rate": .001, "best_epoch": 2, "batch_size": 4}
        identity = {"fixture": "resume"}
        device = torch.device("cpu")
        rng = np.random.default_rng(43)
        with TemporaryDirectory() as directory:
            cache = Path(directory)
            shards = {"x": {"n": 8}}
            fixture = (rng.normal(size=(8, 12, 1)).astype("float32"),
                       rng.normal(size=(8, 1)).astype("float32"),
                       np.zeros(8, dtype="int64"), rng.random(8, dtype=np.float32))
            for name, values in zip(training.TENSOR_NAMES, fixture, strict=True):
                np.save(cache / f"x_{name}.npy", values)
            full, resumed = cache / "full", cache / "resumed"
            full.mkdir()
            resumed.mkdir()

            def fit(output):
                return training._fit_neural("lstm", 42, output, settings, device,
                                            config, cache, shards, identity)

            original_step = training._train_step
            step_count = 0

            def interrupt(network, optimizer, batch, *, song):
                nonlocal step_count
                step_count += 1
                if step_count == 2:
                    raise RuntimeError("simulated interruption")
                return original_step(network, optimizer, batch, song=song)

            with patch.object(training, "PROGRESS_BATCH", 1):
                expected, _ = fit(full)
                with patch.object(training, "_train_step", side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        fit(resumed)
                actual, fit_details = fit(resumed)
            self.assertEqual(fit_details["optimizer_steps"], 4)
            for key, tensor in expected.state_dict().items():
                self.assertTrue(torch.equal(tensor, actual.state_dict()[key]), msg=key)
