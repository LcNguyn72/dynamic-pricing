import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from final_system.bundle import ModelBundle, load_bundle, write_bundle
from final_system.engine import InferenceError, forecast, recommend_price
from final_system.schemas import ForecastRequest, PriceRequest


def fixture():
    return {
        "uci_skus": ["A"],
        "m5": {
            "P@CA_1": {
                "ar1": {"qty_mean": 10.0, "qty_sd": 2.0, "params": [0.5, 0.2, 1.0]},
                "arimax": {"qty_mean": 12.0, "qty_sd": 2.0, "price_min": 1.0,
                           "price_max": 3.0, "price_mean": 2.0, "price_sd": 1.0,
                           "params": [0.0, -2.0, 0.5, 1.0]},
                "fixed_reference_price": 1.8,
            },
            "Q@CA_1": {"ar1": None, "arimax": None, "fixed_reference_price": None},
        },
    }


def bundle():
    config = {"id": "fixture", "pricing": {"grid_points": 41},
              "datasets": {"m5": {"validation_end": "2015-07-27"}}}
    return ModelBundle({"version": "v2-fixture", "config": config}, fixture())


class EngineTests(unittest.TestCase):
    def test_arimax_grid_uses_past_lag_and_preserves_reference(self):
        request = PriceRequest(sku="P", store_id="CA_1", week="2015-08-03",
                               history=[{"week": "2015-07-27", "qty": 12, "price": 2}])
        price = recommend_price(bundle(), request)
        self.assertEqual(len(price["candidates"]), 41)
        self.assertAlmostEqual(price["grid_result"]["price"], 2.5)
        self.assertAlmostEqual(price["grid_result"]["predicted_qty"], 10.0)
        self.assertAlmostEqual(price["grid_result"]["expected_revenue"], 25.0)
        self.assertEqual(price["fixed_reference"]["price"], 1.8)

    def test_price_grid_adds_a_distinct_in_support_current_price(self):
        request = PriceRequest(sku="P", store_id="CA_1", week="2015-08-03", current_price=2.011,
                               history=[{"week": "2015-07-27", "qty": 12, "price": 2}])
        price = recommend_price(bundle(), request)
        self.assertEqual(len(price["candidates"]), 42)
        self.assertIn(2.011, {candidate["price"] for candidate in price["candidates"]})

    def test_series_without_reference_does_not_promise_one_after_validation(self):
        value = bundle()
        value.data["m5"]["P@CA_1"]["fixed_reference_price"] = None
        request = PriceRequest(sku="P", store_id="CA_1", week="2015-07-27",
                               history=[{"week": "2015-07-20", "qty": 12, "price": 2}])
        self.assertEqual(recommend_price(value, request)["fixed_reference_status"], "no_validated_reference")

    def test_forecast_ar1_and_fallback(self):
        history = [{"week": "2015-07-27", "qty": 12, "price": 2}]
        request = dict(dataset="m5", store_id="CA_1", week="2015-08-03", history=history)
        point = forecast(bundle(), ForecastRequest(**request, sku="P"))
        self.assertAlmostEqual(point["predicted_qty"], 11.4)
        point = forecast(bundle(), ForecastRequest(**request, sku="Q"))
        self.assertEqual(point["method"], "last_observed")
        self.assertEqual(point["predicted_qty"], 12)
        self.assertEqual(point["source_week"], date(2015, 7, 27))

    def test_ar1_projects_across_unobserved_weeks_without_zero_filling(self):
        request = ForecastRequest(dataset="m5", store_id="CA_1", sku="P", week="2015-08-03",
                                  history=[{"week": "2015-07-20", "qty": 12}])
        point = forecast(bundle(), request)
        self.assertEqual(point["method"], "arima_1_0_0")
        self.assertAlmostEqual(point["predicted_qty"], 11.28)

    def test_missing_previous_week_and_current_target_are_rejected_for_pricing(self):
        for week, code in (("2015-07-20", "missing_previous_week"), ("2015-08-03", "future_history")):
            request = PriceRequest(sku="P", store_id="CA_1", week="2015-08-03",
                                   history=[{"week": week, "qty": 10, "price": 2}])
            with self.assertRaises(InferenceError) as error:
                recommend_price(bundle(), request)
            self.assertEqual(error.exception.code, code)

    def test_fit_availability_and_out_of_support_price_are_explicit(self):
        history = [{"week": "2015-07-27", "qty": 12, "price": 2}]
        request = PriceRequest(sku="Q", store_id="CA_1", week="2015-08-03", history=history)
        with self.assertRaises(InferenceError) as error:
            recommend_price(bundle(), request)
        self.assertEqual(error.exception.code, "price_model_unavailable")
        request = ForecastRequest(dataset="m5", sku="P", store_id="CA_1", week="2015-08-03",
                                  candidate_price=5, history=history)
        with self.assertRaises(InferenceError) as error:
            forecast(bundle(), request)
        self.assertEqual(error.exception.code, "price_outside_train_range")

    def test_uci_latest_week_and_zero_sales(self):
        request = ForecastRequest(dataset="uci", sku="A", week="2011-10-17",
                                  history=[{"week": "2011-10-10", "qty": 0}])
        self.assertEqual(forecast(bundle(), request)["predicted_qty"], 0)

    def test_bundle_rejects_a_corrupt_payload(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            result = write_bundle(path, fixture(), bundle().manifest["config"], {"fit_scope": "train_only"})
            self.assertEqual(load_bundle(path).data["m5"], fixture()["m5"])
            (path / result["file"]).write_bytes(b"broken")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_bundle(path)

    def test_bundle_rejects_config_changed_without_version_change(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            manifest = write_bundle(path, fixture(), bundle().manifest["config"], {"fit_scope": "train_only"})
            manifest["config"]["pricing"]["grid_points"] = 3
            (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "version"):
                load_bundle(path)
