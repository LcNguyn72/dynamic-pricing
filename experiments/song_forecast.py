"""Paper-aligned Transformer demand forecaster with explicit open assumptions.

Song (2026) specifies 12 historical weeks, a 4-layer/8-head/128-wide encoder,
Adam at 1e-3, MSE demand loss, and an automatic price derivative. This module
implements those choices on train-only per-SKU z scores. The article does not
publish its code, exact SKU eligibility, promotion construction or an
unambiguous UCI Table-1 RMSE normalization; reports retain multiple scales.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


LOOKBACK = 12
HISTORY_FEATURES = (
    "quantity_z", "price_z", "month_sin", "month_cos", "week_sin", "week_cos",
    "event_days", "snap_days", "promotion_flag",
)
CURRENT_FEATURES = (
    "candidate_price_z", "month_sin", "month_cos", "week_sin", "week_cos",
    "event_days", "snap_days", "promotion_flag",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_float(rows: pd.DataFrame, name: str) -> np.ndarray:
    if name not in rows.columns:
        return np.zeros(len(rows), dtype=np.float32)
    return pd.to_numeric(rows[name], errors="raise").fillna(0).to_numpy(dtype=np.float32)


def _scalers(rows: pd.DataFrame, scalers: Mapping[str, Mapping[str, float]]) -> tuple[np.ndarray, ...]:
    try:
        values = [scalers[str(sku)] for sku in rows["sku"]]
    except KeyError as exc:
        raise ValueError(f"No train scaler for {exc}") from exc
    arrays = tuple(np.asarray([v[key] for v in values], dtype=np.float32)
                   for key in ("qty_mean", "qty_std", "price_mean", "price_std"))
    if np.any(arrays[1] <= 0) or np.any(arrays[3] <= 0):
        raise ValueError("Per-SKU standard deviations must be positive")
    return arrays


def _season(week: pd.Series) -> tuple[np.ndarray, ...]:
    dates = pd.to_datetime(week)
    month = dates.dt.month.to_numpy(dtype=np.float32)
    week_no = dates.dt.isocalendar().week.to_numpy(dtype=np.float32)
    return (
        np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
        np.sin(2 * np.pi * week_no / 52.18), np.cos(2 * np.pi * week_no / 52.18),
    )


def feature_arrays(rows: pd.DataFrame, scalers: Mapping[str, Mapping[str, float]],
                   candidate_prices: np.ndarray | None = None,
                   lookback: int = LOOKBACK) -> tuple[np.ndarray, np.ndarray]:
    """Build history with only past demand/price and known calendar context."""
    if rows.empty:
        raise ValueError("No forecast windows")
    if not 1 <= lookback <= LOOKBACK:
        raise ValueError(f"Lookback must be between 1 and {LOOKBACK} weeks")
    q_mean, q_std, p_mean, p_std = _scalers(rows, scalers)
    dates = pd.to_datetime(rows["week"])
    history = np.empty((len(rows), lookback, len(HISTORY_FEATURES)), dtype=np.float32)
    for position, lag in enumerate(range(lookback, 0, -1)):
        q = pd.to_numeric(rows[f"qty_lag_{lag}"], errors="raise").to_numpy(dtype=np.float32)
        p = pd.to_numeric(rows[f"price_lag_{lag}"], errors="raise").to_numpy(dtype=np.float32)
        if not np.isfinite(q).all() or not np.isfinite(p).all() or np.any(p <= 0):
            raise ValueError(f"Invalid historical quantity or price at lag {lag}")
        history[:, position, 0] = (q - q_mean) / q_std
        history[:, position, 1] = (p - p_mean) / p_std
        history[:, position, 2:6] = np.column_stack(_season(dates - pd.Timedelta(weeks=lag)))
        history[:, position, 6] = _as_float(rows, f"event_days_lag_{lag}") / 7.0
        history[:, position, 7] = _as_float(rows, f"snap_days_lag_{lag}") / 7.0
        history[:, position, 8] = _as_float(rows, f"promotion_flag_lag_{lag}")
    prices = (pd.to_numeric(rows["price"], errors="raise").to_numpy(dtype=np.float32)
              if candidate_prices is None else np.asarray(candidate_prices, dtype=np.float32))
    if prices.shape != (len(rows),) or not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("Candidate prices must be one positive finite value per window")
    current = np.column_stack((
        (prices - p_mean) / p_std,
        *_season(dates),
        _as_float(rows, "event_days") / 7.0,
        _as_float(rows, "snap_days") / 7.0,
        _as_float(rows, "promotion_flag"),
    )).astype(np.float32)
    if not np.isfinite(history).all() or not np.isfinite(current).all():
        raise ValueError("Non-finite model features")
    return history, current


class SongTransformer(nn.Module):
    def __init__(self, history_dim: int = len(HISTORY_FEATURES),
                 current_dim: int = len(CURRENT_FEATURES), *, lookback: int = LOOKBACK):
        super().__init__()
        if not 1 <= lookback <= LOOKBACK:
            raise ValueError(f"Lookback must be between 1 and {LOOKBACK} weeks")
        self.lookback = lookback
        self.embedding = nn.Linear(history_dim, 128)
        self.position = nn.Parameter(torch.zeros(1, lookback, 128))
        layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, dim_feedforward=512, dropout=0.1,
            activation="relu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=4)
        self.norm = nn.LayerNorm(128)
        self.head = nn.Sequential(nn.Linear(128 + current_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.apply(self._initialize)
        nn.init.normal_(self.position, std=0.02)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, history: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.embedding(history) + self.position)[:, -1, :]
        return self.head(torch.cat((self.norm(encoded), current), dim=1)).squeeze(-1)


class SongDemandForecaster:
    """Adapter for continuous-price prediction, gradient and online feedback."""

    def __init__(self, model: SongTransformer, scalers: Mapping[str, Mapping[str, float]]):
        self.model = model.cpu().eval()
        self.scalers = dict(scalers)

    def _tensor(self, row: Mapping, price: float):
        frame = pd.DataFrame([{**dict(row), "sku": str(row["sku"])}])
        history, current = feature_arrays(frame, self.scalers, np.asarray([price], dtype=np.float32),
                                          lookback=self.model.lookback)
        return torch.from_numpy(history), torch.tensor(current, requires_grad=True)

    def predict_q(self, row: Mapping, price: float) -> float:
        sku = str(row["sku"])
        history, current = self._tensor(row, price)
        with torch.no_grad():
            z = float(self.model(history, current)[0])
        scaler = self.scalers[sku]
        return max(0.0, z * float(scaler["qty_std"]) + float(scaler["qty_mean"]))

    def price_gradient(self, row: Mapping, price: float) -> float:
        sku = str(row["sku"])
        history, current = self._tensor(row, price)
        z = self.model(history, current)[0]
        scaler = self.scalers[sku]
        if float(z.detach()) * float(scaler["qty_std"]) + float(scaler["qty_mean"]) <= 0:
            return 0.0
        dz_dp_z = torch.autograd.grad(z, current)[0][0, 0]
        return float(dz_dp_z * float(scaler["qty_std"]) / float(scaler["price_std"]))

    def fine_tune(self, samples: Sequence[tuple[Mapping, float, float]], epochs: int = 1,
                  lr: float = 1e-4, batch_size: int = 32) -> None:
        if not samples:
            return
        rows = pd.DataFrame([{**dict(row), "sku": str(row["sku"])} for row, _, _ in samples])
        prices = np.asarray([price for _, price, _ in samples], dtype=np.float32)
        history, current = feature_arrays(rows, self.scalers, prices, lookback=self.model.lookback)
        q_mean, q_std, _, _ = _scalers(rows, self.scalers)
        target = (np.asarray([max(0.0, qty) for _, _, qty in samples], dtype=np.float32) - q_mean) / q_std
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(history), torch.from_numpy(current),
                                           torch.from_numpy(target)),
            batch_size=batch_size, shuffle=True,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for _ in range(epochs):
            for bh, bc, by in loader:
                optimizer.zero_grad(set_to_none=True)
                F.mse_loss(self.model(bh, bc), by).backward()
                optimizer.step()
        self.model.eval()


def _metrics(rows: pd.DataFrame, predictions_z: np.ndarray,
             scalers: Mapping[str, Mapping[str, float]]) -> dict:
    q_mean, q_std, _, _ = _scalers(rows, scalers)
    actual = rows["qty"].to_numpy(dtype=np.float64)
    predicted = np.maximum(0.0, predictions_z.astype(np.float64) * q_std + q_mean)
    error = predicted - actual
    positive = actual > 0
    return {
        "n": len(rows),
        "rmse_units_per_week": float(np.sqrt(np.mean(error ** 2))),
        "mae_units_per_week": float(np.mean(np.abs(error))),
        "mape_pct_positive_targets": float(100 * np.mean(np.abs(error[positive] / actual[positive]))) if positive.any() else None,
        "rmse_train_mean_relative": float(np.sqrt(np.mean((error / np.maximum(q_mean, 1.0)) ** 2))),
        "rmse_train_z": float(np.sqrt(np.mean((error / q_std) ** 2))),
    }


def run(prepared_root: str | Path, output_dir: str | Path, *, seed: int = 42,
        epochs: int = 50, batch_size: int = 64, max_skus: int | None = None,
        refit_train_validation: bool = False) -> dict:
    prepared_root, output_dir = Path(prepared_root), Path(output_dir)
    manifest_path, windows_path = prepared_root / "manifest.json", prepared_root / "windows.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    windows = pd.read_csv(windows_path, dtype={"sku": str}, parse_dates=["week"])
    scalers = manifest["scalers"]
    if max_skus is not None:
        if max_skus < 1:
            raise ValueError("max_skus must be positive")
        train_rank = (windows.loc[windows["split"].eq("train") & windows["observed_sales"],
                                  ["sku", "qty"]].groupby("sku")["qty"].sum().sort_values(ascending=False))
        chosen = set(train_rank.head(max_skus).index)
        windows = windows.loc[windows["sku"].isin(chosen)].copy()
        scalers = {sku: data for sku, data in scalers.items() if sku in chosen}
    observed = windows["observed_sales"].astype(bool)
    groups = {name: windows.loc[windows["split"].eq(name) & observed].reset_index(drop=True)
              for name in ("train", "validation", "test")}
    if any(group.empty for group in groups.values()):
        raise ValueError("Train, validation and test each need observed SKU-weeks")
    arrays = {name: feature_arrays(group, scalers) for name, group in groups.items()}
    targets = {name: (group["qty"].to_numpy(dtype=np.float32) - _scalers(group, scalers)[0]) /
               _scalers(group, scalers)[1] for name, group in groups.items()}
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SongTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            *(torch.from_numpy(a) for a in arrays["train"]), torch.from_numpy(targets["train"])),
        batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed),
    )
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    history_rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_sum = 0.0
        for bh, bc, by in train_loader:
            bh, bc, by = bh.to(device), bc.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(bh, bc), by)
            loss.backward()
            optimizer.step()
            train_sum += float(loss.detach()) * len(by)
        train_mse = train_sum / len(groups["train"])
        model.eval()
        with torch.no_grad():
            vh, vc = arrays["validation"]
            prediction = model(torch.from_numpy(vh).to(device), torch.from_numpy(vc).to(device)).cpu().numpy()
        val_mse = float(np.mean((prediction - targets["validation"]) ** 2))
        history_rows.append({"epoch": epoch, "train_mse_z": train_mse, "validation_mse_z": val_mse})
        print(f"[Song Transformer] seed={seed} epoch={epoch} train_mse={train_mse:.4f} val_mse={val_mse:.4f}", flush=True)
        if val_mse < best_loss:
            best_loss, best_epoch, best_state = val_mse, epoch, copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= 8:
            break
    model.load_state_dict(best_state)
    model.eval()
    test_model = model
    if refit_train_validation:
        torch.manual_seed(seed)
        test_model = SongTransformer().to(device)
        combined_history = np.concatenate((arrays["train"][0], arrays["validation"][0]))
        combined_current = np.concatenate((arrays["train"][1], arrays["validation"][1]))
        combined_target = np.concatenate((targets["train"], targets["validation"]))
        fit_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(combined_history), torch.from_numpy(combined_current),
                torch.from_numpy(combined_target)),
            batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed),
        )
        optimizer = torch.optim.Adam(test_model.parameters(), lr=1e-3)
        for epoch in range(best_epoch):
            test_model.train()
            for bh, bc, by in fit_loader:
                bh, bc, by = bh.to(device), bc.to(device), by.to(device)
                optimizer.zero_grad(set_to_none=True)
                F.mse_loss(test_model(bh, bc), by).backward()
                optimizer.step()
            print(f"[Song refit] seed={seed} epoch={epoch + 1}/{best_epoch}", flush=True)
        test_model.eval()
    scores = {}
    predictions = []
    for name in ("validation", "test"):
        with torch.no_grad():
            h, c = arrays[name]
            predictor = model if name == "validation" else test_model
            z = predictor(torch.from_numpy(h).to(device), torch.from_numpy(c).to(device)).cpu().numpy()
        scores[name] = _metrics(groups[name], z, scalers)
        frame = groups[name][["sku", "week", "qty", "price"]].copy()
        q_mean, q_std, _, _ = _scalers(groups[name], scalers)
        frame["predicted_qty"] = np.maximum(0.0, z * q_std + q_mean)
        frame["split"] = name
        predictions.append(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "scalers": scalers,
                "history_features": HISTORY_FEATURES, "current_features": CURRENT_FEATURES},
               output_dir / f"song_transformer_seed{seed}.pt")
    if refit_train_validation:
        torch.save({"state_dict": test_model.cpu().state_dict(), "scalers": scalers,
                    "history_features": HISTORY_FEATURES, "current_features": CURRENT_FEATURES},
                   output_dir / f"song_transformer_refit_seed{seed}.pt")
    pd.DataFrame(history_rows).to_csv(output_dir / f"training_seed{seed}.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output_dir / f"predictions_seed{seed}.csv", index=False)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": manifest["dataset"], "source_manifest": str((prepared_root / "manifest.json").resolve()),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_windows_sha256": _sha256(windows_path),
        "seed": seed, "epochs_max": epochs, "best_epoch": best_epoch,
        "scope": "all SKUs in prepared panel" if max_skus is None else f"SMOKE: top {max_skus} SKUs by training-period units",
        "test_model_fit_scope": "train+validation" if refit_train_validation else "train",
        "pricing_checkpoint_fit_scope": "train",
        "pricing_checkpoint": f"song_transformer_seed{seed}.pt",
        "n_skus": int(windows["sku"].nunique()), "n_train": len(groups["train"]),
        "n_validation": len(groups["validation"]), "n_test": len(groups["test"]),
        "device": str(device), "architecture": "12-week Transformer 4 layers, 8 heads, d=128",
        "train_target": "per-SKU train-only z-score quantity", "train_loss": "MSE on z-score quantity",
        "optimizer": "Adam 1e-3", "promotion_policy": "Only observed promotion_flag is used; missing flags are zeros and not claimed observed",
        "metric_warning": "The paper conflicts on whether UCI Table 1 RMSE is raw or mean-normalized; both and z-score RMSE are reported separately",
        "scores": scores,
    }
    (output_dir / f"run_seed{seed}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path,
                        default=Path(__file__).resolve().parent / "artifacts" / "song_online_retail" / "paper_70_15_15")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "artifacts" / "song_online_retail" / "song_forecast")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-skus", type=int, help="Explicit small smoke test; omitted means the complete eligible cohort")
    parser.add_argument("--refit-train-validation", action="store_true",
                        help="Refit a separate test model on train+validation after selecting epochs; keep train-only pricing checkpoint")
    args = parser.parse_args()
    result = run(args.prepared_root, args.output_dir, seed=args.seed,
                 epochs=args.epochs, batch_size=args.batch_size, max_skus=args.max_skus,
                 refit_train_validation=args.refit_train_validation)
    print(json.dumps(result["scores"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
