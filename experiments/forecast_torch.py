"""Optional PyTorch sequence forecasters for the research benchmark.

The public data contain observed prices, not randomized price interventions.  A
gradient from these models is therefore a model sensitivity, not a causal price
elasticity.  The pricing simulator uses it only as an experimental state feature.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

KNOWN_CALENDAR_COVARIATES = ("event_days", "snap_days")


def _scaler_arrays(rows: pd.DataFrame, scalers: Mapping[str, Mapping[str, float]]):
    sku = rows["sku"].astype(str).to_numpy()
    try:
        q_mean = np.array([scalers[s]["qty_mean"] for s in sku], dtype=np.float32)
        q_std = np.array([scalers[s]["qty_std"] for s in sku], dtype=np.float32)
        p_mean = np.array([scalers[s]["price_mean"] for s in sku], dtype=np.float32)
        p_std = np.array([scalers[s]["price_std"] for s in sku], dtype=np.float32)
    except KeyError as exc:
        raise ValueError(f"Missing train-only scaler for SKU {exc}") from exc
    if np.any(q_std <= 0) or np.any(p_std <= 0):
        raise ValueError("Train-only scaler standard deviations must be positive")
    return q_mean, q_std, p_mean, p_std


def feature_arrays(
    rows: pd.DataFrame,
    sku_to_idx: Mapping[str, int],
    scalers: Mapping[str, Mapping[str, float]],
    lookback: int,
    with_price: bool,
    calendar_covariates: Sequence[str] | None = None,
    sku_encoding: str = "onehot",
) -> tuple[np.ndarray, ...]:
    """Build chronological history and current features from raw window columns.

    Recomputing z scores here is intentional: the simulator may replace raw lag
    values with simulated price-demand history.  It must never reuse precomputed
    z-score lag columns from the original observational trajectory.
    """
    if rows.empty:
        raise ValueError("No rows available for sequence features")
    q_mean, q_std, p_mean, p_std = _scaler_arrays(rows, scalers)
    history = np.empty((len(rows), lookback, 2 if with_price else 1), dtype=np.float32)
    for offset, lag in enumerate(range(lookback, 0, -1)):
        values = rows[f"qty_lag_{lag}"].to_numpy(dtype=np.float32, copy=False)
        history[:, offset, 0] = (values - q_mean) / q_std
    if with_price:
        for offset, lag in enumerate(range(lookback, 0, -1)):
            values = rows[f"price_lag_{lag}"].to_numpy(dtype=np.float32, copy=False)
            history[:, offset, 1] = (values - p_mean) / p_std

    months = rows["month"].to_numpy(dtype=np.float32)
    weeks = rows["week_of_year"].to_numpy(dtype=np.float32)
    season = np.column_stack(
        [np.sin(2 * math.pi * months / 12), np.cos(2 * math.pi * months / 12),
         np.sin(2 * math.pi * weeks / 52.18), np.cos(2 * math.pi * weeks / 52.18)]
    ).astype(np.float32)
    try:
        indices = np.array([sku_to_idx[s] for s in rows["sku"].astype(str)], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Unknown SKU in sequence input: {exc}") from exc
    if sku_encoding not in {"onehot", "embedding"}:
        raise ValueError("sku_encoding must be 'onehot' or 'embedding'")
    if calendar_covariates is None:
        calendar_covariates = tuple(c for c in KNOWN_CALENDAR_COVARIATES if c in rows.columns)
    missing_covariates = set(calendar_covariates) - set(rows.columns)
    if missing_covariates:
        raise ValueError(f"Missing known calendar covariates: {sorted(missing_covariates)}")
    # M5 records counts within a complete seven-day week.  Scaling to [0, 1]
    # avoids making these counts dominate the seasonal sine/cosine features.
    calendar_parts = []
    for covariate in calendar_covariates:
        values = rows[covariate].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 7)):
            raise ValueError(f"{covariate} must contain day counts from 0 to 7")
        calendar_parts.append((values / 7.0)[:, None])
    a_parts = [season, *calendar_parts]
    if sku_encoding == "onehot":
        a_parts.append(np.eye(len(sku_to_idx), dtype=np.float32)[indices])
    if with_price:
        price = rows["price"].to_numpy(dtype=np.float32)
        a_parts.append(((price - p_mean) / p_std)[:, None])
    auxiliary = np.concatenate(a_parts, axis=1).astype(np.float32, copy=False)
    if not np.isfinite(history).all() or not np.isfinite(auxiliary).all():
        raise ValueError("Non-finite sequence features; inspect source prices and lags")
    if sku_encoding == "embedding":
        return history, auxiliary, indices
    return history, auxiliary


class SequenceForecaster(nn.Module):
    """One-step conditional demand model with an LSTM or Transformer encoder."""

    def __init__(
        self,
        architecture: str,
        lookback: int,
        history_dim: int,
        auxiliary_dim: int,
        hidden_dim: int = 64,
        transformer_dim: int = 128,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        dropout: float = 0.1,
        sku_count: int = 0,
        sku_embedding_dim: int = 32,
    ):
        super().__init__()
        if architecture not in {"lstm", "transformer"}:
            raise ValueError("architecture must be 'lstm' or 'transformer'")
        self.architecture = architecture
        self.sku_count = int(sku_count)
        self.sku_embedding_dim = int(sku_embedding_dim) if self.sku_count else 0
        if self.sku_count < 0 or (self.sku_count and self.sku_embedding_dim < 1):
            raise ValueError("SKU embedding dimensions must be positive")
        if self.sku_count:
            self.sku_embedding = nn.Embedding(self.sku_count, self.sku_embedding_dim)
        if architecture == "lstm":
            self.encoder = nn.LSTM(history_dim, hidden_dim, num_layers=1, batch_first=True)
            encoded_dim = hidden_dim
        else:
            self.embedding = nn.Linear(history_dim, transformer_dim)
            self.position = nn.Parameter(torch.zeros(1, lookback, transformer_dim))
            nn.init.normal_(self.position, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=transformer_dim,
                nhead=transformer_heads,
                dim_feedforward=transformer_dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=transformer_layers)
            self.norm = nn.LayerNorm(transformer_dim)
            encoded_dim = transformer_dim
        self.head = nn.Sequential(
            nn.Linear(encoded_dim + auxiliary_dim + self.sku_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, history: torch.Tensor, auxiliary: torch.Tensor,
        sku_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.architecture == "lstm":
            _, (hidden, _) = self.encoder(history)
            encoded = hidden[-1]
        else:
            tokens = self.embedding(history) + self.position
            encoded = self.norm(self.encoder(tokens)[:, -1, :])
        features = [encoded, auxiliary]
        if self.sku_count:
            if sku_index is None:
                raise ValueError("This model requires SKU indices for its embedding")
            features.append(self.sku_embedding(sku_index.long()))
        raw = self.head(torch.cat(features, dim=1)).squeeze(1)
        # The target is log1p(qty), which is nonnegative. This transform keeps
        # expm1(output) nonnegative and differentiable for policy gradients.
        return F.softplus(raw)


def _fit_epochs(
    model: SequenceForecaster,
    history: np.ndarray,
    auxiliary: np.ndarray,
    target: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    validation: tuple[np.ndarray, ...] | None = None,
    patience: int | None = None,
    sku_indices: np.ndarray | None = None,
) -> tuple[SequenceForecaster, dict]:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    tensors = [torch.from_numpy(history), torch.from_numpy(auxiliary)]
    if model.sku_count:
        if sku_indices is None:
            raise ValueError("SKU indices are required by this sequence model")
        tensors.append(torch.from_numpy(sku_indices.astype(np.int64, copy=False)))
    tensors.append(torch.from_numpy(target))
    dataset = torch.utils.data.TensorDataset(*tensors)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    losses: list[float] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for batch in loader:
            h, a = batch[:2]
            if model.sku_count:
                sku_batch, y = batch[2], batch[3]
            else:
                sku_batch, y = None, batch[2]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(h, a, sku_batch)
            loss = F.mse_loss(prediction, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(y)
            seen += len(y)
        losses.append(total / max(seen, 1))
        if validation is None:
            continue
        model.eval()
        with torch.no_grad():
            vh, va = validation[:2]
            if model.sku_count:
                vsi, vy = validation[2], validation[3]
            else:
                vsi, vy = None, validation[2]
            pred = _predict_array(model, vh, va, vsi).numpy()
            val_loss = float(np.mean((np.expm1(pred) - np.expm1(vy)) ** 2))
        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if patience is not None and stale >= patience:
                break
    if validation is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch or epochs, "validation_rmse": math.sqrt(best_val) if validation else None,
                   "train_loss_last": losses[-1], "epochs_run": len(losses)}


def _predict_array(
    model: SequenceForecaster,
    history: np.ndarray,
    auxiliary: np.ndarray,
    sku_indices: np.ndarray | None = None,
    batch_size: int = 4096,
) -> torch.Tensor:
    """Predict in bounded batches so full-M5 inference does not copy all rows to Torch at once."""
    pieces = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(history), batch_size):
            end = min(start + batch_size, len(history))
            h = torch.from_numpy(history[start:end])
            a = torch.from_numpy(auxiliary[start:end])
            si = torch.from_numpy(sku_indices[start:end]) if sku_indices is not None else None
            pieces.append(model(h, a, si).cpu())
    return torch.cat(pieces) if pieces else torch.empty(0)


def train_sequence_model(
    windows: pd.DataFrame,
    manifest: Mapping,
    architecture: str,
    with_price: bool,
    seed: int = 42,
    max_epochs: int = 25,
    patience: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> tuple[SequenceForecaster, np.ndarray, np.ndarray, dict, dict, SequenceForecaster]:
    """Tune epochs on validation, then refit using train+validation for test.

    Returns the validation-selected train-only model separately. Pricing uses
    that checkpoint so its forecaster does not see labels beyond the simulator
    and RL policies' training interval.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.get_num_threads() > 4:
        torch.set_num_threads(4)
    lookback = int(manifest.get("lookback_weeks", manifest.get("lookback", 12)))
    scalers = manifest["scalers"]
    calendar_covariates = tuple(c for c in KNOWN_CALENDAR_COVARIATES if c in windows.columns)
    skus = sorted(windows["sku"].astype(str).unique())
    sku_to_idx = {sku: i for i, sku in enumerate(skus)}
    # Dense one-hot features are cheap for the original small cohorts, but scale
    # as rows × SKUs and become impractical on the complete M5 panel.
    sku_encoding = "embedding" if len(skus) > 256 else "onehot"
    observed = windows["observed_sales"].astype(bool)
    train = windows[(windows["split"] == "train") & observed].copy()
    val = windows[(windows["split"] == "validation") & observed].copy()
    test = windows[(windows["split"] == "test") & observed].copy()
    if train.empty or test.empty:
        raise ValueError("Train and test each need observed sales rows")
    tr_features = feature_arrays(
        train, sku_to_idx, scalers, lookback, with_price, calendar_covariates, sku_encoding
    )
    tr_h, tr_a = tr_features[:2]
    tr_sku = tr_features[2] if sku_encoding == "embedding" else None
    if not val.empty:
        va_features = feature_arrays(
            val, sku_to_idx, scalers, lookback, with_price, calendar_covariates, sku_encoding
        )
        va_h, va_a = va_features[:2]
        va_sku = va_features[2] if sku_encoding == "embedding" else None
    else:
        va_h = np.empty((0, tr_h.shape[1], tr_h.shape[2]), dtype=np.float32)
        va_a = np.empty((0, tr_a.shape[1]), dtype=np.float32)
        va_sku = np.empty(0, dtype=np.int64) if sku_encoding == "embedding" else None
    te_features = feature_arrays(
        test, sku_to_idx, scalers, lookback, with_price, calendar_covariates, sku_encoding
    )
    te_h, te_a = te_features[:2]
    te_sku = te_features[2] if sku_encoding == "embedding" else None
    tr_y = np.log1p(train["qty"].to_numpy(dtype=np.float32))
    va_y = np.log1p(val["qty"].to_numpy(dtype=np.float32)) if not val.empty else np.empty(0, dtype=np.float32)
    config = {
        "architecture": architecture,
        "lookback": lookback,
        "history_dim": tr_h.shape[2],
        "auxiliary_dim": tr_a.shape[1],
        "sku_count": len(skus) if sku_encoding == "embedding" else 0,
        "sku_embedding_dim": min(32, max(8, int(round(math.sqrt(len(skus)))))) if sku_encoding == "embedding" else 0,
    }
    model = SequenceForecaster(**config)
    model, training_info = _fit_epochs(
        model, tr_h, tr_a, tr_y, max_epochs, batch_size, lr, seed,
        validation=(va_h, va_a, va_sku, va_y) if sku_encoding == "embedding" and not val.empty
        else ((va_h, va_a, va_y) if not val.empty else None),
        patience=patience if not val.empty else None,
        sku_indices=tr_sku,
    )
    model.eval()
    if not val.empty:
        with torch.no_grad():
            val_pred = np.expm1(_predict_array(model, va_h, va_a, va_sku).numpy())
    else:
        val_pred = np.empty(0, dtype=np.float32)

    # Validation chooses the number of epochs; its labels may then be used for
    # the final fit, while the test period remains untouched. With no external
    # validation split, fixed hyperparameters use the already trained model.
    if val.empty:
        final_model, final_info = model, training_info
    else:
        torch.manual_seed(seed)
        final_model = SequenceForecaster(**config)
        fit_h = np.concatenate((tr_h, va_h), axis=0)
        fit_a = np.concatenate((tr_a, va_a), axis=0)
        fit_y = np.concatenate((tr_y, va_y), axis=0)
        fit_sku = np.concatenate((tr_sku, va_sku), axis=0) if sku_encoding == "embedding" else None
        final_model, final_info = _fit_epochs(
            final_model, fit_h, fit_a, fit_y, int(training_info["best_epoch"]),
            batch_size, lr, seed, sku_indices=fit_sku,
        )
    final_model.eval()
    with torch.no_grad():
        test_pred = np.expm1(_predict_array(final_model, te_h, te_a, te_sku).numpy())
    metadata = {
        "config": config,
        "sku_to_idx": sku_to_idx,
        "scalers": scalers,
        "calendar_covariates": list(calendar_covariates),
        "sku_encoding": sku_encoding,
        "with_price": bool(with_price),
        "seed": seed,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "validation_training": training_info,
        "final_training": final_info,
        "target": "log1p(quantity)",
    }
    return (final_model, np.clip(val_pred, 0, None), np.clip(test_pred, 0, None),
            metadata, {"val": val, "test": test}, model)


def save_sequence_model(model: SequenceForecaster, metadata: dict, output_dir: str | Path, name: str):
    import json

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / f"{name}.pt")
    with (out / f"{name}_meta.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)


class PriceForecaster:
    """Differentiable price-aware Transformer for the controlled simulator."""

    def __init__(self, model: SequenceForecaster, metadata: Mapping):
        if not metadata.get("with_price"):
            raise ValueError("The loaded model has no price input")
        self.model = model.cpu().eval()
        self.metadata = dict(metadata)
        self.lookback = int(metadata["config"]["lookback"])
        self.sku_to_idx = metadata["sku_to_idx"]
        self.scalers = metadata["scalers"]
        self.calendar_covariates = tuple(metadata.get("calendar_covariates", []))

    def _tensors(self, window_row: Mapping, candidate_price: float):
        row = dict(window_row)
        row["price"] = float(candidate_price)
        row["sku"] = str(row["sku"])
        if "month" not in row or "week_of_year" not in row:
            week = pd.Timestamp(row["week"])
            row["month"] = int(week.month)
            row["week_of_year"] = int(week.isocalendar().week)
        encoding = self.metadata.get("sku_encoding", "onehot")
        features = feature_arrays(
            pd.DataFrame([row]), self.sku_to_idx, self.scalers, self.lookback,
            True, self.calendar_covariates, encoding,
        )
        h, a = features[:2]
        sku_index = torch.from_numpy(features[2]) if encoding == "embedding" else None
        return torch.from_numpy(h), torch.tensor(a, dtype=torch.float32, requires_grad=True), sku_index

    def predict_q(self, window_row: Mapping, candidate_price: float) -> float:
        self.model.eval()
        h, a, sku_index = self._tensors(window_row, candidate_price)
        with torch.no_grad():
            return max(0.0, float(torch.expm1(self.model(h, a, sku_index))[0]))

    def price_gradient(self, window_row: Mapping, current_price: float) -> float:
        self.model.eval()
        h, a, sku_index = self._tensors(window_row, current_price)
        q = torch.expm1(self.model(h, a, sku_index))[0]
        dq_dnormalized_price = torch.autograd.grad(q, a)[0][0, -1]
        price_std = float(self.scalers[str(window_row["sku"])]["price_std"])
        return float(dq_dnormalized_price / price_std)

    def fine_tune(
        self,
        samples: Sequence[tuple[Mapping, float, float]],
        epochs: int = 1,
        lr: float = 1e-4,
        batch_size: int = 32,
    ) -> float | None:
        if not samples:
            return None
        frames = []
        quantities = []
        for row, price, qty in samples:
            item = dict(row)
            item["price"] = float(price)
            frames.append(item)
            quantities.append(max(0.0, float(qty)))
        features = feature_arrays(
            pd.DataFrame(frames), self.sku_to_idx, self.scalers, self.lookback,
            True, self.calendar_covariates, self.metadata.get("sku_encoding", "onehot"),
        )
        h, a = features[:2]
        sku_index = features[2] if len(features) == 3 else None
        y = np.log1p(np.array(quantities, dtype=np.float32))
        tensors = [torch.from_numpy(h), torch.from_numpy(a)]
        if sku_index is not None:
            tensors.append(torch.from_numpy(sku_index))
        tensors.append(torch.from_numpy(y))
        dataset = torch.utils.data.TensorDataset(*tensors)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        mean_loss = 0.0
        for _ in range(epochs):
            total = 0.0
            count = 0
            for batch in loader:
                bh, ba = batch[:2]
                if self.model.sku_count:
                    bsku, by = batch[2], batch[3]
                else:
                    bsku, by = None, batch[2]
                optimizer.zero_grad(set_to_none=True)
                loss = F.mse_loss(self.model(bh, ba, bsku), by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach()) * len(by)
                count += len(by)
            mean_loss = total / max(count, 1)
        self.model.eval()
        return mean_loss


def load_price_forecaster(output_dir: str | Path, train_only: bool = False) -> PriceForecaster:
    """Load either the final forecast or train-only pricing checkpoint."""
    import json

    out = Path(output_dir)
    name = "transformer_price_train_only" if train_only else "transformer_price"
    with (out / f"{name}_meta.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    model = SequenceForecaster(**metadata["config"])
    model.load_state_dict(torch.load(out / f"{name}.pt", map_location="cpu", weights_only=True))
    return PriceForecaster(model, metadata)
