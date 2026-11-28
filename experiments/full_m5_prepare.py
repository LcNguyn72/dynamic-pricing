"""Prepare reproducible weekly panels for all ten M5 stores, one at a time."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .m5 import load_m5_weekly


STORE_IDS = (
    "CA_1", "CA_2", "CA_3", "CA_4",
    "TX_1", "TX_2", "TX_3",
    "WI_1", "WI_2", "WI_3",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(data_dir: str | Path, output_root: str | Path, *, chunksize: int = 128) -> dict:
    data_dir, output_root = Path(data_dir), Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    source_sha256 = {
        name: _sha256(data_dir / name)
        for name in ("sales_train_evaluation.csv", "sell_prices.csv", "calendar.csv")
    }
    prepared = []
    for store in STORE_IDS:
        print(f"[prepare-m5] {store}: loading", flush=True)
        weekly, manifest = load_m5_weekly(
            data_dir,
            sales_file="sales_train_evaluation.csv",
            store_ids=(store,),
            split_profile="paper_70_15_15",
            cold_start_policy="drop",
            chunksize=chunksize,
        )
        weekly_path = output_root / f"weekly_{store}.csv"
        manifest_path = output_root / f"manifest_{store}.json"
        weekly.to_csv(weekly_path, index=False)
        manifest["lookback_weeks"] = 12
        manifest["protocol_id"] = "song_full_scope_v2"
        manifest["store_id"] = store
        manifest["source_sha256"] = source_sha256
        manifest["weekly_sha256"] = _sha256(weekly_path)
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
        prepared.append({
            "store": store,
            "requested_series": int(manifest["requested_series"]),
            "n_series": int(manifest["n_series"]),
            "weekly_rows": int(len(weekly)),
            "weekly_path": str(weekly_path),
            "weekly_sha256": manifest["weekly_sha256"],
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
        })
        print(f"[prepare-m5] {store}: saved {len(weekly):,} rows", flush=True)
        del weekly, manifest
    summary = {
        "schema_version": 2,
        "protocol_id": "song_full_scope_v2",
        "source": str(data_dir.resolve()),
        "sales_file": "sales_train_evaluation.csv",
        "source_sha256": source_sha256,
        "split_profile": "paper_70_15_15",
        "lookback_weeks": 12,
        "stores": prepared,
    }
    summary_path = output_root / "full_m5_preparation.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[prepare-m5] complete: {summary_path}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/song_replication/m5"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/artifacts/song_m5_full"))
    parser.add_argument("--chunksize", type=int, default=128)
    args = parser.parse_args()
    prepare(args.data_dir, args.output_root, chunksize=args.chunksize)


if __name__ == "__main__":
    main()
