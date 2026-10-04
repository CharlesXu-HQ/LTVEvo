"""Build leakage-safe customer snapshots from the complete retail workbook."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from .task import TaskSpec


FEATURE_COLUMNS = [
    "history_spend",
    "history_purchase_lines",
    "history_orders",
    "history_distinct_products",
    "recency_days",
    "spend_30d",
    "spend_90d",
    "avg_order_value",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_workbooks(task: TaskSpec) -> pd.DataFrame:
    def sheets(source: str | Path | io.BytesIO) -> list[pd.DataFrame]:
        frames = pd.read_excel(source, sheet_name=None, engine="openpyxl")
        return [_normalize_sheet(frame, task) for frame in frames.values()]

    if task.raw_path.suffix.lower() == ".xlsx":
        parts = sheets(task.raw_path)
    elif task.raw_path.suffix.lower() == ".csv":
        parts = [_normalize_sheet(pd.read_csv(task.raw_path, low_memory=False), task)]
    elif task.raw_path.suffix.lower() == ".zip":
        parts = []
        with ZipFile(task.raw_path) as archive:
            workbook_names = sorted(name for name in archive.namelist() if name.lower().endswith(".xlsx"))
            csv_names = sorted(name for name in archive.namelist() if name.lower().endswith(".csv"))
            if workbook_names:
                for name in workbook_names:
                    parts.extend(sheets(io.BytesIO(archive.read(name))))
            elif csv_names:
                for name in csv_names:
                    with archive.open(name) as source:
                        parts.append(_normalize_sheet(pd.read_csv(source, low_memory=False), task))
            else:
                raise ValueError("ZIP contains no XLSX workbook or CSV file")
    else:
        raise ValueError("raw_path must be XLSX, CSV, or ZIP containing either format")
    if not parts:
        raise ValueError("workbook contains no sheets")
    return pd.concat(parts, ignore_index=True)


def _normalize_sheet(frame: pd.DataFrame, task: TaskSpec) -> pd.DataFrame:
    selected = {}
    for canonical, aliases in task.column_mapping.items():
        source = next((name for name in aliases if name in frame.columns), None)
        if source is None:
            raise ValueError(f"missing {canonical} column; expected one of {aliases}")
        selected[canonical] = frame[source]
    return pd.DataFrame(selected)


def _purchases(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    rows = raw.copy()
    rows["invoice_date"] = pd.to_datetime(rows["invoice_date"], errors="coerce", format="mixed")
    rows["quantity"] = pd.to_numeric(rows["quantity"], errors="coerce")
    rows["unit_price"] = pd.to_numeric(rows["unit_price"], errors="coerce")
    rows["customer_id"] = rows["customer_id"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    rows["invoice"] = rows["invoice"].astype("string").str.strip()
    rows["stock_code"] = rows["stock_code"].astype("string").str.strip()

    remaining = pd.Series(True, index=rows.index)
    excluded = {}
    conditions = {
        "invalid_timestamp": rows["invoice_date"].isna(),
        "missing_customer_id": rows["customer_id"].isna() | rows["customer_id"].eq(""),
        "cancellation": rows["invoice"].str.upper().str.startswith("C").fillna(False),
        "nonpositive_quantity": ~(rows["quantity"] > 0),
        "nonpositive_price": ~(rows["unit_price"] > 0),
    }
    for reason, condition in conditions.items():
        rejected = remaining & condition.fillna(False)
        excluded[reason] = int(rejected.sum())
        remaining &= ~rejected
    valid = rows.loc[remaining].copy()
    valid["amount"] = valid["quantity"] * valid["unit_price"]
    return valid, excluded


def _snapshot(purchases: pd.DataFrame, as_of: pd.Timestamp, horizon_days: int) -> pd.DataFrame:
    history = purchases.loc[purchases["invoice_date"] < as_of]
    if history.empty:
        return pd.DataFrame(columns=["customer_id", "as_of", "target", *FEATURE_COLUMNS])
    grouped = history.groupby("customer_id", sort=True)
    result = grouped.agg(
        history_spend=("amount", "sum"),
        history_purchase_lines=("amount", "size"),
        history_orders=("invoice", "nunique"),
        history_distinct_products=("stock_code", "nunique"),
        last_purchase=("invoice_date", "max"),
    )
    result["recency_days"] = (as_of - result.pop("last_purchase")).dt.total_seconds() / 86400
    for days in (30, 90):
        recent = history.loc[history["invoice_date"] >= as_of - pd.Timedelta(days=days)]
        spend = recent.groupby("customer_id")["amount"].sum()
        result[f"spend_{days}d"] = spend.reindex(result.index, fill_value=0)
    result["avg_order_value"] = result["history_spend"] / result["history_orders"].replace(0, float("nan"))
    result["avg_order_value"] = result["avg_order_value"].fillna(0)
    future = purchases.loc[
        (purchases["invoice_date"] >= as_of)
        & (purchases["invoice_date"] < as_of + pd.Timedelta(days=horizon_days))
    ]
    labels = future.groupby("customer_id")["amount"].sum()
    result["target"] = labels.reindex(result.index, fill_value=0)
    result["as_of"] = as_of.date().isoformat()
    return result.reset_index()[["customer_id", "as_of", "target", *FEATURE_COLUMNS]]


def prepare(task: TaskSpec, output_dir: str | Path) -> dict:
    """Read all rows, verify provenance, and write frozen split CSVs plus metadata."""
    actual_hash = _sha256(task.raw_path)
    if actual_hash != task.raw_sha256:
        raise ValueError("raw data SHA-256 hash differs from task specification")
    raw = _read_workbooks(task)
    purchases, excluded = _purchases(raw)
    if purchases.empty:
        raise ValueError("no eligible purchase rows remain")
    observed_until = purchases["invoice_date"].max()
    last_label_end = pd.Timestamp(task.test_end) + pd.Timedelta(days=task.horizon_days)
    if last_label_end > observed_until:
        raise ValueError("test label window is not fully observable; choose a mature end date")
    after_window = int((purchases["invoice_date"] >= last_label_end).sum())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    split_hashes = {}
    for split in ("train", "validation", "test"):
        snapshots = [
            _snapshot(purchases, pd.Timestamp(day), task.horizon_days)
            for day in task.dates_for(split)
        ]
        frame = pd.concat(snapshots, ignore_index=True)
        if frame.empty:
            raise ValueError(f"{split} has no known customers before its observation dates")
        split_path = output_dir / f"{split}.csv"
        frame.to_csv(split_path, index=False)
        counts[split] = len(frame)
        split_hashes[split] = _sha256(split_path)
    metadata = {
        "raw_sha256": actual_hash,
        "task_fingerprint": task.fingerprint,
        "target_policy": task.target_policy,
        "horizon_days": task.horizon_days,
        "feature_columns": FEATURE_COLUMNS,
        "raw_rows": len(raw),
        "eligible_purchase_rows": len(purchases),
        "transaction_rows_used": len(purchases) - after_window,
        "excluded_rows": excluded,
        "rows_after_last_label_window": after_window,
        "data_span": {
            "first_purchase": purchases["invoice_date"].min().isoformat(),
            "last_purchase": observed_until.isoformat(),
        },
        "snapshot_counts": counts,
        "split_sha256": split_hashes,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
