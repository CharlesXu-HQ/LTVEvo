"""Frozen data and evaluation identity for an LTV prediction task."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


DEFAULT_COLUMNS = {
    "invoice": ("Invoice", "InvoiceNo"),
    "stock_code": ("StockCode",),
    "quantity": ("Quantity",),
    "invoice_date": ("InvoiceDate",),
    "unit_price": ("Price", "UnitPrice"),
    "customer_id": ("Customer ID", "CustomerID"),
}
DATE_FIELDS = (
    "as_of_start", "as_of_end", "train_end", "validation_start",
    "validation_end", "test_start", "test_end",
)


@dataclass(frozen=True)
class TaskSpec:
    raw_path: Path
    raw_sha256: str
    horizon_days: int
    as_of_start: date
    as_of_end: date
    train_end: date
    validation_start: date
    validation_end: date
    test_start: date
    test_end: date
    seed: int
    target_policy: str
    primary_metric: str
    evaluator_version: str
    column_mapping: Mapping[str, tuple[str, ...]]

    @classmethod
    def load(cls, path: str | Path) -> "TaskSpec":
        path = Path(path).resolve()
        config = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("task JSON must be an object")
        required = {
            "raw_path", "raw_sha256", "horizon_days", "as_of_start", "as_of_end",
            "train_end", "validation_start", "validation_end", "test_start", "test_end",
            "seed", "target_policy", "primary_metric", "evaluator_version",
        }
        missing = sorted(required - config.keys())
        if missing:
            raise ValueError(f"missing task fields: {', '.join(missing)}")
        raw_path = Path(config["raw_path"]).expanduser()
        if not raw_path.is_absolute():
            raw_path = path.parent / raw_path
        digest = config["raw_sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", digest):
            raise ValueError("raw_sha256 must be a SHA-256 hex digest")
        horizon = config["horizon_days"]
        seed = config["seed"]
        if type(horizon) is not int or horizon < 1:
            raise ValueError("horizon_days must be a positive integer")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if config["target_policy"] != "positive_purchase_amount":
            raise ValueError("unsupported target_policy")
        if config["primary_metric"] != "mae":
            raise ValueError("unsupported primary_metric")
        if not isinstance(config["evaluator_version"], str) or not config["evaluator_version"]:
            raise ValueError("evaluator_version must be a nonempty string")

        dates = {}
        for field in DATE_FIELDS:
            try:
                value = datetime.strptime(config[field], "%Y-%m-%d").date()
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field} must use YYYY-MM-DD") from exc
            if value.day != 1:
                raise ValueError(f"{field} must be the first day of a month")
            dates[field] = value
        if not (
            dates["as_of_start"] <= dates["train_end"]
            < dates["validation_start"] <= dates["validation_end"]
            < dates["test_start"] <= dates["test_end"]
            <= dates["as_of_end"]
        ):
            raise ValueError("train, validation, and test observation ranges must be ordered")
        if dates["train_end"] + timedelta(days=horizon) > dates["validation_start"]:
            raise ValueError("training label window overlaps validation observations; purge the split")
        if dates["validation_end"] + timedelta(days=horizon) > dates["test_start"]:
            raise ValueError("validation label window overlaps test observations; purge the split")

        mapping = dict(DEFAULT_COLUMNS)
        custom = config.get("column_mapping", {})
        if not isinstance(custom, dict) or set(custom) - set(DEFAULT_COLUMNS):
            raise ValueError("column_mapping has unknown canonical columns")
        for canonical, aliases in custom.items():
            if isinstance(aliases, str):
                aliases = [aliases]
            if not isinstance(aliases, list) or not aliases or not all(
                isinstance(alias, str) and alias for alias in aliases
            ):
                raise ValueError(f"column_mapping.{canonical} must be a nonempty list of names")
            mapping[canonical] = tuple(aliases)
        return cls(
            raw_path=raw_path.resolve(),
            raw_sha256=digest.lower(),
            horizon_days=horizon,
            seed=seed,
            target_policy=config["target_policy"],
            primary_metric=config["primary_metric"],
            evaluator_version=config["evaluator_version"],
            column_mapping=MappingProxyType(mapping),
            **dates,
        )

    @property
    def fingerprint(self) -> str:
        identity = {
            "raw_sha256": self.raw_sha256,
            "horizon_days": self.horizon_days,
            "seed": self.seed,
            "target_policy": self.target_policy,
            "primary_metric": self.primary_metric,
            "evaluator_version": self.evaluator_version,
            "column_mapping": dict(self.column_mapping),
            **{field: getattr(self, field).isoformat() for field in DATE_FIELDS},
        }
        payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def dates_for(self, split: str) -> list[date]:
        if split == "train":
            start, end = self.as_of_start, self.train_end
        elif split == "validation":
            start, end = self.validation_start, self.validation_end
        elif split == "test":
            start, end = self.test_start, self.test_end
        else:
            raise ValueError(f"unknown split: {split}")
        current = start
        dates = []
        while current <= end:
            dates.append(current)
            year, month = current.year, current.month + 1
            if month == 13:
                year, month = year + 1, 1
            current = date(year, month, 1)
        return dates
