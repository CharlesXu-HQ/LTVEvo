"""The small workbook here tests the full import path, not model quality."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from ltvevo.data import prepare
from ltvevo.task import TaskSpec


class TaskDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workbook = self.root / "retail.xlsx"
        older = pd.DataFrame(
            [
                ("A", "S1", 1, "2020-12-31 12:00", 10, 1),
                ("B", "S1", 1, "2021-01-01 00:00", 20, 1),
                ("N3", "S2", 1, "2021-01-03 12:00", 5, 2),
                ("D", "S3", 1, "2021-01-03 12:00", 100, None),
                ("C1", "S1", -1, "2021-01-04 12:00", 20, 1),
                ("E", "S1", 0, "2021-01-05 12:00", 20, 1),
                ("F", "S1", 1, "2021-01-06 12:00", 0, 1),
                ("G", "S1", 1, None, 1, 1),
            ],
            columns=["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Customer ID"],
        )
        older["InvoiceDate"] = pd.to_datetime(older["InvoiceDate"])
        modern = pd.DataFrame(
            [
                ("H", "S2", 2, "2021-02-01 00:00", 7, 2),
                ("I", "S1", 1, "2021-03-01 00:00", 3, 1),
                ("J", "S1", 1, "2021-03-05 12:00", 4, 1),
                ("K", "S1", 1, "2021-03-10 12:00", 100, 1),
            ],
            columns=["InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"],
        )
        with pd.ExcelWriter(self.workbook, engine="openpyxl") as writer:
            older.to_excel(writer, sheet_name="Year 2009-2010", index=False)
            modern.to_excel(writer, sheet_name="Year 2010-2011", index=False)

    def task_path(self, **overrides: object) -> Path:
        task = {
            "raw_path": str(self.workbook),
            "raw_sha256": hashlib.sha256(self.workbook.read_bytes()).hexdigest(),
            "horizon_days": 7,
            "as_of_start": "2021-01-01",
            "as_of_end": "2021-03-01",
            "train_end": "2021-01-01",
            "validation_start": "2021-02-01",
            "validation_end": "2021-02-01",
            "test_start": "2021-03-01",
            "test_end": "2021-03-01",
            "seed": 7,
            "target_policy": "positive_purchase_amount",
            "primary_metric": "mae",
            "evaluator_version": "1",
        }
        task.update(overrides)
        path = self.root / "task.json"
        path.write_text(json.dumps(task), encoding="utf-8")
        return path

    def test_rejects_label_windows_crossing_next_split(self) -> None:
        path = self.task_path(horizon_days=32)
        with self.assertRaisesRegex(ValueError, "purge|overlap|window"):
            TaskSpec.load(path)

    def test_fingerprint_uses_dataset_hash_not_local_path(self) -> None:
        path = self.task_path()
        first = TaskSpec.load(path)
        copy = self.root / "copy.xlsx"
        copy.write_bytes(self.workbook.read_bytes())
        second = TaskSpec.load(self.task_path(raw_path=str(copy)))
        self.assertEqual(first.fingerprint, second.fingerprint)
        changed = TaskSpec.load(self.task_path(horizon_days=8))
        self.assertNotEqual(first.fingerprint, changed.fingerprint)

    def test_task_column_mapping_cannot_mutate_after_fingerprint(self) -> None:
        task = TaskSpec.load(self.task_path())
        with self.assertRaises(TypeError):
            task.column_mapping["invoice"] = ("other",)

    def test_prepares_both_sheets_without_leakage_or_sampling(self) -> None:
        task = TaskSpec.load(self.task_path())
        metadata = prepare(task, self.root / "prepared")
        train = pd.read_csv(self.root / "prepared/train.csv", dtype={"customer_id": "string"})
        validation = pd.read_csv(self.root / "prepared/validation.csv", dtype={"customer_id": "string"})
        test = pd.read_csv(self.root / "prepared/test.csv", dtype={"customer_id": "string"})

        self.assertEqual(metadata["raw_rows"], 12)
        self.assertEqual(metadata["eligible_purchase_rows"], 7)
        self.assertEqual(metadata["transaction_rows_used"], 6)
        self.assertEqual(metadata["rows_after_last_label_window"], 1)
        self.assertEqual(metadata["excluded_rows"], {
            "invalid_timestamp": 1,
            "missing_customer_id": 1,
            "cancellation": 1,
            "nonpositive_quantity": 1,
            "nonpositive_price": 1,
        })
        self.assertEqual(metadata["snapshot_counts"], {"train": 1, "validation": 2, "test": 2})
        self.assertEqual(metadata["task_fingerprint"], task.fingerprint)
        self.assertEqual(
            metadata["split_sha256"]["test"],
            hashlib.sha256((self.root / "prepared/test.csv").read_bytes()).hexdigest(),
        )
        self.assertIn("spend_90d", metadata["feature_columns"])
        self.assertEqual(train.loc[0, "customer_id"], "1")
        self.assertEqual(train.loc[0, "history_spend"], 10)
        self.assertEqual(train.loc[0, "spend_90d"], 10)
        self.assertEqual(train.loc[0, "target"], 20)
        self.assertEqual(validation.loc[validation.customer_id == "1", "history_spend"].iloc[0], 30)
        self.assertEqual(validation.loc[validation.customer_id == "2", "target"].iloc[0], 14)
        self.assertEqual(test.loc[test.customer_id == "1", "target"].iloc[0], 7)
        self.assertEqual(test.loc[test.customer_id == "2", "target"].iloc[0], 0)

    def test_rejects_raw_data_changed_after_task_creation(self) -> None:
        task = TaskSpec.load(self.task_path(raw_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "SHA-256|hash"):
            prepare(task, self.root / "prepared")

    def test_rejects_immature_test_label_window(self) -> None:
        task = TaskSpec.load(self.task_path(horizon_days=10))
        with self.assertRaisesRegex(ValueError, "mature|observable|window"):
            prepare(task, self.root / "prepared")

    def test_reads_workbook_from_uci_style_zip(self) -> None:
        archive = self.root / "retail.zip"
        with ZipFile(archive, "w") as output:
            output.write(self.workbook, "online_retail_II.xlsx")
        task = TaskSpec.load(self.task_path(
            raw_path=str(archive),
            raw_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        ))
        metadata = prepare(task, self.root / "prepared")
        self.assertEqual(metadata["raw_rows"], 12)
        self.assertEqual(metadata["snapshot_counts"]["test"], 2)

    def test_reads_complete_csv_from_public_mirror_zip(self) -> None:
        csv_path = self.root / "retail.csv"
        sheets = list(pd.read_excel(self.workbook, sheet_name=None).values())
        second = sheets[1].rename(columns={
            "InvoiceNo": "Invoice", "UnitPrice": "Price", "CustomerID": "Customer ID",
        })
        pd.concat([sheets[0], second], ignore_index=True).to_csv(csv_path, index=False)
        archive = self.root / "retail-csv.zip"
        with ZipFile(archive, "w") as output:
            output.write(csv_path, "online_retail_II.csv")
        task = TaskSpec.load(self.task_path(
            raw_path=str(archive),
            raw_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        ))
        metadata = prepare(task, self.root / "prepared")
        self.assertEqual(metadata["raw_rows"], 12)
        self.assertEqual(metadata["eligible_purchase_rows"], 7)


if __name__ == "__main__":
    unittest.main()
