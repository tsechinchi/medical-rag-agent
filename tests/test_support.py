from __future__ import annotations

import copy
import math
import sys
import types
from typing import Any


class FakeSeries:
    def __init__(self, values: Any = None, index: Any = None) -> None:
        if values is None:
            self._values = []
        elif isinstance(values, FakeSeries):
            self._values = list(values._values)
        else:
            self._values = list(values)

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    @property
    def empty(self) -> bool:
        return len(self._values) == 0

    def __getitem__(self, key):
        if isinstance(key, FakeSeries):
            key = key._values
        if isinstance(key, list) and not key:
            return FakeSeries([])
        if isinstance(key, list) and key and all(isinstance(item, bool) for item in key):
            return FakeSeries([value for value, keep in zip(self._values, key) if keep])
        return self._values[key]

    def fillna(self, value: Any):
        return FakeSeries(
            [
                value if item is None or (isinstance(item, float) and math.isnan(item)) else item
                for item in self._values
            ]
        )

    def astype(self, dtype: Any):
        if dtype in (str, "str"):
            return FakeSeries(["" if item is None else str(item) for item in self._values])
        if dtype in (int, "int"):
            return FakeSeries([int(float(item)) for item in self._values])
        if dtype in (float, "float"):
            return FakeSeries([float(item) for item in self._values])
        return FakeSeries(self._values)

    def unique(self):
        seen = []
        for item in self._values:
            if item not in seen:
                seen.append(item)
        return FakeSeries(seen)

    def tolist(self):
        return list(self._values)

    def dropna(self):
        return FakeSeries(
            [
                item
                for item in self._values
                if item is not None and not (isinstance(item, float) and math.isnan(item))
            ]
        )

    def mean(self):
        numeric = [
            float(item)
            for item in self._values
            if item is not None and not (isinstance(item, float) and math.isnan(item))
        ]
        return sum(numeric) / len(numeric) if numeric else float("nan")

    def __eq__(self, other: Any):
        return FakeSeries([item == other for item in self._values])


class FakeDataFrame:
    def __init__(self, data: Any = None) -> None:
        if data is None:
            self._rows: list[dict[str, Any]] = []
        elif isinstance(data, list):
            self._rows = [dict(row) for row in data]
        elif isinstance(data, dict):
            keys = list(data.keys())
            lengths = [len(value) for value in data.values() if isinstance(value, list)]
            row_count = max(lengths, default=1)
            rows: list[dict[str, Any]] = []
            for idx in range(row_count):
                row: dict[str, Any] = {}
                for key in keys:
                    value = data[key]
                    if isinstance(value, list):
                        row[key] = value[idx] if idx < len(value) else None
                    else:
                        row[key] = value
                rows.append(row)
            self._rows = rows
        else:
            raise TypeError("Unsupported FakeDataFrame data type")

    @property
    def columns(self):
        seen: list[str] = []
        for row in self._rows:
            for key in row.keys():
                if key not in seen:
                    seen.append(key)
        return seen

    @property
    def empty(self) -> bool:
        return len(self._rows) == 0

    @property
    def index(self):
        return list(range(len(self._rows)))

    def copy(self):
        return FakeDataFrame(copy.deepcopy(self._rows))

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return FakeSeries([row.get(key) for row in self._rows])
        if isinstance(key, FakeSeries):
            key = key.tolist()
        if isinstance(key, list) and not key:
            return FakeDataFrame([])
        if isinstance(key, list) and key and all(isinstance(item, bool) for item in key):
            return FakeDataFrame([row for row, keep in zip(self._rows, key) if keep])
        if isinstance(key, list) and all(isinstance(item, str) for item in key):
            return FakeDataFrame([{column: row.get(column) for column in key} for row in self._rows])
        raise KeyError(key)

    def __setitem__(self, key, value):
        if isinstance(value, FakeSeries):
            value = value.tolist()
        if isinstance(value, list):
            for row, item in zip(self._rows, value):
                row[key] = item
            if len(value) == 1 and len(self._rows) > 1:
                for row in self._rows[1:]:
                    row[key] = value[0]
        else:
            for row in self._rows:
                row[key] = value

    def drop_duplicates(self, subset, keep="last"):
        if isinstance(subset, str):
            subset = [subset]
        seen: dict[tuple[Any, ...], dict[str, Any]] = {}
        order: list[tuple[Any, ...]] = []
        for row in self._rows:
            key = tuple(row.get(column) for column in subset)
            if key not in seen:
                order.append(key)
            if keep == "last" or key not in seen:
                seen[key] = dict(row)
        rows = [seen[key] for key in order]
        return FakeDataFrame(rows)

    def merge(self, other, on, how="inner", suffixes=("_x", "_y")):
        if isinstance(on, str):
            on = [on]
        other_rows = other._rows if isinstance(other, FakeDataFrame) else list(other)
        merged: list[dict[str, Any]] = []
        for left in self._rows:
            for right in other_rows:
                if all(left.get(column) == right.get(column) for column in on):
                    row = dict(left)
                    for key, value in right.items():
                        if key in on:
                            continue
                        if key in row:
                            row[f"{key}{suffixes[1]}"] = value
                        else:
                            row[key] = value
                    merged.append(row)
        return FakeDataFrame(merged)

    def to_dict(self, orient="records"):
        if orient != "records":
            raise ValueError("FakeDataFrame only supports orient='records'")
        return [dict(row) for row in self._rows]


def install_fake_pandas() -> None:
    if "pandas" in sys.modules:
        return

    fake = types.ModuleType("pandas")
    fake.DataFrame = FakeDataFrame
    fake.Series = FakeSeries

    def to_numeric(values, errors="coerce"):
        if isinstance(values, FakeSeries):
            iterable = values.tolist()
        else:
            iterable = list(values)
        converted = []
        for item in iterable:
            try:
                converted.append(float(item))
            except Exception:
                converted.append(float("nan"))
        return FakeSeries(converted)

    def read_csv(*args, **kwargs):
        raise RuntimeError("read_csv is not needed in these tests")

    def isna(value):
        return value is None or (isinstance(value, float) and math.isnan(value))

    def notna(value):
        return not isna(value)

    fake.to_numeric = to_numeric
    fake.read_csv = read_csv
    fake.isna = isna
    fake.notna = notna
    sys.modules["pandas"] = fake


def install_fake_torch_transformers() -> None:
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")

        class _Cuda:
            @staticmethod
            def is_available():
                return False

        class _InferenceMode:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        torch.cuda = _Cuda()
        torch.inference_mode = lambda: _InferenceMode()
        torch.softmax = lambda values, dim=-1: values
        torch.bfloat16 = "bfloat16"
        torch.float16 = "float16"
        torch.device = lambda value: value
        sys.modules["torch"] = torch

    if "transformers" not in sys.modules:
        transformers = types.ModuleType("transformers")

        class _AutoTokenizer:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

        class _AutoModelForSequenceClassification:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def to(self, *args, **kwargs):
                return self

            def eval(self):
                return self

        transformers.AutoTokenizer = _AutoTokenizer
        transformers.AutoModelForSequenceClassification = _AutoModelForSequenceClassification
        sys.modules["transformers"] = transformers
