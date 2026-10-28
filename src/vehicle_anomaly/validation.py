"""Event validation rules for raw JSON telemetry (pandas / pure-Python implementation).

The Spark engine (``streaming.py``) implements the same rules with Spark SQL expressions. Checks
run in this order and the first failure becomes the ``invalid_reason``:

1. ``malformed_json``                 - line is not a JSON object
2. ``missing_field:<name>``          - required field absent, null or blank
3. ``non_numeric:<name>``            - numeric field cannot be parsed as a finite-or-infinite double
4. ``invalid_timestamp``             - event_timestamp is not ISO-8601
5. ``out_of_range:<name>``           - numeric value outside ``VALID_RANGES``
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from vehicle_anomaly.schema import ID_FIELDS, NUMERIC_FIELDS, REQUIRED_FIELDS, TIMESTAMP_FIELD, VALID_RANGES

INVALID_COLUMNS = ["raw_value", "invalid_reason"]
VALID_COLUMNS = list(REQUIRED_FIELDS)


@dataclass
class ValidationResult:
    valid: pd.DataFrame
    invalid: pd.DataFrame
    input_rows: int


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _to_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def validate_record(record: Any) -> tuple[str | None, dict[str, Any] | None]:
    """Return ``(invalid_reason, None)`` or ``(None, typed_record)``."""
    if not isinstance(record, dict):
        return "malformed_json", None
    for name in REQUIRED_FIELDS:
        if _is_missing(record.get(name)):
            return f"missing_field:{name}", None
    numbers: dict[str, float] = {}
    for name in NUMERIC_FIELDS:
        number = _to_float(record[name])
        if number is None or math.isnan(number):
            return f"non_numeric:{name}", None
        numbers[name] = number
    timestamp = _to_timestamp(record[TIMESTAMP_FIELD])
    if timestamp is None:
        return "invalid_timestamp", None
    for name in NUMERIC_FIELDS:
        low, high = VALID_RANGES[name]
        if not low <= numbers[name] <= high:
            return f"out_of_range:{name}", None
    typed: dict[str, Any] = {name: str(record[name]).strip() for name in ID_FIELDS}
    typed[TIMESTAMP_FIELD] = timestamp
    typed.update(numbers)
    return None, typed


def validate_lines(lines: Iterable[str]) -> ValidationResult:
    """Validate raw JSON Lines. Blank lines are ignored (they are not counted as input rows)."""
    valid_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, str]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            record = None
        reason, typed = validate_record(record)
        if reason is None:
            valid_rows.append(typed)  # type: ignore[arg-type]
        else:
            invalid_rows.append({"raw_value": line.rstrip("\r\n"), "invalid_reason": reason})

    valid = pd.DataFrame(valid_rows, columns=VALID_COLUMNS)
    valid[TIMESTAMP_FIELD] = pd.to_datetime(valid[TIMESTAMP_FIELD], utc=True)
    for name in NUMERIC_FIELDS:
        valid[name] = valid[name].astype("float64")
    invalid = pd.DataFrame(invalid_rows, columns=INVALID_COLUMNS)
    return ValidationResult(valid=valid, invalid=invalid, input_rows=len(valid_rows) + len(invalid_rows))
