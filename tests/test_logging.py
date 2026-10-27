from __future__ import annotations

import json
import logging

from vehicle_anomaly.logging_utils import StructuredFormatter


def _record(**extra: object) -> logging.LogRecord:
    record = logging.LogRecord("vehicle_anomaly.test", logging.INFO, __file__, 1, "processed %s", ("batch",), None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_includes_extras() -> None:
    payload = json.loads(StructuredFormatter(json_format=True).format(_record(batch_id=3, invalid_rows=7)))
    assert payload["message"] == "processed batch"
    assert payload["level"] == "INFO" and payload["logger"] == "vehicle_anomaly.test"
    assert payload["batch_id"] == 3 and payload["invalid_rows"] == 7
    assert "args" not in payload and "msg" not in payload


def test_json_formatter_drops_uvicorn_color_message() -> None:
    payload = json.loads(StructuredFormatter(json_format=True).format(_record(color_message="\x1b[36mx")))
    assert "color_message" not in payload


def test_text_formatter_appends_key_values() -> None:
    line = StructuredFormatter(json_format=False).format(_record(batch_id=3))
    assert "processed batch" in line and "batch_id=3" in line
