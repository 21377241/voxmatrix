import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


BASE_COLUMNS = ("id", "prompt", "inference", "post_process", "error")
EXCEL_TEXT_LIMIT = 32767


def _excel_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False)
    if isinstance(value, str):
        value = "".join(
            character
            for character in value
            if ord(character) >= 32 or character in "\t\n\r"
        )
        if len(value) > EXCEL_TEXT_LIMIT:
            suffix = "...[truncated]"
            value = value[: EXCEL_TEXT_LIMIT - len(suffix)] + suffix
    return value


def _iter_events(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                event["id"] = int(event["id"])
                event["type"] = str(event["type"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid event at {path}:{line_no}: {exc}") from exc
            yield event


def _eval_columns(path: Path) -> list[str]:
    keys = set()
    for event in _iter_events(path):
        if event["type"] == "eval" and isinstance(event.get("data"), dict):
            keys.update(str(key) for key in event["data"])
    return sorted(keys)


def export_events_xlsx(
    event_path: Path,
    output_path: Path,
    expected_count: Optional[int] = None,
    expected_start: int = 0,
) -> Dict[str, Any]:
    from openpyxl import Workbook

    eval_keys = _eval_columns(event_path)
    eval_headers = {
        key: key if key not in BASE_COLUMNS else f"eval_{key}" for key in eval_keys
    }
    columns = list(BASE_COLUMNS) + [eval_headers[key] for key in eval_keys]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp")

    workbook = Workbook(write_only=True)
    sheet = workbook.create_sheet("events")
    sheet.append(columns)

    current_id = None
    first_id = None
    current_row: Dict[str, Any] = {}
    current_types = set()
    row_count = 0

    def append_current() -> None:
        nonlocal row_count
        if current_id is None:
            return
        sheet.append([_excel_value(current_row.get(column)) for column in columns])
        row_count += 1

    try:
        for event in _iter_events(event_path):
            event_id = event["id"]
            event_type = event["type"]
            if first_id is None:
                first_id = event_id
            if current_id is not None and event_id < current_id:
                raise ValueError(
                    "events must be ordered by non-decreasing id before XLSX export"
                )
            if event_id != current_id:
                append_current()
                current_id = event_id
                current_row = {"id": event_id}
                current_types = set()
            if event_type in current_types:
                raise ValueError(f"duplicate event type for id {event_id}: {event_type}")
            current_types.add(event_type)

            data = event.get("data")
            if event_type in ("prompt", "inference", "post_process"):
                current_row[event_type] = (
                    data.get("content") if isinstance(data, dict) else data
                )
            elif event_type == "eval" and isinstance(data, dict):
                for key, value in data.items():
                    current_row[eval_headers[str(key)]] = value
            elif event_type == "error":
                current_row["error"] = data

        append_current()
        if expected_count is not None:
            expected_ids = range(expected_start, expected_start + expected_count)
            if row_count != expected_count:
                raise ValueError(
                    f"expected {expected_count} rows, exported {row_count}"
                )
            if first_id != expected_start:
                raise ValueError(
                    f"expected first id {expected_start}, got {first_id}"
                )
            if current_id != expected_ids.stop - 1:
                raise ValueError(
                    f"expected final id {expected_ids.stop - 1}, got {current_id}"
                )
        workbook.save(temp_path)
        os.replace(temp_path, output_path)
    except BaseException:
        if temp_path.exists():
            temp_path.unlink()
        raise

    return {
        "event_path": str(event_path),
        "output_path": str(output_path),
        "row_count": row_count,
        "column_count": len(columns),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export ID-sorted event JSONL to XLSX with bounded memory"
    )
    parser.add_argument("event_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--expected-start", type=int, default=0)
    args = parser.parse_args()

    report = export_events_xlsx(
        args.event_path,
        args.output_path,
        expected_count=args.expected_count,
        expected_start=args.expected_start,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
