"""Load scenario identifiers and metadata for evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_scenarios(path: str | Path, split: str | None = None) -> list[dict[str, Any]]:
    """Read JSON/JSONL records or IDs and return records with string scenario_id.

    JSON objects must contain the requested split, or scenarios when no split
    is requested. Records with a different explicit split are excluded.
    """
    source = Path(path)
    if source.suffix == '.jsonl':
        records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    else:
        records = json.loads(source.read_text())
    if isinstance(records, dict):
        key = split or 'scenarios'
        if key not in records:
            raise ValueError(f'Scenario object must contain {key!r}')
        records = records[key]
    if not isinstance(records, list):
        raise ValueError('Scenarios must be a list of records or identifiers')
    result = []
    for record in records:
        item = dict(record) if isinstance(record, dict) else {'scenario_id': str(record)}
        scenario_id = item.get('scenario_id') or item.get('id')
        if not scenario_id:
            raise ValueError('Every scenario must have an id or scenario_id')
        if split and item.get('split', split) != split:
            continue
        result.append({**item, 'scenario_id': str(scenario_id)})
    return result
