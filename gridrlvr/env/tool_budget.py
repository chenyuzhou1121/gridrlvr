"""Compact RestoreBench diagnostic payloads using estimated token budgets."""

from __future__ import annotations
import copy
import json
import os
import re
from collections import deque
from typing import Any
GET_GRID_TOPOLOGY_TOOL_NAME = 'get_grid_topology'
TOOL_TOKEN_BUDGET_ENV = 'GRIDRLVR_RESTOREBENCH_TOOL_TOKEN_BUDGET'
TOPOLOGY_FORMAT_ENV = 'GRIDRLVR_RESTOREBENCH_TOPOLOGY_FORMAT'
DEFAULT_TOOL_TOKEN_BUDGET = 2048
ESTIMATOR_CHARS_PER_WORDPIECE = 5
TOOL_MESSAGE_TEMPLATE_OVERHEAD_TOKENS = 16
TOPOLOGY_FORMAT_TABLE = 'table'
TOPOLOGY_FORMAT_RECORDS = 'records'
# Each section defines its ID field, connected buses, and overview budget share.
_TOPOLOGY_SECTIONS: tuple[tuple[str, str, tuple[str, ...], float], ...] = (
    ('ext_grids', 'ext_grid_id', ('bus',), 0.04),
    ('shunts', 'shunt_id', ('bus',), 0.2),
    ('trafos', 'trafo_id', ('hv_bus', 'lv_bus'), 0.2),
    ('gens', 'gen_id', ('bus',), 0.36),
    ('buses', 'bus_id', ('bus_id',), 0.08),
    ('lines', 'line_id', ('from_bus', 'to_bus'), 0.06),
    ('loads', 'load_id', ('bus',), 0.06),
)
_TOPOLOGY_SECTION_KEYS = tuple((spec[0] for spec in _TOPOLOGY_SECTIONS))
_UNREACHABLE_HOPS = 1 << 30
_MAX_STRING_CHARS = 4000

def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or '').strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default

def tool_token_budget() -> int:
    """Return the configured estimated-token budget; nonpositive disables it."""
    return _env_int(TOOL_TOKEN_BUDGET_ENV, DEFAULT_TOOL_TOKEN_BUDGET)

def tool_budget() -> int:
    """Return the payload budget after reserving tool-message template overhead."""
    token_budget = tool_token_budget()
    if token_budget <= 0:
        return 0
    return max(1, token_budget - TOOL_MESSAGE_TEMPLATE_OVERHEAD_TOKENS)
_WORD_OR_DIGIT_RUN = re.compile('[A-Za-z_]+|[0-9]+')

def estimate_tokens(text: str) -> int:
    """Estimate token cost from word chunks, counting other characters singly."""
    total = len(text)
    for match in _WORD_OR_DIGIT_RUN.finditer(text):
        span = match.end() - match.start()
        if not text[match.start()].isdigit():
            # Ceiling division estimates the word's number of wordpieces.
            pieces = (span + ESTIMATOR_CHARS_PER_WORDPIECE - 1) // ESTIMATOR_CHARS_PER_WORDPIECE
            total -= span - pieces
    return total

def topology_format() -> str:
    """Return records only when explicitly configured, otherwise table."""
    raw = (os.environ.get(TOPOLOGY_FORMAT_ENV) or '').strip().lower()
    return TOPOLOGY_FORMAT_RECORDS if raw == TOPOLOGY_FORMAT_RECORDS else TOPOLOGY_FORMAT_TABLE

def _json_text(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True)

def _cost(payload: Any) -> int:
    return estimate_tokens(_json_text(payload))

def _focus_buses(payload: dict[str, Any], diagnostics: Any) -> tuple[list[int], dict[str, Any]]:
    anchors: list[int] = []
    note: dict[str, Any] = {}

    def add(value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            return
        if value not in anchors:
            anchors.append(value)
    if diagnostics is not None:
        worst = getattr(diagnostics, 'worst_bus', None)
        lowest = getattr(diagnostics, 'lowest_vm_bus', None)
        saturated = list(getattr(diagnostics, 'gens_at_q_limit', None) or ())
        add(worst)
        add(lowest)
        note['worst_bus'] = worst
        note['lowest_vm_bus'] = lowest
        note['gens_at_q_limit'] = saturated
        gen_bus = {int(g['gen_id']): g.get('bus') for g in payload.get('gens') or () if isinstance(g, dict) and isinstance(g.get('gen_id'), int)}
        for gen_id in saturated:
            add(gen_bus.get(gen_id))
    out_of_service: dict[str, list[int]] = {}
    for (key, id_field, bus_fields, _) in _TOPOLOGY_SECTIONS:
        ids = [int(item[id_field]) for item in payload.get(key) or () if isinstance(item, dict) and (not item.get('in_service', True)) and isinstance(item.get(id_field), int)]
        if ids:
            out_of_service[key] = ids
            for item in payload.get(key) or ():
                if isinstance(item, dict) and (not item.get('in_service', True)):
                    for field_name in bus_fields:
                        add(item.get(field_name))
    if out_of_service:
        note['out_of_service'] = out_of_service
    if not anchors:
        add(payload.get('slack_bus'))
    return (anchors, note)

def _bus_hops(payload: dict[str, Any], anchors: list[int]) -> dict[int, int]:
    """Compute multi-source shortest hops over in-service lines/transformers."""
    if not anchors:
        return {}
    adjacency: dict[int, set[int]] = {}

    def link(a: Any, b: Any) -> None:
        if not isinstance(a, int) or not isinstance(b, int) or isinstance(a, bool) or isinstance(b, bool):
            return
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    for item in payload.get('lines') or ():
        if isinstance(item, dict) and item.get('in_service', True):
            link(item.get('from_bus'), item.get('to_bus'))
    for item in payload.get('trafos') or ():
        if isinstance(item, dict) and item.get('in_service', True):
            link(item.get('hv_bus'), item.get('lv_bus'))
    hops = {bus: 0 for bus in anchors}
    frontier = deque(anchors)
    while frontier:
        bus = frontier.popleft()
        for neighbour in adjacency.get(bus, ()):
            if neighbour not in hops:
                hops[neighbour] = hops[bus] + 1
                frontier.append(neighbour)
    return hops

def _item_hops(item: dict[str, Any], bus_fields: tuple[str, ...], hops: dict[int, int]) -> int | None:
    values = [hops.get(item[field]) for field in bus_fields if isinstance(item.get(field), int) and (not isinstance(item.get(field), bool))]
    values = [v for v in values if v is not None]
    return min(values) if values else None

def _rank_section(items: list[dict[str, Any]], id_field: str, bus_fields: tuple[str, ...], hops: dict[int, int]) -> list[tuple[dict[str, Any], int | None]]:
    decorated = [(item, _item_hops(item, bus_fields, hops)) for item in items if isinstance(item, dict)]
    decorated.sort(key=lambda pair: (_UNREACHABLE_HOPS if pair[1] is None else pair[1], pair[0].get(id_field) if isinstance(pair[0].get(id_field), int) else _UNREACHABLE_HOPS))
    return decorated

def _section_columns(items: list[dict[str, Any]], id_field: str) -> list[str]:
    keys: set[str] = set()
    for item in items:
        keys.update(item.keys())
    keys.discard('in_service')
    keys.discard(id_field)
    return [id_field, *sorted(keys), 'hops']

def _render_row(item: dict[str, Any], columns: list[str], hop: int | None) -> list[Any]:
    return [hop if column == 'hops' else item.get(column) for column in columns]

def _render_record(item: dict[str, Any], hop: int | None) -> dict[str, Any]:
    record = {key: value for (key, value) in item.items() if key != 'in_service'}
    record['hops'] = hop
    return record

def shrink_topology_payload(payload: dict[str, Any], *, diagnostics: Any=None, token_budget: int, params: dict[str, Any] | None=None, fmt: str | None=None) -> dict[str, Any]:
    """Return a ranked topology overview or section page under an estimated cap.

    Diagnostic and out-of-service buses anchor distance ranking. params may
    contain section, offset, and limit. fmt selects table rows or records;
    a nonpositive token_budget keeps all requested rows.
    """
    fmt = fmt or topology_format()
    params = params or {}
    (anchors, focus_note) = _focus_buses(payload, diagnostics)
    hops = _bus_hops(payload, anchors)
    scalars = {key: value for (key, value) in payload.items() if key not in _TOPOLOGY_SECTION_KEYS}
    focus = dict(focus_note)
    focus['anchor_buses'] = anchors
    ranked: dict[str, list[tuple[dict[str, Any], int | None]]] = {}
    columns: dict[str, list[str]] = {}
    for (key, id_field, bus_fields, _) in _TOPOLOGY_SECTIONS:
        items = [item for item in payload.get(key) or () if isinstance(item, dict)]
        ranked[key] = _rank_section(items, id_field, bus_fields, hops)
        columns[key] = _section_columns(items, id_field) if items else [id_field, 'hops']
    requested_section = params.get('section')
    if requested_section is not None:
        requested_section = str(requested_section)
        if requested_section not in _TOPOLOGY_SECTION_KEYS:
            return {'error': f'unknown section {requested_section!r}', 'valid_sections': list(_TOPOLOGY_SECTION_KEYS)}
        return _paged_section(requested_section, ranked[requested_section], columns[requested_section], scalars=scalars, focus=focus, params=params, token_budget=token_budget, fmt=fmt)
    rendered: dict[str, list[Any]] = {}
    costs: dict[str, list[int]] = {}
    for key in _TOPOLOGY_SECTION_KEYS:
        rows = [_render_row(item, columns[key], hop) if fmt == TOPOLOGY_FORMAT_TABLE else _render_record(item, hop) for (item, hop) in ranked[key]]
        rendered[key] = rows
        costs[key] = [_cost(row) + 2 for row in rows]
    counts = {key: 0 for key in _TOPOLOGY_SECTION_KEYS}
    if token_budget <= 0:
        counts = {key: len(rendered[key]) for key in _TOPOLOGY_SECTION_KEYS}
    else:
        # Seed each section, distribute weighted quotas, then spend the remainder.
        for (key, _, _, _) in _TOPOLOGY_SECTIONS:
            if not rendered[key]:
                continue
            counts[key] += 1
            if _cost(_assemble_topology(scalars, focus, columns, rendered, counts, fmt)) > token_budget:
                counts[key] -= 1
        available = token_budget - _cost(_assemble_topology(scalars, focus, columns, rendered, counts, fmt)) - 4 * len(_TOPOLOGY_SECTION_KEYS)
        spent = 0
        for (key, _, _, weight) in _TOPOLOGY_SECTIONS:
            quota = int(max(0, available) * weight)
            used = 0
            for cost in costs[key][counts[key]:]:
                if used + cost > quota:
                    break
                used += cost
                counts[key] += 1
            spent += used
        leftover = max(0, available - spent)
        for (key, _, _, _) in _TOPOLOGY_SECTIONS:
            for cost in costs[key][counts[key]:]:
                if cost > leftover:
                    break
                leftover -= cost
                counts[key] += 1
    result = _assemble_topology(scalars, focus, columns, rendered, counts, fmt)
    if token_budget > 0:
        for key in reversed(_TOPOLOGY_SECTION_KEYS):
            while counts[key] > 0 and _cost(result) > token_budget:
                counts[key] -= 1
                result = _assemble_topology(scalars, focus, columns, rendered, counts, fmt)
            if _cost(result) <= token_budget:
                break
    return _degrade_to_budget(result, token_budget)

def _degrade_to_budget(result: dict[str, Any], token_budget: int) -> dict[str, Any]:
    if token_budget <= 0 or _cost(result) <= token_budget:
        return result
    note = 'no element rows fit the per-tool budget; ask for one section with {"section": ..., "limit": ...} or raise GRIDRLVR_RESTOREBENCH_TOOL_TOKEN_BUDGET'
    totals = {key: value.get('total') for (key, value) in result.items() if isinstance(value, dict) and 'total' in value}
    collapsed: dict[str, Any] = {key: value for (key, value) in result.items() if not (isinstance(value, dict) and 'total' in value) and key not in ('pagination', 'columns', 'rows', 'items')}
    collapsed['truncated'] = True
    collapsed['note'] = note
    if totals:
        collapsed['section_totals'] = totals
    if _cost(collapsed) <= token_budget:
        return collapsed
    return {'truncated': True, 'returned': 0, 'section_totals': totals, 'note': note}

def _assemble_topology(scalars: dict[str, Any], focus: dict[str, Any], columns: dict[str, list[str]], rendered: dict[str, list[Any]], counts: dict[str, int], fmt: str, keep_empty_columns: bool=False) -> dict[str, Any]:
    payload: dict[str, Any] = dict(scalars)
    payload['focus'] = focus
    truncated_sections: dict[str, Any] = {}
    for key in _TOPOLOGY_SECTION_KEYS:
        total = len(rendered[key])
        kept = counts[key]
        section: dict[str, Any] = {'returned': kept, 'total': total}
        if fmt == TOPOLOGY_FORMAT_TABLE:
            if kept or keep_empty_columns or (not total):
                section['columns'] = columns[key]
            section['rows'] = rendered[key][:kept]
        else:
            section['items'] = rendered[key][:kept]
        if kept < total:
            section['next_offset'] = kept
            truncated_sections[key] = {'returned': kept, 'total': total, 'next_offset': kept}
        payload[key] = section
    payload['truncated'] = bool(truncated_sections)
    return payload

def _paged_section(key: str, ranked: list[tuple[dict[str, Any], int | None]], columns: list[str], *, scalars: dict[str, Any], focus: dict[str, Any], params: dict[str, Any], token_budget: int, fmt: str) -> dict[str, Any]:
    total = len(ranked)
    try:
        offset = max(0, int(params.get('offset') or 0))
    except (TypeError, ValueError):
        offset = 0
    try:
        limit = int(params['limit']) if params.get('limit') is not None else total
    except (TypeError, ValueError):
        limit = total
    limit = max(0, limit)
    window = ranked[offset:offset + limit]
    rows = [_render_row(item, columns, hop) if fmt == TOPOLOGY_FORMAT_TABLE else _render_record(item, hop) for (item, hop) in window]

    def build(count: int) -> dict[str, Any]:
        payload: dict[str, Any] = {'section': key, 'offset': offset, 'returned': count, 'total': total, 'focus': focus}
        for scalar_key in ('n_buses', 'n_lines', 'n_trafos', 'n_gens', 'n_ext_grids', 'n_loads', 'n_shunts', 'slack_bus'):
            if scalar_key in scalars:
                payload[scalar_key] = scalars[scalar_key]
        if fmt == TOPOLOGY_FORMAT_TABLE:
            payload['columns'] = columns
            payload['rows'] = rows[:count]
        else:
            payload['items'] = rows[:count]
        if offset + count < total:
            payload['next_offset'] = offset + count
        payload['truncated'] = bool(offset > 0 or offset + count < total)
        return payload
    count = len(rows)
    result = build(count)
    if token_budget > 0 and _cost(result) > token_budget:
        # Find the longest prefix fitting the page budget without reordering rows.
        (low, high) = (0, count)
        while low < high:
            mid = (low + high + 1) // 2
            if _cost(build(mid)) <= token_budget:
                low = mid
            else:
                high = mid - 1
        result = build(low)
    return _degrade_to_budget(result, token_budget)

def fit_payload_to_budget(payload: Any, token_budget: int) -> Any:
    """Copy oversized payloads, clip long strings, and trim costly nested lists.

    Dictionary outputs receive truncation metadata. The budget is estimated;
    scalar fields and metadata are retained even if they exceed a tiny budget.
    """
    if token_budget <= 0 or _cost(payload) <= token_budget:
        return payload
    work = copy.deepcopy(payload)
    work = _clip_strings(work)
    dropped: dict[str, int] = {}
    for _ in range(200):
        if _cost(work) <= token_budget:
            break
        target = _longest_list(work, ())
        if target is None:
            break
        (path, container) = target
        keep = len(container) * 3 // 4 if len(container) > 4 else 0
        del container[keep:]
        dropped['/'.join((str(p) for p in path))] = keep
    if isinstance(work, dict):
        work['truncated'] = True
        work['truncation_note'] = f'trimmed to fit the {token_budget}-token per-tool budget; kept item counts: {dropped}'
    return work

def _clip_strings(node: Any) -> Any:
    if isinstance(node, str):
        return node if len(node) <= _MAX_STRING_CHARS else node[:_MAX_STRING_CHARS] + '...[clipped]'
    if isinstance(node, dict):
        return {key: _clip_strings(value) for (key, value) in node.items()}
    if isinstance(node, list):
        return [_clip_strings(value) for value in node]
    return node

def _longest_list(node: Any, path: tuple[Any, ...]) -> tuple[tuple[Any, ...], list[Any]] | None:
    found = _longest_list_sized(node, path)
    return None if found is None else (found[1], found[2])

def _longest_list_sized(node: Any, path: tuple[Any, ...]) -> tuple[int, tuple[Any, ...], list[Any]] | None:
    best: tuple[int, tuple[Any, ...], list[Any]] | None = None
    if isinstance(node, list) and node:
        best = (_cost(node), path, node)
    if isinstance(node, dict):
        children: list[tuple[Any, Any]] = list(node.items())
    elif isinstance(node, list):
        children = list(enumerate(node))
    else:
        children = []
    for (key, value) in children:
        candidate = _longest_list_sized(value, (*path, key))
        if candidate is not None and (best is None or candidate[0] > best[0]):
            best = candidate
    return best

def apply_tool_budget(tool_name: str, payload: Any, *, diagnostics: Any=None, params: dict[str, Any] | None=None) -> Any:
    """Apply configured topology ranking/paging and generic payload trimming."""
    params = params or {}
    token_budget = tool_budget()
    if token_budget <= 0 and (not (tool_name == GET_GRID_TOPOLOGY_TOOL_NAME and params.get('section'))):
        return payload
    if tool_name == GET_GRID_TOPOLOGY_TOOL_NAME and isinstance(payload, dict):
        payload = shrink_topology_payload(payload, diagnostics=diagnostics, token_budget=token_budget, params=params)
    return fit_payload_to_budget(payload, token_budget)
_TOPOLOGY_PARAMS_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'section': {
            'type': 'string',
            'enum': list(_TOPOLOGY_SECTION_KEYS),
            'description': 'return only this section, paged; omit for a capped overview of every section',
        },
        'offset': {
            'type': 'integer',
            'minimum': 0,
            'description': 'index of the first element to return within the ranked section (use next_offset)',
        },
        'limit': {
            'type': 'integer',
            'minimum': 1,
            'description': 'how many elements to return; the per-call budget may still return fewer',
        },
    },
    'additionalProperties': False,
}
_TOPOLOGY_DESCRIPTION_SUFFIX = (
    ' The return is capped to fit the response budget and is shaped like this. '
    'Each of buses/lines/trafos/gens/ext_grids/loads/shunts is '
    '{"columns": [...], "rows": [[...]], "returned": n, "total": N}, '
    'where rows[i][j] is the value of columns[j]; '
    'every element is in service unless its id is listed under focus.out_of_service. '
    'Rows are ordered by "hops", the shortest in-service path from the element to the collapse '
    '(focus.worst_bus, focus.lowest_vm_bus and the buses of focus.gens_at_q_limit), '
    'so the nearest and most relevant elements come first. '
    'A section with returned < total also carries "next_offset": call get_grid_topology again with '
    '{"section": "<name>", "offset": <next_offset>} to read the next page of that one section.'
)

def augment_tool_schemas(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add topology pagination and matching format help when budgeting is enabled."""
    if tool_token_budget() <= 0:
        return schemas
    augmented: list[dict[str, Any]] = []
    for schema in schemas:
        function = schema.get('function') or {}
        if function.get('name') == GET_GRID_TOPOLOGY_TOOL_NAME:
            schema = copy.deepcopy(schema)
            schema['function']['parameters'] = copy.deepcopy(_TOPOLOGY_PARAMS_SCHEMA)
            description = _TOPOLOGY_DESCRIPTION_SUFFIX
            if topology_format() == TOPOLOGY_FORMAT_RECORDS:
                description = description.replace(
                    '{"columns": [...], "rows": [[...]], "returned": n, "total": N}, '
                    'where rows[i][j] is the value of columns[j]',
                    '{"items": [{...}], "returned": n, "total": N}, '
                    'where each item is an element record',
                )
            schema['function']['description'] = (function.get('description') or '') + description
        augmented.append(schema)
    return augmented
