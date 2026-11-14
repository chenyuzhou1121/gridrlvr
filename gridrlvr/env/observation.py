"""Render ANM state, constraint feedback, and the tool-interaction prompt."""

from __future__ import annotations
import json
import math
from typing import Any
from .verifier import check_manager

def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and (not math.isfinite(value)):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for (key, item) in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value

def grid_observation(manager: Any) -> dict[str, Any]:
    """Return grid metrics and checks, replacing non-finite floats with null.

    Voltages and SoC use simulator per-unit values; controls use MW/MVAr and
    branch loading uses percent. recovered means no checker violations.
    """
    simulator = manager.system_state
    checks = check_manager(manager)
    violations = [violation.to_dict() for check in checks for violation in check.violations]
    buses = []
    for (bus_id, bus) in simulator.buses.items():
        voltage = float(abs(bus.v))
        (lower, upper) = (float(bus.v_min), float(bus.v_max))
        margin = 0.2 * (upper - lower)
        buses.append({
            'bus_id': bus_id,
            'v_pu': round(voltage, 4) if math.isfinite(voltage) else None,
            'v_min': lower,
            'v_max': upper,
            'stressed': not math.isfinite(voltage) or voltage < lower + margin or voltage > upper - margin,
        })
    branches = []
    for branch in simulator.branches.values():
        rate = float(branch.rate)
        flow = float(abs(branch.s_apparent_max))
        loading = flow / rate * 100.0 if rate > 0.0 and math.isfinite(flow) else None
        branches.append({
            'branch_id': f'{branch.f_bus}-{branch.t_bus}',
            'loading_pct': round(loading, 1) if loading is not None else None,
            'rate_pu': round(rate, 4),
            'stressed': loading is None or loading > 80.0,
        })
    base = manager.base_mva
    generators = []
    for device_id in manager._gen_ids:
        device = simulator.devices[device_id]
        generators.append({
            'gen_id': device_id,
            'P_pot_mw': round(float(manager._P_pot.get(device_id, 0.0)), 4),
            'p_set_mw': round(float(manager._P_set.get(device_id, 0.0)), 4),
            'q_set_mvar': round(float(manager._Q_set.get(device_id, 0.0)), 4),
            'p_range_mw': [float(device.p_min) * base, float(device.p_max) * base],
            'q_range_mvar': [float(device.q_min) * base, float(device.q_max) * base],
        })
    storage = []
    for device_id in manager._des_ids:
        device = simulator.devices[device_id]
        soc = None if device.soc is None else float(device.soc)
        storage.append({
            'storage_id': device_id,
            'soc': round(soc, 4) if soc is not None and math.isfinite(soc) else None,
            'soc_range': [float(device.soc_min), float(device.soc_max)],
            'p_set_mw': round(float(manager._P_set.get(device_id, 0.0)), 4),
            'q_set_mvar': round(float(manager._Q_set.get(device_id, 0.0)), 4),
            'p_range_mw': [float(device.p_min) * base, float(device.p_max) * base],
            'q_range_mvar': [float(device.q_min) * base, float(device.q_max) * base],
        })
    return _json_safe({
        'buses': buses,
        'branches': branches,
        'generators': generators,
        'storage': storage,
        'checks': {check.checker_name: check.summary for check in checks},
        'violations': violations,
        'violation_count': len(violations),
        'recovered': not violations,
        'native_penalty': float(getattr(manager, 'last_penalty', 0.0)),
    })

def format_observation(manager: Any, *, scenario_id: str, step_index: int, verifier: dict[str, Any] | None=None, limits: dict[str, Any] | None=None) -> str:
    """Return compact JSON for a grid state, with optional limits and feedback."""
    payload: dict[str, Any] = {'scenario_id': scenario_id, 'step': step_index, 'grid': grid_observation(manager)}
    if limits:
        payload['limits'] = dict(limits)
    if verifier is not None:
        payload['verifier'] = verifier
    return json.dumps(_json_safe(payload), ensure_ascii=False, separators=(',', ':'))

def build_system_prompt(manager: Any) -> str:
    """Return the gated tool protocol with this manager's device IDs and bounds."""
    simulator = manager.system_state
    voltage_windows = sorted({(float(bus.v_min), float(bus.v_max)) for bus in simulator.buses.values()})
    return (
        'You are an active network management operator. Restore this frozen distribution-grid snapshot '
        'while respecting every bus-voltage, branch-rating, and storage-SoC limit. '
        'Each response must contain exactly one native tool call. '
        "Generator P is capped by the observation's P_pot; storage P<0 charges and P>0 discharges. "
        'The verifier first executes your proposal on an isolated shadow simulator. '
        'PASS means fully recovered; SAFE_PROGRESS means the action is applied but another action is needed; '
        'FAIL/ERROR means it is rejected and the real grid is unchanged. '
        'Read the structured verifier feedback before retrying. '
        f'Buses={len(simulator.buses)}, generators={list(manager._gen_ids)}, '
        f'storage={list(manager._des_ids)}, baseMVA={manager.base_mva}, voltage_windows={voltage_windows}.'
    )
