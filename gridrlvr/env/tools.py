"""Validate and apply generator/storage controls and describe their schemas."""

from __future__ import annotations
import math
from typing import Any

class ToolError(ValueError):
    """Report an unknown action, invalid parameter, or device-bound violation."""

def _as_device_id(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ToolError(f'{name} must be an integer, got {value!r}')
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ToolError(f'{name} must be an integer, got {value!r}') from exc
    if isinstance(value, float) and (not value.is_integer()):
        raise ToolError(f'{name} must be an integer, got {value!r}')
    return parsed

def _finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ToolError(f'{name} must be a finite number, got {value!r}')
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ToolError(f'{name} must be a finite number, got {value!r}') from exc
    if not math.isfinite(number):
        raise ToolError(f'{name} must be finite, got {value!r}')
    return number

def _in_range(name: str, value: float, lower: float, upper: float) -> None:
    if value < lower or value > upper:
        raise ToolError(f'{name}={value:.4f} outside device limits [{lower:.4f}, {upper:.4f}]')

def apply_tool(manager: Any, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Update one device's setpoints and return the resulting control fields.

    P is in MW and optional Q is in MVAr. Generator P is capped by available
    potential power; storage P is negative for charging. This does not solve
    the simulator. Invalid names, keys, IDs, or bounds raise ToolError.
    """
    if not isinstance(params, dict):
        raise ToolError(f'params must be an object, got {type(params).__name__}')
    if tool_name == 'set_generator_setpoint':
        unknown = set(params) - {'gen_id', 'p_mw', 'q_mvar'}
        if unknown:
            raise ToolError(f'Unknown parameter(s): {sorted(unknown)}')
        if 'gen_id' not in params or 'p_mw' not in params:
            raise ToolError('gen_id and p_mw are required')
        device_id = _as_device_id('gen_id', params['gen_id'])
        if device_id not in manager._gen_ids:
            raise ToolError(f'Generator {device_id} not found; available={manager._gen_ids}')
        p_mw = _finite_number('p_mw', params['p_mw'])
        device = manager._sim.devices[device_id]
        base = manager.base_mva
        lower = float(device.p_min) * base
        upper = min(float(device.p_max) * base, float(manager._P_pot.get(device_id, float(device.p_max) * base)))
        _in_range(f'p_mw (generator {device_id})', p_mw, lower, upper)
        q_mvar = None
        if params.get('q_mvar') is not None:
            q_mvar = _finite_number('q_mvar', params['q_mvar'])
            _in_range(f'q_mvar (generator {device_id})', q_mvar, float(device.q_min) * base, float(device.q_max) * base)
        manager._P_set[device_id] = p_mw
        if q_mvar is not None:
            manager._Q_set[device_id] = q_mvar
        return {'gen_id': device_id, 'p_set_mw': p_mw, 'q_set_mvar': manager._Q_set.get(device_id, 0.0)}
    if tool_name == 'set_storage_setpoint':
        unknown = set(params) - {'storage_id', 'p_mw', 'q_mvar'}
        if unknown:
            raise ToolError(f'Unknown parameter(s): {sorted(unknown)}')
        if 'storage_id' not in params or 'p_mw' not in params:
            raise ToolError('storage_id and p_mw are required')
        device_id = _as_device_id('storage_id', params['storage_id'])
        if device_id not in manager._des_ids:
            raise ToolError(f'Storage {device_id} not found; available={manager._des_ids}')
        p_mw = _finite_number('p_mw', params['p_mw'])
        device = manager._sim.devices[device_id]
        base = manager.base_mva
        _in_range(f'p_mw (storage {device_id})', p_mw, float(device.p_min) * base, float(device.p_max) * base)
        q_mvar = None
        if params.get('q_mvar') is not None:
            q_mvar = _finite_number('q_mvar', params['q_mvar'])
            _in_range(f'q_mvar (storage {device_id})', q_mvar, float(device.q_min) * base, float(device.q_max) * base)
        manager._P_set[device_id] = p_mw
        if q_mvar is not None:
            manager._Q_set[device_id] = q_mvar
        return {'storage_id': device_id, 'p_set_mw': p_mw, 'q_set_mvar': manager._Q_set.get(device_id, 0.0)}
    raise ToolError(f"Unknown action {tool_name!r}; available=['set_generator_setpoint', 'set_storage_setpoint']")

def build_tool_schemas(manager: Any) -> list[dict[str, Any]]:
    """Return OpenAI function schemas using the manager's device IDs and bounds."""
    base = manager.base_mva
    generator_ids = list(manager._gen_ids)
    storage_ids = list(manager._des_ids)
    generator_description = ' | '.join(
        f'id={idx}: P in [{float(manager._sim.devices[idx].p_min) * base:.1f},'
        f'{float(manager._sim.devices[idx].p_max) * base:.1f}] MW, '
        f'Q in [{float(manager._sim.devices[idx].q_min) * base:.1f},'
        f'{float(manager._sim.devices[idx].q_max) * base:.1f}] MVAr'
        for idx in generator_ids
    )
    storage_description = ' | '.join(
        f'id={idx}: P in [{float(manager._sim.devices[idx].p_min) * base:.1f},'
        f'{float(manager._sim.devices[idx].p_max) * base:.1f}] MW, '
        f'Q in [{float(manager._sim.devices[idx].q_min) * base:.1f},'
        f'{float(manager._sim.devices[idx].q_max) * base:.1f}] MVAr, '
        f'SoC in [{float(manager._sim.devices[idx].soc_min):.2f},'
        f'{float(manager._sim.devices[idx].soc_max):.2f}]'
        for idx in storage_ids
    )
    return [
        {
            'type': 'function',
            'function': {
                'name': 'set_generator_setpoint',
                'description': (
                    "Set one non-slack renewable generator's active power in MW and optional reactive power in MVAr. "
                    'Active power is also capped by the current P_pot shown in the observation.'
                ),
                'parameters': {
                    'type': 'object',
                    'additionalProperties': False,
                    'properties': {
                        'gen_id': {'type': 'integer', 'enum': generator_ids, 'description': generator_description},
                        'p_mw': {'type': 'number', 'description': 'P within [p_min, min(p_max, P_pot)] MW.'},
                        'q_mvar': {'type': 'number', 'description': 'Optional Q within the device MVAr bounds.'},
                    },
                    'required': ['gen_id', 'p_mw'],
                },
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'set_storage_setpoint',
                'description': (
                    'Set storage active power in MW (negative=charge, positive=discharge) '
                    'and optional reactive power in MVAr.'
                ),
                'parameters': {
                    'type': 'object',
                    'additionalProperties': False,
                    'properties': {
                        'storage_id': {'type': 'integer', 'enum': storage_ids, 'description': storage_description},
                        'p_mw': {'type': 'number', 'description': 'Active set-point within device MW bounds.'},
                        'q_mvar': {'type': 'number', 'description': 'Optional Q within the device MVAr bounds.'},
                    },
                    'required': ['storage_id', 'p_mw'],
                },
            },
        },
    ]
ALLOWED_ACTIONS = frozenset({'set_generator_setpoint', 'set_storage_setpoint'})
