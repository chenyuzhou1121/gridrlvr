"""Load frozen or seed-parameterized ANM scenarios and apply their limits."""

from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
_SCENARIO_ID_RE = re.compile('^scenario_s(?P<source_seed>\\d+)_l(?P<load_mul>\\d+(?:p\\d+)?)_g(?P<gen_mul>\\d+(?:p\\d+)?)_soc(?P<soc_pert>native|near_min|near_max)(?:__(?P<limits>[a-z]+\\d+(?:-[a-z]+\\d+)*))?$')
_REPLICATE_SEED_STRIDE = 1000003
LIMIT_KEYS: dict[str, str] = {'r': 'branch_rate_mul', 'gp': 'gen_p_mul', 'sp': 'storage_p_mul', 'v': 'v_band'}
_LIMIT_DEFAULTS: dict[str, float] = {'branch_rate_mul': 1.0, 'gen_p_mul': 1.0, 'storage_p_mul': 1.0, 'v_band': 0.0}
_LIMIT_TOKEN_RE = re.compile('^(?P<key>[a-z]+)(?P<pct>\\d{1,3})$')

def parse_limit_suffix(text: str | None) -> dict[str, float]:
    """Decode r/gp/sp/v integer-percent tokens into nondefault limit values."""
    if not text:
        return {}
    limits: dict[str, float] = {}
    for token in text.split('-'):
        match = _LIMIT_TOKEN_RE.fullmatch(token)
        if match is None or match.group('key') not in LIMIT_KEYS:
            raise ValueError(f'Unknown limit token {token!r} in {text!r}')
        limits[LIMIT_KEYS[match.group('key')]] = int(match.group('pct')) / 100.0
    return normalize_limits(limits)

def normalize_limits(limits: dict[str, Any] | None) -> dict[str, float]:
    """Return recognized nondefault limits as floats, ignoring other keys."""
    if not limits:
        return {}
    out: dict[str, float] = {}
    for (key, default) in _LIMIT_DEFAULTS.items():
        if key in limits and limits[key] is not None:
            value = float(limits[key])
            if value != default:
                out[key] = value
    return out

def encode_limit_suffix(limits: dict[str, Any] | None) -> str:
    """Encode recognized limits as percent tokens; reject fractional percents."""
    norm = normalize_limits(limits)
    tokens = []
    for (short, key) in LIMIT_KEYS.items():
        if key in norm:
            pct = int(round(norm[key] * 100))
            if abs(norm[key] * 100 - pct) > 1e-06:
                raise ValueError(f'{key}={norm[key]} is not representable as an integer percent')
            tokens.append(f'{short}{pct:02d}' if key == 'v_band' else f'{short}{pct}')
    return '-'.join(tokens)

def with_limits(scenario_id: str, limits: dict[str, Any] | None) -> str:
    """Replace an ID's limit suffix with the encoded supplied limits."""
    base = scenario_id.split('__', 1)[0]
    suffix = encode_limit_suffix(limits)
    return f'{base}__{suffix}' if suffix else base

def apply_limits(manager: Any, limits: dict[str, Any] | None) -> None:
    """Mutate branch ratings, non-slack voltage bounds, and device P bounds.

    Multipliers scale existing bounds. v_band sets bounds to 1 +/- v_band;
    omitted/default values leave the corresponding bounds unchanged.
    """
    norm = normalize_limits(limits)
    if not norm:
        return
    sim = manager._sim
    rate_mul = norm.get('branch_rate_mul')
    if rate_mul is not None:
        for branch in sim.branches.values():
            if float(branch.rate) > 0.0:
                branch.rate = float(branch.rate) * rate_mul
    v_band = norm.get('v_band')
    if v_band:
        for bus in sim.buses.values():
            if not bool(getattr(bus, 'is_slack', False)):
                (bus.v_min, bus.v_max) = (1.0 - v_band, 1.0 + v_band)
    gen_mul = norm.get('gen_p_mul')
    if gen_mul is not None:
        for device_id in manager._gen_ids:
            device = sim.devices[device_id]
            device.p_max = float(device.p_max) * gen_mul
    storage_mul = norm.get('storage_p_mul')
    if storage_mul is not None:
        for device_id in manager._des_ids:
            device = sim.devices[device_id]
            device.p_max = float(device.p_max) * storage_mul
            device.p_min = float(device.p_min) * storage_mul

@dataclass
class ANMScenario:
    """Describe a frozen MW snapshot or a seeded load/generation perturbation.

    Power/SoC mappings use integer device IDs. Seeded scenarios require all of
    source_seed, load_mul, gen_mul, and soc_pert; they take precedence over
    explicit P_load/P_pot and initial_soc mappings when applied.
    """
    id: str
    description: str = ''
    difficulty: str = 'unknown'
    P_load: dict[int, float] | None = None
    P_pot: dict[int, float] | None = None
    initial_P_set: dict[int, float] | None = None
    initial_Q_set: dict[int, float] | None = None
    initial_soc: dict[int, float] | None = None
    source_seed: int | None = None
    load_mul: float | None = None
    gen_mul: float | None = None
    soc_pert: str | None = None
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    limits: dict[str, float] | None = None

    @classmethod
    def from_record(cls, record: dict[str, Any], *, split: str | None=None) -> 'ANMScenario':
        """Parse id/scenario_id, mappings, and optional perturbation/limits.

        Missing perturbation values fall back to those encoded in the ID.
        split is a fallback when the record does not specify its own split.
        """
        params = record.get('perturbation')
        if not isinstance(params, dict):
            params = record
        scenario_id = record.get('id', record.get('scenario_id'))
        if not scenario_id:
            raise ValueError('Scenario record is missing id/scenario_id')
        parsed_id = parse_scenario_id(str(scenario_id))

        def numeric_map(name: str) -> dict[int, float] | None:
            value = record.get(name)
            if not isinstance(value, dict):
                return None
            return {int(key): float(item) for (key, item) in value.items()}
        source_seed = params.get('source_seed')
        if source_seed is None and parsed_id is not None:
            source_seed = parsed_id.source_seed
        load_mul = params.get('load_mul')
        if load_mul is None and parsed_id is not None:
            load_mul = parsed_id.load_mul
        gen_mul = params.get('gen_mul')
        if gen_mul is None and parsed_id is not None:
            gen_mul = parsed_id.gen_mul
        soc_pert = params.get('soc_pert')
        if soc_pert is None and parsed_id is not None:
            soc_pert = parsed_id.soc_pert
        limits = normalize_limits(record.get('limits') if isinstance(record.get('limits'), dict) else params.get('limits'))
        if not limits and parsed_id is not None and parsed_id.limits:
            limits = dict(parsed_id.limits)
        return cls(
            id=str(scenario_id),
            description=str(record.get('description', '')),
            difficulty=str(record.get('difficulty', 'unknown')),
            P_load=numeric_map('P_load'),
            P_pot=numeric_map('P_pot'),
            initial_P_set=numeric_map('initial_P_set'),
            initial_Q_set=numeric_map('initial_Q_set'),
            initial_soc=numeric_map('initial_soc'),
            source_seed=int(source_seed) if source_seed is not None else None,
            load_mul=float(load_mul) if load_mul is not None else None,
            gen_mul=float(gen_mul) if gen_mul is not None else None,
            soc_pert=str(soc_pert) if soc_pert is not None else None,
            split=str(record.get('split', split)) if record.get('split', split) else None,
            metadata=dict(record),
            limits=limits or None,
        )

    @property
    def is_perturbed_seed_scenario(self) -> bool:
        """Return whether all four seed/perturbation parameters are present."""
        return self.source_seed is not None and self.load_mul is not None and (self.gen_mul is not None) and (self.soc_pert is not None)

def _decode_number(text: str) -> float:
    return float(text.replace('p', '.'))

def parse_scenario_id(scenario_id: str) -> ANMScenario | None:
    """Decode scenario_sN_lX_gY_socMODE with optional __limit tokens.

    Decimal multipliers use p as the decimal point. Return None when the ID
    does not match; unknown limit tokens raise ValueError.
    """
    match = _SCENARIO_ID_RE.fullmatch(scenario_id)
    if match is None:
        return None
    fields = match.groupdict()
    limits = parse_limit_suffix(fields.get('limits'))
    return ANMScenario(
        id=scenario_id,
        description='Dynamically reconstructed parameterized scenario.',
        difficulty='parameterized',
        source_seed=int(fields['source_seed']),
        load_mul=_decode_number(fields['load_mul']),
        gen_mul=_decode_number(fields['gen_mul']),
        soc_pert=fields['soc_pert'],
        limits=limits or None,
    )

def derive_env_seed(scenario: ANMScenario, rep_seed: int) -> int:
    """Combine source and replicate seeds, using stride 1000003 for perturbations."""
    source_seed = scenario.source_seed if scenario.source_seed is not None else 0
    if scenario.is_perturbed_seed_scenario:
        return int(source_seed) + int(rep_seed) * _REPLICATE_SEED_STRIDE
    return int(source_seed) + int(rep_seed)

def apply_scenario(manager: Any, scenario: ANMScenario) -> bool:
    """Set conditions, SoC, controls, and limits, then return solve convergence.

    Seeded multipliers act on the manager's sampled conditions. Otherwise both
    P_load and P_pot must supply complete MW mappings for the manager.
    """
    if scenario.is_perturbed_seed_scenario:
        P_load = {key: float(value) * float(scenario.load_mul) for (key, value) in manager._P_load.items()}
        P_pot = {key: float(value) * float(scenario.gen_mul) for (key, value) in manager._P_pot.items()}
    elif scenario.P_load is not None and scenario.P_pot is not None:
        P_load = dict(scenario.P_load)
        P_pot = dict(scenario.P_pot)
    else:
        raise ValueError(f'Scenario {scenario.id!r} has neither perturbation parameters nor frozen P_load/P_pot conditions')
    manager.set_conditions(P_load, P_pot, reset_setpoints=True, solve=False)
    if scenario.is_perturbed_seed_scenario:
        _apply_soc_perturbation(manager, str(scenario.soc_pert))
    elif scenario.initial_soc is not None:
        for (device_id, value) in scenario.initial_soc.items():
            if device_id not in manager._des_ids:
                raise ValueError(f'Unknown storage id {device_id} in initial_soc')
            manager._sim.devices[device_id].soc = float(value)
    for (device_id, value) in (scenario.initial_P_set or {}).items():
        manager._P_set[device_id] = float(value)
    for (device_id, value) in (scenario.initial_Q_set or {}).items():
        manager._Q_set[device_id] = float(value)
    apply_limits(manager, scenario.limits)
    return bool(manager.solve())

def _apply_soc_perturbation(manager: Any, perturbation: str) -> None:
    for storage_id in manager._des_ids:
        device = manager._sim.devices[storage_id]
        span = float(device.soc_max) - float(device.soc_min)
        if perturbation == 'native':
            continue
        if perturbation == 'near_min':
            device.soc = float(device.soc_min) + 0.01 * span
        elif perturbation == 'near_max':
            device.soc = float(device.soc_max) - 0.01 * span
        else:
            raise ValueError(f'Unknown SoC perturbation {perturbation!r}')

class ScenarioCatalog:
    """Resolve loaded scenario records or reconstruct parameterized IDs."""

    def __init__(self, scenarios_path: str | Path | Iterable[str | Path] | None=None, scenarios: Iterable[ANMScenario | dict[str, Any]] | None=None) -> None:
        """Load records plus JSON/JSONL files or directories containing them."""
        self._scenarios = {}
        if scenarios is not None:
            for value in scenarios:
                scenario = value if isinstance(value, ANMScenario) else ANMScenario.from_record(value)
                self._insert(scenario)
        paths: list[Path] = []
        if scenarios_path is not None:
            if isinstance(scenarios_path, (str, Path)):
                paths = [Path(scenarios_path)]
            else:
                paths = [Path(value) for value in scenarios_path]
        for path in paths:
            self._load_path(path)

    def _insert(self, scenario: ANMScenario) -> None:
        current = self._scenarios.get(scenario.id)
        if current is None or scenario.is_perturbed_seed_scenario or scenario.P_load:
            self._scenarios[scenario.id] = scenario

    def _load_path(self, path: Path) -> None:
        if path.is_dir():
            for child in sorted(path.glob('*.json')):
                self._load_path(child)
            for child in sorted(path.glob('*.jsonl')):
                self._load_path(child)
            return
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix == '.jsonl':
            for line in path.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    self._collect(json.loads(line), split=None)
            return
        self._collect(json.loads(path.read_text(encoding='utf-8')), split=None)

    def _collect(self, value: Any, split: str | None) -> None:
        if isinstance(value, str):
            parsed = parse_scenario_id(value)
            if parsed is not None:
                parsed.split = split
                self._insert(parsed)
            return
        if isinstance(value, list):
            for item in value:
                self._collect(item, split)
            return
        if not isinstance(value, dict):
            return
        if value.get('id') or value.get('scenario_id'):
            try:
                scenario = ANMScenario.from_record(value, split=split)
                if not scenario.is_perturbed_seed_scenario and scenario.P_load is None:
                    parsed = parse_scenario_id(scenario.id)
                    if parsed is not None:
                        parsed.description = scenario.description
                        parsed.difficulty = scenario.difficulty
                        parsed.split = scenario.split
                        parsed.metadata = scenario.metadata
                        if scenario.limits:
                            parsed.limits = dict(scenario.limits)
                        scenario = parsed
                self._insert(scenario)
            except (TypeError, ValueError):
                pass
            return
        container_split = value.get('split', split)
        for key in ('scenarios', 'records', 'candidates', 'train', 'held_out', 'heldout'):
            if key in value:
                child_split = key if key in {'train', 'held_out', 'heldout'} else container_split
                self._collect(value[key], child_split)

    def resolve(self, scenario_id: str) -> ANMScenario:
        """Return a loaded or ID-encoded scenario; raise KeyError if unknown."""
        scenario = self._scenarios.get(scenario_id)
        if scenario is not None:
            return scenario
        parsed = parse_scenario_id(scenario_id)
        if parsed is not None:
            return parsed
        raise KeyError(f'Unknown ANM scenario {scenario_id!r}; provide a record via scenarios_path')

    def ids(self) -> list[str]:
        """Return sorted IDs of loaded scenarios."""
        return sorted(self._scenarios)

    def all(self) -> list[ANMScenario]:
        """Return loaded scenarios sorted by ID."""
        return [self._scenarios[key] for key in self.ids()]
