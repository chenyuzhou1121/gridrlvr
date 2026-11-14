"""Attach per-constraint and recovery reward dimensions to ANM rollouts."""

from __future__ import annotations
from typing import Any, Mapping, Optional, Sequence
from .anm_agent_loop import ANMAgentLoop
from .reward_fn import _as_bool, _trajectory_from_extra
REWARD_FAMILIES: tuple[str, ...] = ('voltage', 'branch_loading', 'storage_soc')
GDPO_REWARD_KEYS: tuple[str, ...] = tuple((f'{family}_reward' for family in REWARD_FAMILIES)) + ('recovered_reward',)

def _branch_family(serialized_key: str) -> str:
    return str(serialized_key).split('::', 1)[0]

def _severity_from_branches(branches: Optional[Mapping[str, Any]]) -> Optional[dict[str, float]]:
    if not isinstance(branches, Mapping):
        return None
    severity: dict[str, float] = {}
    for (key, value) in branches.items():
        try:
            magnitude = abs(float(value))
        except (TypeError, ValueError):
            continue
        family = _branch_family(str(key))
        severity[family] = severity.get(family, 0.0) + magnitude
    return severity

def _severity_from_violations(violations: Any) -> dict[str, float]:
    severity: dict[str, float] = {}
    if not isinstance(violations, Sequence) or isinstance(violations, (str, bytes, bytearray)):
        return severity
    for violation in violations:
        if not isinstance(violation, Mapping):
            continue
        family = str(violation.get('constraint_type') or '').strip()
        if not family:
            continue
        try:
            magnitude = abs(float(violation.get('value')) - float(violation.get('limit')))
        except (TypeError, ValueError):
            magnitude = 1.0
        severity[family] = severity.get(family, 0.0) + magnitude
    return severity

def _step_verification(step: Mapping[str, Any]) -> Mapping[str, Any]:
    info = step.get('info', {})
    if isinstance(info, Mapping):
        verification = info.get('verification', {})
        if isinstance(verification, Mapping):
            return verification
    return {}

def _initial_severity(steps: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not steps:
        return {}
    verification = _step_verification(steps[0])
    severity = _severity_from_branches(verification.get('baseline_branches'))
    if severity is not None:
        return severity
    info = steps[0].get('info', {})
    if isinstance(info, Mapping):
        return _severity_from_violations(info.get('violations'))
    return {}

def _final_severity(steps: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not steps:
        return {}
    verification = _step_verification(steps[-1])
    severity = _severity_from_branches(verification.get('post_branches'))
    if severity is not None:
        return severity
    info = steps[-1].get('info', {})
    if isinstance(info, Mapping):
        return _severity_from_violations(info.get('violations'))
    return {}

def gdpo_reward_components(steps: Sequence[Mapping[str, Any]], recovered: bool, *, clear_weight: float=1.0, severity_weight: float=1.0) -> dict[str, float]:
    """Return voltage, loading, SoC, and recovered reward dimensions.

    Each family receives a clearing term plus fractional severity reduction
    clipped to [0, 1]. Initial/final magnitudes come from verification branch
    mappings, with violation records as a fallback.
    """
    initial = _initial_severity(steps)
    final = _final_severity(steps)
    components: dict[str, float] = {}
    for family in REWARD_FAMILIES:
        sev_initial = float(initial.get(family, 0.0))
        sev_final = float(final.get(family, 0.0))
        cleared = 1.0 if sev_final <= 0.0 else 0.0
        reduction = 0.0
        if sev_initial > 0.0:
            reduction = min(1.0, max(0.0, (sev_initial - sev_final) / sev_initial))
        components[f'{family}_reward'] = float(clear_weight * cleared + severity_weight * reduction)
    components['recovered_reward'] = 1.0 if _as_bool(recovered) else 0.0
    return components

def compute_score_gdpo(data_source: str, solution_str: str, ground_truth: Any, extra_info: Mapping[str, Any] | None=None) -> dict[str, float | bool]:
    """Return recovery as score/acc plus dimensions read from trajectory metadata."""
    del data_source, solution_str, ground_truth
    (recovered, steps) = _trajectory_from_extra(extra_info or {})
    components = gdpo_reward_components(steps, recovered)
    outcome = components['recovered_reward']
    return {'score': outcome, 'acc': outcome, 'recovered': bool(recovered), **components}

class GDPOANMAgentLoop(ANMAgentLoop):
    """Extend ANM rollouts with reward dimensions for the patched verl trainer."""

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> Any:
        """Run an ANM trajectory and add dimensions to output reward_extra_info."""
        import asyncio
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not None and getattr(self, 'loop', None) is not running:
            self.loop = running
        output = await super().run(sampling_params, **kwargs)
        steps = output.extra_fields.get('anm_trajectory') or []
        recovered = bool(output.extra_fields.get('recovered', False))
        components = gdpo_reward_components(steps, recovered)
        reward_extra_info = output.extra_fields.setdefault('reward_extra_info', {})
        reward_extra_info.update(components)
        output.extra_fields['gdpo_reward_keys'] = list(GDPO_REWARD_KEYS)
        return output
__all__ = ['REWARD_FAMILIES', 'GDPO_REWARD_KEYS', 'gdpo_reward_components', 'compute_score_gdpo', 'GDPOANMAgentLoop']
