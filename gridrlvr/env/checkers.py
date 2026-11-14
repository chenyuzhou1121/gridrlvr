"""Check ANM bus voltages, branch loading, and storage state of charge."""

from __future__ import annotations
import math
from typing import Any
from .types import CheckResult, Violation

def _slack_id(simulator: Any) -> Any:
    for (bus_id, bus) in simulator.buses.items():
        if bool(bus.is_slack):
            return bus_id
    return None

class ANMVoltageChecker:
    """Check every bus voltage; summarize extrema over non-slack buses."""
    name = 'voltage'

    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        """Return per-unit voltage violations from simulator buses; ignore base_mva."""
        del base_mva
        slack = _slack_id(system_state)
        violations: list[Violation] = []
        nonslack_values: list[float] = []
        for (bus_id, bus) in system_state.buses.items():
            raw_voltage = abs(bus.v)
            if not math.isfinite(raw_voltage):
                violations.append(Violation(
                    'voltage', 'bus', bus_id, 'v_pu', float('nan'), float(bus.v_min),
                    'p.u.', 'critical', f'Bus {bus_id}: V is non-finite ({raw_voltage!r})',
                ))
                continue
            voltage = float(raw_voltage)
            if bus_id != slack:
                nonslack_values.append(voltage)
            (lower, upper) = (float(bus.v_min), float(bus.v_max))
            if voltage < lower or voltage > upper:
                below = voltage < lower
                margin = lower - voltage if below else voltage - upper
                violations.append(Violation(
                    'voltage', 'bus', bus_id, 'v_pu', round(voltage, 4),
                    round(lower if below else upper, 4), 'p.u.',
                    'critical' if margin > 0.05 else 'violation',
                    f'Bus {bus_id}: V = {voltage:.4f} p.u. outside [{lower:.3f}, {upper:.3f}]',
                ))
        return CheckResult(
            checker_name=self.name,
            passed=not violations,
            summary={
                'min_pu': round(min(nonslack_values), 4) if nonslack_values else None,
                'max_pu': round(max(nonslack_values), 4) if nonslack_values else None,
                'n_violations': len(violations),
            },
            violations=violations,
        )

class ANMBranchLoadingChecker:
    """Check apparent branch power against positive per-unit branch ratings."""
    name = 'branch_loading'

    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        """Return loading violations above 100 percent; skip nonpositive ratings."""
        del base_mva
        violations: list[Violation] = []
        maximum = 0.0
        rated = 0
        for branch in system_state.branches.values():
            rate = float(branch.rate)
            if rate <= 0.0:
                continue
            rated += 1
            raw_flow = abs(branch.s_apparent_max)
            branch_id = f'{branch.f_bus}-{branch.t_bus}'
            if not math.isfinite(raw_flow):
                violations.append(Violation(
                    'branch_loading', 'branch', branch_id, 'loading_pct', float('nan'),
                    100.0, '%', 'critical', f'Branch {branch_id}: apparent power is non-finite',
                ))
                continue
            flow = float(raw_flow)
            loading = flow / rate * 100.0
            maximum = max(maximum, loading)
            if loading > 100.0:
                violations.append(Violation(
                    'branch_loading', 'branch', branch_id, 'loading_pct', round(loading, 1),
                    100.0, '%', 'critical' if loading > 120.0 else 'violation',
                    f'Branch {branch_id}: loading = {loading:.1f}% '
                    f'(S = {flow:.4f} p.u. > rate {rate:.4f} p.u.)',
                ))
        return CheckResult(
            checker_name=self.name,
            passed=not violations,
            summary={
                'max_loading_pct': round(maximum, 1) if rated else None,
                'n_rated_branches': rated,
                'n_violations': len(violations),
            },
            violations=violations,
        )

class ANMStorageSoCChecker:
    """Check storage SoC against each storage device's simulator bounds."""
    name = 'storage_soc'

    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        """Return SoC violations and summaries; unconvertible SoC is reported null."""
        del base_mva
        try:
            from gym_anm.simulator.components import StorageUnit
        except ImportError:
            storage_type: type | tuple[type, ...] = ()
        else:
            storage_type = StorageUnit
        storages = [(idx, dev) for (idx, dev) in system_state.devices.items() if isinstance(dev, storage_type) or bool(getattr(dev, 'is_storage', False))]
        violations: list[Violation] = []
        soc_summary: dict[Any, float | None] = {}
        for (device_id, device) in storages:
            raw_soc = device.soc
            try:
                soc = float(raw_soc)
            except (TypeError, ValueError):
                soc_summary[device_id] = None
                continue
            (lower, upper) = (float(device.soc_min), float(device.soc_max))
            soc_summary[device_id] = round(soc, 4) if math.isfinite(soc) else None
            if not math.isfinite(soc):
                violations.append(Violation(
                    'storage_soc', 'storage', device_id, 'soc', float('nan'), lower,
                    'p.u.', 'critical', f'Storage {device_id}: SoC is non-finite',
                ))
            elif soc < lower or soc > upper:
                below = soc < lower
                violations.append(Violation(
                    'storage_soc', 'storage', device_id, 'soc', round(soc, 4),
                    round(lower if below else upper, 4), 'p.u.', 'critical',
                    f'Storage {device_id}: SoC = {soc:.4f} outside [{lower:.4f}, {upper:.4f}]',
                ))
        return CheckResult(
            checker_name=self.name,
            passed=not violations,
            summary={'n_storage': len(storages), 'soc': soc_summary, 'n_violations': len(violations)},
            violations=violations,
        )
DEFAULT_CHECKERS = (ANMVoltageChecker(), ANMBranchLoadingChecker(), ANMStorageSoCChecker())
