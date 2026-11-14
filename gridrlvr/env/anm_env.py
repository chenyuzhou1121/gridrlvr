"""Expose tool-based restoration episodes with shadow verification and rollback."""

from __future__ import annotations
import copy
import inspect
from pathlib import Path
from typing import Any, Callable, Iterable
from ..reward import RewardConfig, reward_breakdown
from .manager import GymANMManager
from .mpc import get_mpc_action, run_mpc
from .observation import build_system_prompt, format_observation, grid_observation
from .scenarios import ANMScenario, ScenarioCatalog, apply_scenario, derive_env_seed
from .tools import ToolError, apply_tool, build_tool_schemas
from .types import Verdict, VerificationResult
from .verifier import ANMVerifier, check_manager
ManagerFactory = Callable[..., Any]

class ANMEnv:
    """Restore fixed ANM load/generation conditions through one control per step.

    The gate admits PASS and SAFE_PROGRESS proposals. With gating disabled,
    non-ERROR proposals are attempted on the live grid. Failed live solves
    restore the simulator and setpoints from a snapshot.
    """

    def __init__(
        self,
        scenarios_path: str | Path | Iterable[str | Path] | None = None,
        *,
        scenarios: Iterable[ANMScenario | dict[str, Any]] | None = None,
        max_steps: int = 8,
        gated: bool = True,
        gating_policy: str = 'progress_mag',
        env_name: str = 'ANM6Easy',
        manager_factory: ManagerFactory | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        """Load scenario sources and configure the proposal budget and rewards.

        manager_factory receives seed and env_name when its signature permits.
        reset constructs the simulator; construction alone does not start it.
        """
        if int(max_steps) <= 0:
            raise ValueError('max_steps must be positive')
        self.max_steps = int(max_steps)
        self.gated = bool(gated)
        self.gating_policy = gating_policy
        self.env_name = env_name
        self._manager_factory = manager_factory or GymANMManager
        self._catalog = ScenarioCatalog(scenarios_path, scenarios)
        self.reward_config = reward_config or RewardConfig()
        self.manager: Any | None = None
        self.current_scenario: ANMScenario | None = None
        self.current_scenario_id: str | None = None
        self.rep_seed: int | None = None
        self.env_seed: int | None = None
        self.current_step = 0
        self.recovered = False
        self.done = False
        self.tool_schemas: list[dict[str, Any]] = []
        self.system_prompt = ''
        self.last_verification: VerificationResult | None = None
        self.last_observation = ''
        self._verifier: ANMVerifier | None = None

    @property
    def scenario_ids(self) -> list[str]:
        """Return sorted IDs explicitly loaded into the catalog."""
        return self._catalog.ids()

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Return the tool schemas populated by reset."""
        return self.tool_schemas

    def _make_manager(self, env_seed: int) -> Any:
        factory = self._manager_factory
        try:
            parameters = inspect.signature(factory).parameters.values()
        except (TypeError, ValueError):
            parameters = ()
        names = {parameter.name for parameter in parameters}
        accepts_kwargs = any((parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters))
        kwargs: dict[str, Any] = {}
        if accepts_kwargs or 'seed' in names:
            kwargs['seed'] = env_seed
        if accepts_kwargs or 'env_name' in names:
            kwargs['env_name'] = self.env_name
        return factory(**kwargs)

    def reset(self, scenario_id: str, seed: int | None=0) -> str:
        """Construct the seeded scenario and return its initial JSON observation.

        seed selects a replicate and combines with the scenario source seed.
        An initially constraint-free state is immediately marked done.
        """
        scenario = self._catalog.resolve(str(scenario_id))
        rep_seed = 0 if seed is None else int(seed)
        env_seed = derive_env_seed(scenario, rep_seed)
        manager = self._make_manager(env_seed)
        if not apply_scenario(manager, scenario):
            raise RuntimeError(f'Initial solve did not converge for {scenario_id!r} rep_seed={seed}')
        self.manager = manager
        self.current_scenario = scenario
        self.current_scenario_id = scenario.id
        self.rep_seed = rep_seed
        self.env_seed = env_seed
        self.current_step = 0
        self.last_verification = None
        self.tool_schemas = build_tool_schemas(manager)
        self.system_prompt = build_system_prompt(manager)
        self._verifier = ANMVerifier(manager, gating_policy=self.gating_policy)
        self.recovered = bool(grid_observation(manager)['recovered'])
        self.done = self.recovered
        self.last_observation = format_observation(manager, scenario_id=scenario.id, step_index=0, limits=scenario.limits)
        return self.last_observation

    def step(self, tool_name: str, params: dict[str, Any] | None=None) -> tuple[str, Verdict, dict[str, Any], bool]:
        """Return observation, verdict, reward/info fields, and done for a tool.

        Every proposal consumes one step, including rejection. Recovery or the
        step budget ends the episode. Call reset before the first proposal.
        """
        if self.manager is None or self._verifier is None or self.current_scenario_id is None:
            raise RuntimeError('reset() must be called before step()')
        action = {'tool_name': str(tool_name), 'params': {} if params is None else params}
        if self.done:
            result = VerificationResult(verdict=Verdict.ERROR, action=action, fail_reason='Episode is already finished; call reset()')
            info = self._build_info(result, applied=False, penalty_before=self.manager.last_penalty)
            observation = format_observation(self.manager, scenario_id=self.current_scenario_id, step_index=self.current_step, verifier=self._feedback(result, applied=False), limits=self.current_scenario.limits if self.current_scenario else None)
            self.last_observation = observation
            return (observation, result.verdict, info, True)
        penalty_before = float(getattr(self.manager, 'last_penalty', 0.0))
        result = self._verifier.verify(action)
        self.current_step += 1
        admitted = result.verdict in {Verdict.PASS, Verdict.SAFE_PROGRESS}
        should_apply = admitted if self.gated else result.verdict != Verdict.ERROR
        applied = False
        if should_apply:
            # Keep the live state recoverable even if a shadow-approved solve fails.
            live_snapshot = {
                'sim': copy.deepcopy(self.manager._sim),
                'P_set': dict(self.manager._P_set),
                'Q_set': dict(self.manager._Q_set),
                'last_reward': float(getattr(self.manager, '_last_reward', 0.0)),
                'last_penalty': float(getattr(self.manager, '_last_penalty', 0.0)),
            }
            try:
                apply_tool(self.manager, action['tool_name'], action['params'])
                live_converged = bool(self.manager.solve())
            except ToolError as exc:
                self._restore_live_snapshot(live_snapshot)
                result = VerificationResult(verdict=Verdict.ERROR, action=action, fail_reason=f'Live action application failed after verification: {exc}', baseline_branches=result.baseline_branches)
            except Exception as exc:
                self._restore_live_snapshot(live_snapshot)
                result = VerificationResult(verdict=Verdict.FAIL, action=action, solver_converged=False, fail_reason=f'Live solve raised after shadow admission: {type(exc).__name__}: {exc}', baseline_branches=result.baseline_branches)
            else:
                if live_converged:
                    applied = True
                else:
                    self._restore_live_snapshot(live_snapshot)
                    result = VerificationResult(verdict=Verdict.FAIL, action=action, solver_converged=False, fail_reason='Live solve did not converge after shadow admission; action rolled back', baseline_branches=result.baseline_branches)
        self.last_verification = result
        checks = check_manager(self.manager)
        self.recovered = all((check.passed for check in checks))
        self.done = self.recovered or self.current_step >= self.max_steps
        info = self._build_info(result, applied=applied, penalty_before=penalty_before)
        observation = format_observation(self.manager, scenario_id=self.current_scenario_id, step_index=self.current_step, verifier=self._feedback(result, applied=applied), limits=self.current_scenario.limits if self.current_scenario else None)
        self.last_observation = observation
        return (observation, result.verdict, info, self.done)

    def _restore_live_snapshot(self, snapshot: dict[str, Any]) -> None:
        assert self.manager is not None
        self.manager._sim = snapshot['sim']
        self.manager._P_set = snapshot['P_set']
        self.manager._Q_set = snapshot['Q_set']
        self.manager._last_reward = snapshot['last_reward']
        self.manager._last_penalty = snapshot['last_penalty']
        native_env = getattr(self.manager, '_env', None)
        if native_env is not None:
            native_env.simulator = self.manager._sim

    def _feedback(self, result: VerificationResult, *, applied: bool) -> dict[str, Any]:
        return {'verdict': result.verdict.value, 'applied': applied, 'reason': result.fail_reason, 'solver_converged': result.solver_converged, 'proposed_post_violations': [violation.to_dict() for violation in result.violations]}

    def _build_info(self, result: VerificationResult, *, applied: bool, penalty_before: float) -> dict[str, Any]:
        assert self.manager is not None
        live_checks = check_manager(self.manager)
        live_violations = [violation.to_dict() for check in live_checks for violation in check.violations]
        terminal = self.recovered or self.current_step >= self.max_steps
        rewards = reward_breakdown(result, recovered=self.recovered, terminal=terminal, config=self.reward_config)
        return {
            'scenario_id': self.current_scenario_id,
            'difficulty': self.current_scenario.difficulty if self.current_scenario else 'unknown',
            'rep_seed': self.rep_seed,
            'env_seed': self.env_seed,
            'step_index': self.current_step,
            'max_steps': self.max_steps,
            'gated': self.gated,
            'gating_policy': self.gating_policy,
            'applied': applied,
            'verdict': result.verdict.value,
            'recovered': self.recovered,
            'violation_count': len(live_violations),
            'violations': live_violations,
            'penalty_before': float(penalty_before),
            'penalty_after': float(getattr(self.manager, 'last_penalty', 0.0)),
            'verification': result.to_dict(),
            **rewards,
            'reward': rewards['total_reward'],
            'step_reward': rewards['total_reward'],
        }

    def mpc_action(self, *, planning_steps: int=8, safety_margin: float=0.9) -> dict[str, Any]:
        """Return an MPC action vector and MW/MVAr mappings without applying it."""
        if self.manager is None:
            raise RuntimeError('reset() must be called before mpc_action()')
        return get_mpc_action(self.manager, planning_steps=planning_steps, safety_margin=safety_margin)

    def run_mpc(self, *, apply: bool=True, planning_steps: int=8, safety_margin: float=0.9) -> dict[str, Any]:
        """Solve an MPC control on the live manager or a shadow; return checks.

        Applying MPC updates recovery and done without consuming a tool step.
        """
        if self.manager is None:
            raise RuntimeError('reset() must be called before run_mpc()')
        result = run_mpc(self.manager, apply=apply, planning_steps=planning_steps, safety_margin=safety_margin)
        if apply:
            self.recovered = bool(result['recovered'])
            self.done = self.recovered or self.current_step >= self.max_steps
        return result

    def close(self) -> None:
        """Close the underlying gym environment when it provides a close method."""
        if self.manager is not None:
            close = getattr(getattr(self.manager, '_env', None), 'close', None)
            if callable(close):
                close()
__all__ = ['ANMEnv']
