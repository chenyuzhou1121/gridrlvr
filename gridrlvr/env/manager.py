"""Manage gym-anm conditions, control setpoints, and isolated simulator copies."""

from __future__ import annotations
import copy
import logging
from typing import Any
logger = logging.getLogger(__name__)

class GymANMManager:
    """Wrap a gym-anm simulator with fixed exogenous conditions and MW controls.

    solve transitions the simulator without resampling loads or generation.
    advance also samples the next conditions and advances the manager clock.
    """

    def __init__(self, env: Any | None=None, env_name: str='ANM6Easy', seed: int | None=None) -> None:
        """Reset a supplied environment, or construct env_name, and solve once."""
        if env is None:
            import gym_anm.envs as gym_anm_envs
            try:
                env_class = getattr(gym_anm_envs, env_name)
            except AttributeError as exc:
                raise ValueError(f'Unknown gym-anm environment {env_name!r}') from exc
            env = env_class()
        self._env = env
        try:
            env.reset(seed=int(seed)) if seed is not None else env.reset()
        except TypeError:
            env.reset()
        self._sim = env.simulator
        self._seed = seed
        from gym_anm.simulator.components import Generator, Load, StorageUnit
        devices = self._sim.devices
        self._load_ids = [idx for (idx, dev) in devices.items() if isinstance(dev, Load)]
        self._gen_ids = [idx for (idx, dev) in devices.items() if isinstance(dev, Generator) and (not dev.is_slack)]
        self._des_ids = [idx for (idx, dev) in devices.items() if isinstance(dev, StorageUnit)]
        if len(self._load_ids) != self._sim.N_load:
            raise RuntimeError('gym-anm load ordering changed')
        if len(self._gen_ids) != self._sim.N_non_slack_gen:
            raise RuntimeError('gym-anm generator ordering changed')
        self._time = 0.0
        self._dt = float(getattr(env, 'delta_t', 1.0))
        self._P_load: dict[int, float] = {}
        self._P_pot: dict[int, float] = {}
        self._P_set: dict[int, float] = {}
        self._Q_set: dict[int, float] = {}
        self._last_reward = 0.0
        self._last_penalty = 0.0
        self._sample_conditions()
        self._init_default_setpoints()
        if not self.solve():
            logger.warning('Initial gym-anm transition did not converge')

    @property
    def sim_time(self) -> float:
        """Return accumulated advance time in the environment's time units."""
        return self._time

    @property
    def base_mva(self) -> float:
        """Return the simulator base used to convert per-unit power to MW/MVAr."""
        return float(self._sim.baseMVA)

    @property
    def system_state(self) -> Any:
        """Return the underlying simulator with buses, branches, and devices."""
        return self._sim

    @property
    def last_reward(self) -> float:
        """Return the native reward from the most recent completed transition."""
        return self._last_reward

    @property
    def last_penalty(self) -> float:
        """Return the native penalty from the most recent completed transition."""
        return self._last_penalty

    @property
    def seed(self) -> int | None:
        """Return the seed requested at construction."""
        return self._seed

    def get_generator_ids(self) -> list[int]:
        """Return non-slack generator IDs in simulator device order."""
        return list(self._gen_ids)

    def get_storage_ids(self) -> list[int]:
        """Return storage IDs in simulator device order."""
        return list(self._des_ids)

    def _sample_conditions(self) -> None:
        values = self._env.next_vars(self._env.state)
        n_load = self._sim.N_load
        self._P_load = {device_id: float(values[index]) for (index, device_id) in enumerate(self._load_ids)}
        self._P_pot = {device_id: float(values[n_load + index]) for (index, device_id) in enumerate(self._gen_ids)}

    def _init_default_setpoints(self) -> None:
        self._P_set = {idx: self._P_pot.get(idx, 0.0) for idx in self._gen_ids}
        self._Q_set = {idx: 0.0 for idx in self._gen_ids}
        for idx in self._des_ids:
            self._P_set[idx] = 0.0
            self._Q_set[idx] = 0.0

    def set_conditions(self, P_load: dict[int, float], P_pot: dict[int, float], *, reset_setpoints: bool=True, solve: bool=True) -> bool:
        """Set complete load/potential-power mappings in MW and optionally solve.

        Mapping keys must exactly match the manager's load and generator IDs.
        Resetting setpoints uses available generation, zero Q, and idle storage.
        Return convergence, or True when solving is disabled.
        """
        if set(P_load) != set(self._load_ids):
            raise ValueError(f'P_load ids {sorted(P_load)} != load ids {sorted(self._load_ids)}')
        if set(P_pot) != set(self._gen_ids):
            raise ValueError(f'P_pot ids {sorted(P_pot)} != generator ids {sorted(self._gen_ids)}')
        self._P_load = {int(key): float(value) for (key, value) in P_load.items()}
        self._P_pot = {int(key): float(value) for (key, value) in P_pot.items()}
        if reset_setpoints:
            self._init_default_setpoints()
        return self.solve() if solve else True

    def create_shadow_copy(self) -> 'GymANMManager':
        """Return an independent simulator/control copy without a live gym env."""
        shadow = GymANMManager.__new__(GymANMManager)
        shadow._env = None
        shadow._sim = copy.deepcopy(self._sim)
        self._copy_metadata_to(shadow)
        return shadow

    def _copy_metadata_to(self, target: 'GymANMManager') -> None:
        target._seed = self._seed
        target._load_ids = list(self._load_ids)
        target._gen_ids = list(self._gen_ids)
        target._des_ids = list(self._des_ids)
        target._time = self._time
        target._dt = self._dt
        target._P_load = dict(self._P_load)
        target._P_pot = dict(self._P_pot)
        target._P_set = dict(self._P_set)
        target._Q_set = dict(self._Q_set)
        target._last_reward = self._last_reward
        target._last_penalty = self._last_penalty

    def solve(self) -> bool:
        """Transition using current conditions and controls; return convergence.

        Loads and potential generation stay fixed; simulator state, including
        storage SoC, follows the simulator's transition semantics.
        """
        try:
            (_, reward, _energy_loss, penalty, converged) = self._sim.transition(self._P_load, self._P_pot, self._P_set, self._Q_set)
        except Exception as exc:
            logger.warning('gym-anm transition failed (%s: %s)', type(exc).__name__, exc)
            return False
        self._last_reward = float(reward)
        self._last_penalty = float(penalty)
        return bool(converged)

    def advance(self, *, reset_setpoints: bool=True) -> bool:
        """Solve, advance time, and sample new conditions; return solve status.

        A live gym environment is required for stochastic condition sampling.
        """
        converged = self.solve()
        self._time += self._dt
        if self._env is None:
            raise RuntimeError('A verifier shadow cannot advance stochastic time')
        try:
            if hasattr(self._sim, 'state_values'):
                self._env.state = self._sim.state_values
        except Exception:
            logger.debug('Could not sync gym-anm environment state', exc_info=True)
        self._sample_conditions()
        if reset_setpoints:
            self._init_default_setpoints()
        return converged
    step = advance
