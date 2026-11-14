"""Convert gym-anm MPC controls to manager setpoints and check their outcomes."""

from __future__ import annotations
from typing import Any
from .verifier import check_manager

def get_mpc_action(manager: Any, *, planning_steps: int=8, safety_margin: float=0.9) -> dict[str, Any]:
    """Return constant-forecast MPC controls for a manager with a live gym env.

    The vector order is generator P, generator Q, storage P, storage Q, in
    MW/MVAr. Also return device-ID mappings and the planning parameters.
    """
    if manager._env is None:
        raise RuntimeError('MPC requires a live manager with its gym environment')
    import numpy as np
    from gym_anm import MPCAgentConstant
    env = manager._env
    agent = MPCAgentConstant(simulator=env.simulator, action_space=env.action_space, gamma=env.gamma, safety_margin=float(safety_margin), planning_steps=int(planning_steps))
    vector = np.asarray(agent.act(env), dtype=float)
    n_gen = len(manager._gen_ids)
    n_storage = len(manager._des_ids)
    expected = 2 * n_gen + 2 * n_storage
    if vector.shape != (expected,):
        raise RuntimeError(f'MPC action shape {vector.shape} != ({expected},) for this environment')
    P_set: dict[int, float] = {}
    Q_set: dict[int, float] = {}
    for (value, device_id) in zip(vector[:n_gen], manager._gen_ids):
        P_set[int(device_id)] = float(value)
    for (value, device_id) in zip(vector[n_gen:2 * n_gen], manager._gen_ids):
        Q_set[int(device_id)] = float(value)
    for (value, device_id) in zip(vector[2 * n_gen:2 * n_gen + n_storage], manager._des_ids):
        P_set[int(device_id)] = float(value)
    for (value, device_id) in zip(vector[2 * n_gen + n_storage:], manager._des_ids):
        Q_set[int(device_id)] = float(value)
    return {'action_vector': [float(value) for value in vector.tolist()], 'P_set': P_set, 'Q_set': Q_set, 'planning_steps': int(planning_steps), 'safety_margin': float(safety_margin)}

def run_mpc(manager: Any, *, apply: bool=True, planning_steps: int=8, safety_margin: float=0.9) -> dict[str, Any]:
    """Solve MPC setpoints on the live manager or a shadow and return checks.

    Recovery requires convergence and no violations. applied records the
    requested live/shadow mode; this helper does not use the action gate.
    """
    action = get_mpc_action(manager, planning_steps=planning_steps, safety_margin=safety_margin)
    target = manager if apply else manager.create_shadow_copy()
    target._P_set = dict(action['P_set'])
    target._Q_set = dict(action['Q_set'])
    converged = bool(target.solve())
    checks = check_manager(target) if converged else []
    violations = sum((len(check.violations) for check in checks))
    return {**action, 'applied': bool(apply), 'converged': converged, 'recovered': converged and violations == 0, 'violation_count': violations, 'native_penalty': float(getattr(target, 'last_penalty', 0.0))}
