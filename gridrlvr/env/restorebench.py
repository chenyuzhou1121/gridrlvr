"""Serve sandboxed RestoreBench episodes and diagnostic tools over HTTP."""

from __future__ import annotations
import argparse
import json
import os
import sys
import threading
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .tool_budget import apply_tool_budget, augment_tool_schemas

def _openai_tools(config: dict[str, Any]) -> list[dict[str, Any]]:
    tools = []
    for entry in config.get('tools', []):
        spec = entry.get('toolSpec', entry)
        tools.append({
            'type': 'function',
            'function': {
                'name': spec['name'],
                'description': spec.get('description', ''),
                'parameters': spec.get('inputSchema', {}).get('json', spec.get('parameters', {})),
            },
        })
    return tools

def _meta_for(corpus: str, scenario_id: str) -> dict[str, Any]:
    return {'scenario_id': scenario_id, 'corpus': corpus, 'maneuver_budget': _CONFIG.MANEUVER_BUDGET}

def _prepare(data_dir: Path, scenario_id: str) -> tuple[Any, Any, PowerFlowResult, str]:
    data_dir = Path(data_dir)
    scenario = load_scenario(scenario_id, data_dir=data_dir)
    lean_net = pp.from_json(str(data_dir / 'lean' / f'{scenario_id}.json'))
    lean_card = render_scenario_card(lean_net)
    working = load_full_net(scenario)
    baseline = run_ac_pf(working)
    if baseline.converged:
        raise ValueError(f'{scenario_id} converges on load; corpus integrity failed')
    return (scenario, working, baseline, lean_card)

def _overstress_of(result: PowerFlowResult | None) -> float | None:
    if result is None or result.diagnostics is None:
        return None
    return result.diagnostics.overstress

def _violation_count(result: PowerFlowResult | None) -> int:
    if result is None:
        return 0
    if not result.converged:
        return 1
    if result.feasibility is None:
        return 1
    return sum((1 for reason in result.feasibility.failure_reasons if reason.code != 'HARD_VOLTAGE_ENVELOPE'))

def _violation_reduction(before: PowerFlowResult | None, after: PowerFlowResult | None) -> float:
    (ob, oa) = (_overstress_of(before), _overstress_of(after))
    if ob is not None and oa is not None:
        return min(1.0, max(0.0, (ob - oa) / max(ob, 1e-09)))
    (vb, va) = (_violation_count(before), _violation_count(after))
    return min(1.0, max(0.0, (vb - va) / max(vb, 1e-09)))

def _quality_summary(result: PowerFlowResult) -> dict[str, Any] | None:
    if result.quality is None:
        return None
    q = result.quality
    return {'clean': q.clean, 'n_buses_out_of_band': q.n_buses_out_of_band, 'worst_vm_pu': q.worst_vm_pu, 'worst_vm_bus': q.worst_vm_bus}

def _diagnostics_summary(result: PowerFlowResult | None) -> dict[str, Any] | None:
    if result is None or result.diagnostics is None:
        return None
    d = result.diagnostics
    return {'worst_bus': d.worst_bus, 'lowest_vm_pu': d.lowest_vm_pu, 'gens_at_q_limit': d.gens_at_q_limit, 'overstress': d.overstress, 'error_message': d.error_message}

@dataclass
class EpisodeState:
    """Keep one corpus scenario's working grid, feedback, and proposal budget."""
    episode_id: str
    corpus: str
    scenario_id: str
    data_dir: Path
    terminate_on_propose: bool
    working: Any
    latest_result: PowerFlowResult
    lock: threading.Lock = field(default_factory=threading.Lock)
    accepted: list[AcceptedManeuver] = field(default_factory=list)
    state_failures: list[FailureFeedback] = field(default_factory=list)
    n_proposes: int = 0
    done: bool = False
    closed: bool = False

    def saturated_gens(self) -> frozenset[int]:
        """Return generator IDs currently reported at reactive-power limits."""
        diagnostics = self.latest_result.diagnostics
        return frozenset(diagnostics.gens_at_q_limit) if diagnostics is not None else frozenset()

    def failed_actions(self) -> tuple[Any, ...]:
        """Return actions rejected on the current grid state."""
        return tuple((f.maneuver.action for f in self.state_failures if f.maneuver is not None))
_BACKEND_LOADED = False

def load_backend(path: str | Path | None=None) -> None:
    """Load RestoreBench dependencies once, optionally from an external source root."""
    global _BACKEND_LOADED, pp, ValidationError, sa, coerce_nested_json
    global GET_ACTION_APPLICABILITY_TOOL_NAME, GET_GRID_TOPOLOGY_TOOL_NAME
    global RANK_CANDIDATE_MANEUVERS_TOOL_NAME, RUN_AC_PF_TOOL_NAME
    global default_diagnostic_tools, render_scenario_card, load_full_net, load_scenario
    global satisfies_non_voltage_constraints, Maneuver, LLMAssignment, OrchestratorConfig
    global InvalidActionError, ToolFailureError, AcceptedManeuver, FailureFeedback
    global PowerFlowResult, run_ac_pf, apply_maneuver, create_sandbox, discard_sandbox
    global promote_sandbox, resolve_net, DIAGNOSTIC_TOOL_NAMES, PROPOSE_TOOL_NAME, _CONFIG
    if _BACKEND_LOADED:
        return
    if path is not None:
        sys.path.insert(0, str(Path(path).expanduser().resolve()))
    import pandapower as pp
    from pydantic import ValidationError
    from restorebench.agents import single_agent as sa
    from restorebench.agents.structured_output import coerce_nested_json
    from restorebench.agents.tool_loop import (
        GET_ACTION_APPLICABILITY_TOOL_NAME,
        GET_GRID_TOPOLOGY_TOOL_NAME,
        RANK_CANDIDATE_MANEUVERS_TOOL_NAME,
        RUN_AC_PF_TOOL_NAME,
        default_diagnostic_tools,
    )
    from restorebench.environment.card_render import render_scenario_card
    from restorebench.environment.scenarios import load_full_net, load_scenario
    from restorebench.physics.feasibility import satisfies_non_voltage_constraints
    from restorebench.schemas.actions import Maneuver
    from restorebench.schemas.config import LLMAssignment, OrchestratorConfig
    from restorebench.schemas.errors import InvalidActionError, ToolFailureError
    from restorebench.schemas.feedback import AcceptedManeuver, FailureFeedback
    from restorebench.schemas.power_flow import PowerFlowResult
    from restorebench.tools.power_flow import run_ac_pf
    from restorebench.tools.sandbox import (
        apply_maneuver, create_sandbox, discard_sandbox, promote_sandbox, resolve_net,
    )
    DIAGNOSTIC_TOOL_NAMES = (
        GET_GRID_TOPOLOGY_TOOL_NAME, GET_ACTION_APPLICABILITY_TOOL_NAME,
        RUN_AC_PF_TOOL_NAME, RANK_CANDIDATE_MANEUVERS_TOOL_NAME,
    )
    PROPOSE_TOOL_NAME = sa.PROPOSE_MANEUVER_TOOL_NAME
    _CONFIG = OrchestratorConfig(
        CONFIGURATION=2,
        LLM_ASSIGNMENT=LLMAssignment(
            single_agent='grid-restoration', analyst=None, executor=None, orchestrator=None,
        ),
        repetition_index=0,
    )
    _BACKEND_LOADED = True

class EnvCore:
    """Manage corpus registrations and serialized tool calls within each episode."""

    def __init__(self, corpora: dict[str, Path]) -> None:
        """Register corpus names/paths and load the simulator backend."""
        load_backend()
        self.corpora = {name: Path(p) for (name, p) in corpora.items()}
        self._episodes: dict[str, EpisodeState] = {}
        self._episodes_lock = threading.Lock()

    def reset(self, scenario_id: str, corpus: str, terminate_on_propose: bool=True) -> dict[str, Any]:
        """Load a diverged scenario and return its episode ID, prompt, and tools."""
        data_dir = self._corpus_dir(corpus)
        (_, working, baseline, lean_card) = _prepare(data_dir, scenario_id)
        payload = {
            'system_prompt': sa._system_prompt(_CONFIG),
            'observation': sa._user_prompt(lean_card, baseline.diagnostics, [], []),
            'tool_schemas': augment_tool_schemas(_openai_tools(sa.single_agent_tool_config())),
            'meta': _meta_for(corpus, scenario_id),
        }
        episode = EpisodeState(
            episode_id=uuid.uuid4().hex, corpus=corpus, scenario_id=scenario_id,
            data_dir=data_dir, terminate_on_propose=bool(terminate_on_propose),
            working=working, latest_result=baseline,
        )
        with self._episodes_lock:
            self._episodes[episode.episode_id] = episode
        return {'episode_id': episode.episode_id, **payload}

    def step(self, episode_id: str, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a diagnostic or maneuver and return the HTTP response payload.

        The episode lock serializes calls. Diagnostics run in discarded
        sandboxes; valid maneuvers promote a sandbox even before recovery.
        """
        episode = self._episode(episode_id)
        with episode.lock:
            if episode.done:
                return self._step_response(
                    observation=json.dumps({'error': 'episode is done; reset or close it'}),
                    done=True, verdict='ERROR',
                    info={'episode_id': episode_id, 'tool_name': tool_name},
                )
            if tool_name == PROPOSE_TOOL_NAME:
                return self._step_propose(episode, params)
            if tool_name in DIAGNOSTIC_TOOL_NAMES:
                return self._step_diagnostic(episode, tool_name, params)
            return self._step_response(
                observation=json.dumps({'error': f'unknown tool {tool_name!r}'}),
                done=False, verdict='ERROR',
                info={'episode_id': episode_id, 'tool_name': tool_name},
            )

    def _step_diagnostic(self, episode: EpisodeState, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        tools = {t.name: t for t in default_diagnostic_tools(episode.saturated_gens(), failed_actions=episode.failed_actions())}
        handler = tools[tool_name].handler
        sandbox = create_sandbox(episode.working)
        try:
            payload = handler(sandbox, self._handler_params(tool_name, params))
        except (InvalidActionError, ValidationError, ToolFailureError, ValueError) as exc:
            return self._step_response(
                observation=json.dumps({'error': f'{type(exc).__name__}: {exc}'}),
                done=False, verdict='ERROR',
                info={'episode_id': episode.episode_id, 'tool_name': tool_name},
            )
        finally:
            discard_sandbox(sandbox)
        payload = apply_tool_budget(tool_name, payload, diagnostics=episode.latest_result.diagnostics, params=params or {})
        return self._step_response(
            observation=json.dumps(payload, sort_keys=True), done=False, verdict='PASS',
            info={'episode_id': episode.episode_id, 'tool_name': tool_name},
        )

    @staticmethod
    def _handler_params(tool_name: str, params: dict[str, Any] | None) -> dict[str, Any]:
        params = params or {}
        if tool_name == GET_GRID_TOPOLOGY_TOOL_NAME:
            return {key: value for (key, value) in params.items() if key not in ('section', 'offset', 'limit')}
        return params

    def _step_propose(self, episode: EpisodeState, params: dict[str, Any]) -> dict[str, Any]:
        try:
            maneuver = Maneuver.model_validate(coerce_nested_json(params, Maneuver))
        except ValidationError as exc:
            episode.n_proposes += 1
            return self._step_response(
                observation=json.dumps({'applied': False, 'kind': 'MALFORMED_OUTPUT', 'detail': str(exc)}),
                done=self._propose_done(episode, recovered=False), verdict='ERROR',
                info={'episode_id': episode.episode_id, 'tool_name': PROPOSE_TOOL_NAME},
                reward_components=self._reward_components(
                    applied=False, recovered=False, before=episode.latest_result, after=None,
                ),
            )
        episode.n_proposes += 1
        if any((failed == maneuver.action for failed in episode.failed_actions())):
            detail = 'the same action already failed on the current grid state; choose a different action'
            feedback = FailureFeedback(iteration=episode.n_proposes - 1, kind='INVALID_ACTION', diagnostics=episode.latest_result.diagnostics, detail=detail, maneuver=maneuver)
            episode.state_failures.append(feedback)
            return self._step_response(
                observation=self._verifier_text(
                    applied=False, kind='INVALID_ACTION', detail=detail,
                    before=episode.latest_result, after=None,
                ),
                done=self._propose_done(episode, recovered=False), verdict='FAIL',
                info={'episode_id': episode.episode_id, 'tool_name': PROPOSE_TOOL_NAME, 'kind': 'INVALID_ACTION'},
                reward_components=self._reward_components(
                    applied=False, recovered=False, before=episode.latest_result, after=episode.latest_result,
                ),
            )
        before = episode.latest_result
        sandbox = create_sandbox(episode.working)
        try:
            apply_maneuver(sandbox, maneuver, saturated_gens=episode.saturated_gens())
        except InvalidActionError as exc:
            discard_sandbox(sandbox)
            feedback = FailureFeedback(iteration=episode.n_proposes - 1, kind='INVALID_ACTION', diagnostics=before.diagnostics, detail=str(exc), maneuver=maneuver)
            episode.state_failures.append(feedback)
            return self._step_response(
                observation=self._verifier_text(
                    applied=False, kind='INVALID_ACTION', detail=str(exc), before=before, after=None,
                ),
                done=self._propose_done(episode, recovered=False), verdict='FAIL',
                info={'episode_id': episode.episode_id, 'tool_name': PROPOSE_TOOL_NAME, 'kind': 'INVALID_ACTION'},
                reward_components=self._reward_components(
                    applied=False, recovered=False, before=before, after=None,
                ),
            )
        except ToolFailureError as exc:
            discard_sandbox(sandbox)
            return self._step_response(
                observation=json.dumps({'applied': False, 'kind': 'TOOL_FAILURE', 'detail': str(exc)}),
                done=self._propose_done(episode, recovered=False), verdict='ERROR',
                info={'episode_id': episode.episode_id, 'tool_name': PROPOSE_TOOL_NAME, 'kind': 'TOOL_FAILURE'},
                reward_components=self._reward_components(
                    applied=False, recovered=False, before=before, after=None,
                ),
            )
        result = run_ac_pf(sandbox)
        # A valid maneuver advances the working grid even if power flow diverges.
        promote_sandbox(sandbox)
        recovered = bool(result.converged and satisfies_non_voltage_constraints(result.feasibility))
        episode.accepted.append(AcceptedManeuver(maneuver=maneuver, overstress_before=_overstress_of(before), overstress_after=_overstress_of(result)))
        if recovered:
            kind = 'RECOVERED'
            detail = 'power flow converged and all non-voltage constraints are satisfied'
            discard_sandbox(sandbox)
            episode.latest_result = result
            episode.done = True
        else:
            episode.working = resolve_net(sandbox)
            discard_sandbox(sandbox)
            episode.latest_result = result
            if result.converged:
                kind = 'SOLVED_INFEASIBLE'
                reasons = [r.detail for r in (result.feasibility.failure_reasons if result.feasibility else ()) if r.code != 'HARD_VOLTAGE_ENVELOPE']
                detail = '; '.join(reasons) if reasons else 'power flow converged but violates a non-voltage benchmark constraint'
                episode.state_failures[:] = [FailureFeedback(iteration=episode.n_proposes - 1, kind='SOLVED_INFEASIBLE', diagnostics=None, detail=detail, maneuver=maneuver)]
            else:
                kind = 'STILL_DIVERGED'
                detail = result.error_message or 'power flow did not converge'
                episode.state_failures.clear()
        return self._step_response(
            observation=self._verifier_text(
                applied=True, kind=kind, detail=detail, before=before, after=result,
            ),
            done=self._propose_done(episode, recovered=recovered), verdict='PASS',
            info={'episode_id': episode.episode_id, 'tool_name': PROPOSE_TOOL_NAME, 'kind': kind},
            reward_components=self._reward_components(
                applied=True, recovered=recovered, before=before, after=result,
            ),
        )

    def _propose_done(self, episode: EpisodeState, *, recovered: bool) -> bool:
        if episode.terminate_on_propose:
            episode.done = True
            return True
        done = recovered or episode.n_proposes >= _CONFIG.MANEUVER_BUDGET
        episode.done = episode.done or done
        return done

    def close(self, episode_id: str) -> dict[str, Any]:
        """Remove an episode if present and return an idempotent ok response."""
        with self._episodes_lock:
            episode = self._episodes.pop(episode_id, None)
        if episode is not None:
            episode.closed = True
        return {'ok': True}

    def _corpus_dir(self, corpus: str) -> Path:
        try:
            return self.corpora[corpus]
        except KeyError:
            raise ValueError(f'unknown corpus {corpus!r}; registered: {sorted(self.corpora)}') from None

    def _episode(self, episode_id: str) -> EpisodeState:
        with self._episodes_lock:
            episode = self._episodes.get(episode_id)
        if episode is None:
            raise KeyError(f'unknown episode_id {episode_id!r}')
        return episode

    @staticmethod
    def _reward_components(*, applied: bool, recovered: bool, before: PowerFlowResult | None, after: PowerFlowResult | None) -> dict[str, Any]:
        return {
            'applied': bool(applied),
            'recovered': bool(recovered),
            'converged': bool(after.converged) if after is not None else False,
            'overstress_before': _overstress_of(before),
            'overstress_after': _overstress_of(after),
            'violations_before': _violation_count(before),
            'violations_after': _violation_count(after),
            'violation_reduction': _violation_reduction(before, after),
        }

    @staticmethod
    def _verifier_text(*, applied: bool, kind: str, detail: str | None, before: PowerFlowResult | None, after: PowerFlowResult | None) -> str:
        payload: dict[str, Any] = {
            'applied': applied,
            'kind': kind,
            'detail': detail,
            'recovered': bool(
                after is not None and after.converged and satisfies_non_voltage_constraints(after.feasibility)
            ),
            'converged': bool(after.converged) if after is not None else False,
            'overstress_before': _overstress_of(before),
            'overstress_after': _overstress_of(after),
            'post_diagnostics': _diagnostics_summary(after),
            'quality': _quality_summary(after) if after is not None else None,
        }
        return json.dumps(payload, sort_keys=True)

    @staticmethod
    def _step_response(*, observation: str, done: bool, verdict: str, info: dict[str, Any], reward_components: dict[str, Any] | None=None) -> dict[str, Any]:
        return {'observation': observation, 'done': bool(done), 'verdict': verdict, 'info': info, 'reward_components': reward_components or {}}

class _Handler(BaseHTTPRequestHandler):
    server: '_EnvHTTPServer'
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt: str, *args: Any) -> None:
        """Write the HTTP request log to standard error."""
        sys.stderr.write('%s - %s\n' % (self.address_string(), fmt % args))

    def do_GET(self) -> None:
        """Return service/corpus metadata at /health, or a JSON 404 response."""
        if self.path.split('?', 1)[0] == '/health':
            self._send(200, {'ok': True, 'corpora': sorted(self.server.core.corpora), 'pid': os.getpid()})
            return
        self._send(404, {'error': f'unknown path {self.path}'})

    def do_POST(self) -> None:
        """Dispatch JSON reset/step/close requests and translate errors to HTTP."""
        try:
            length = int(self.headers.get('Content-Length') or 0)
            body = self.rfile.read(length) if length else b'{}'
            payload = json.loads(body.decode('utf-8') or '{}')
            if not isinstance(payload, dict):
                raise ValueError('request body must be a JSON object')
        except (ValueError, UnicodeDecodeError) as exc:
            self._send(400, {'error': f'bad request: {exc}'})
            return
        path = self.path.split('?', 1)[0]
        try:
            if path == '/reset':
                out = self.server.core.reset(scenario_id=str(payload['scenario_id']), corpus=str(payload['corpus']), terminate_on_propose=bool(payload.get('terminate_on_propose', True)))
            elif path == '/step':
                out = self.server.core.step(episode_id=str(payload['episode_id']), tool_name=str(payload['tool_name']), params=payload.get('params') or {})
            elif path == '/close':
                out = self.server.core.close(episode_id=str(payload['episode_id']))
            else:
                self._send(404, {'error': f'unknown path {self.path}'})
                return
        except KeyError as exc:
            self._send(404, {'error': str(exc).strip('"')})
            return
        except (ValueError, TypeError) as exc:
            self._send(400, {'error': f'{type(exc).__name__}: {exc}'})
            return
        except Exception as exc:
            self._send(500, {'error': f'{type(exc).__name__}: {exc}'})
            return
        self._send(200, out)

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

class _EnvHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr: tuple[str, int], core: EnvCore) -> None:
        super().__init__(addr, _Handler)
        self.core = core

def _parse_corpus(values: list[str]) -> dict[str, Path]:
    corpora: dict[str, Path] = {}
    for value in values:
        if '=' not in value:
            raise SystemExit(f'--corpus expects name=dir, got {value!r}')
        (name, _, directory) = value.partition('=')
        path = Path(directory).expanduser().resolve()
        if not (path / 'manifest.json').is_file():
            raise SystemExit(f'corpus {name!r}: no manifest.json at {path}')
        if name in corpora:
            raise SystemExit(f'duplicate corpus name {name!r}')
        corpora[name] = path
    if not corpora:
        raise SystemExit('at least one --corpus name=dir is required')
    return corpora

def main(argv: list[str] | None=None) -> None:
    """Serve /health, /reset, /step, and /close for repeatable corpus mappings."""
    parser = argparse.ArgumentParser(description='Serve external RestoreBench scenarios over HTTP', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--host', default='127.0.0.1', help='HTTP listen address')
    parser.add_argument('--port', type=int, default=8737, help='HTTP listen port')
    parser.add_argument('--corpus', action='append', default=[], metavar='name=dir', help='repeatable corpus registration')
    parser.add_argument('--restorebench-path', type=Path, help='External source root containing the restorebench package')
    args = parser.parse_args(argv)
    load_backend(args.restorebench_path)
    corpora = _parse_corpus(args.corpus)
    server = _EnvHTTPServer((args.host, args.port), EnvCore(corpora))
    print(f'RestoreBench environment listening on http://{args.host}:{args.port} corpora={sorted(corpora)} pid={os.getpid()}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
if __name__ == '__main__':
    main()
