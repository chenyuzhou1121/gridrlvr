"""Run chat-driven restoration episodes with tool parsing and proposal retries."""

from __future__ import annotations
import copy
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field, replace
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence
logger = logging.getLogger(__name__)

class ActionParseError(ValueError):
    """Report a response that cannot be decoded into one tool action."""

class SupportsANMEnv(Protocol):
    """Specify the observation, tool-schema, and recovery interface for runners."""
    tool_schemas: list[dict[str, Any]]
    system_prompt: str
    recovered: bool

    def reset(self, scenario_id: str, seed: int | None=None) -> str:
        """Reset a scenario/replicate and return its initial observation text."""
        ...

    def step(self, tool_name: str, params: dict[str, Any]) -> tuple[str, Any, dict[str, Any], bool]:
        """Return observation, verdict, info, and done for one tool proposal."""
        ...

@dataclass(frozen=True)
class SamplingConfig:
    """Set chat sampling options and endpoint-specific request extensions.

    max_tokens is per request. max_tokens_cap limits budget growth after empty
    length-limited responses. extra_body can supply endpoint-specific values.
    """
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int | None = 20
    presence_penalty: float = 1.5
    max_tokens: int = 4096
    max_tokens_cap: int | None = None
    seed: int | None = None
    enable_thinking: bool | None = False
    reasoning_effort: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ReActConfig:
    """Set logical-step and per-step retry budgets for one episode.

    gated controls runner retries and interception accounting; the environment
    itself determines whether a proposal changes simulator state.
    """
    max_steps: int = 8
    max_fail_retries: int = 2
    max_empty_length_rescues: int = 2
    gated: bool = True
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

@dataclass
class CompletionResult:
    """Hold normalized text/tool calls, finish reason, usage, and model identity."""
    content: str = ''
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = 'stop'
    usage: dict[str, int] = field(default_factory=dict)
    response_model: str | None = None
    raw_id: str | None = None

def load_api_key(key_file: str | os.PathLike[str] | None=None, env_names: Sequence[str]=('OPENAI_API_KEY',)) -> str | None:
    """Return the first nonempty environment key, then optional file text, or None."""
    for name in env_names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    if key_file:
        path = Path(key_file).expanduser()
        if path.is_file():
            value = path.read_text(encoding='utf-8').strip()
            if value:
                return value
    return None

class OpenAICompatibleClient:
    """Adapt OpenAI-compatible chat completions to CompletionResult objects."""

    def __init__(self, base_url: str, model: str, *, key_file: str | os.PathLike[str] | None=None, timeout_s: float=180.0, max_retries: int=3, default_headers: Mapping[str, str] | None=None, client: Any | None=None) -> None:
        """Configure an endpoint/model or use an injected SDK-compatible client.

        Authentication uses OPENAI_API_KEY, then key_file; EMPTY supports
        endpoints that do not require credentials. timeout_s is in seconds.
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        if client is not None:
            self._client = client
            return
        try:
            import openai
        except ImportError as exc:
            raise ImportError("Install the 'openai' package to call an endpoint") from exc
        key = load_api_key(key_file)
        kwargs: dict[str, Any] = {'api_key': key or 'EMPTY', 'base_url': self.base_url, 'timeout': timeout_s, 'max_retries': max_retries}
        if default_headers:
            kwargs['default_headers'] = dict(default_headers)
        self._client = openai.OpenAI(**kwargs)

    def list_models(self) -> list[str]:
        """Return model identifiers advertised by the configured endpoint."""
        response = self._client.models.list()
        return [str(_get(item, 'id', '')) for item in _get(response, 'data', []) or []]

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], sampling: SamplingConfig) -> CompletionResult:
        """Request one completion with automatic tool choice and normalize it.

        Nonstandard sampling options go in extra_body. An empty choices list
        raises RuntimeError; usage and returned model identity are preserved.
        """
        extra_body = copy.deepcopy(sampling.extra_body)
        if sampling.top_k is not None:
            extra_body.setdefault('top_k', sampling.top_k)
        if sampling.enable_thinking is not None:
            ctk = extra_body.setdefault('chat_template_kwargs', {})
            ctk.setdefault('enable_thinking', sampling.enable_thinking)
        if sampling.reasoning_effort:
            extra_body.setdefault('reasoning_effort', sampling.reasoning_effort)
        kwargs: dict[str, Any] = {
            'model': self.model,
            'messages': messages,
            'tools': tools,
            'tool_choice': 'auto',
            'temperature': sampling.temperature,
            'top_p': sampling.top_p,
            'presence_penalty': sampling.presence_penalty,
            'max_tokens': sampling.max_tokens,
        }
        if sampling.seed is not None:
            kwargs['seed'] = sampling.seed
        if extra_body:
            kwargs['extra_body'] = extra_body
        response = self._client.chat.completions.create(**kwargs)
        choices = _get(response, 'choices', []) or []
        if not choices:
            raise RuntimeError('Endpoint returned a completion with no choices')
        choice = choices[0]
        message = _get(choice, 'message', {})
        raw_tool_calls = _get(message, 'tool_calls', []) or []
        tool_calls = [_normalise_native_tool_call(tc) for tc in raw_tool_calls]
        tool_calls = _merge_split_tool_call(tool_calls)
        usage_obj = _get(response, 'usage', None)
        usage = _normalise_usage(usage_obj)
        content = _content_to_text(_get(message, 'content', ''))
        # Normalize byte-level tokenizer markers sometimes returned as text.
        content = content.replace(chr(288), ' ').replace(chr(266), '\n')
        return CompletionResult(content=content, tool_calls=tool_calls, finish_reason=str(_get(choice, 'finish_reason', None) or 'stop'), usage=usage, response_model=_get(response, 'model', None), raw_id=_get(response, 'id', None))

def parse_action(response: CompletionResult, allowed_tools: Sequence[str] | set[str] | frozenset[str]) -> tuple[str, dict[str, Any], str | None]:
    """Return thought text, a tool_name/params mapping, and an optional call ID.

    Prefer the first native call, then JSON and function-like text fallbacks.
    Names are normalized and close matches corrected; the environment still
    validates tool membership and parameters. Raise ActionParseError on failure.
    """
    allowed = {str(name) for name in allowed_tools}
    if response.tool_calls:
        call = response.tool_calls[0]
        name = _normalise_tool_name(str(call.get('name', '')), allowed)
        arguments = call.get('arguments', {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ActionParseError(f'Native tool arguments are not JSON: {exc}') from exc
        if not isinstance(arguments, dict):
            raise ActionParseError('Native tool arguments must be a JSON object')
        return (response.content.strip(), {'tool_name': name, 'params': arguments}, call.get('id'))
    text = response.content.strip()
    if not text:
        raise ActionParseError(f'Empty model response (finish_reason={response.finish_reason!r})')
    action = _parse_json_fallback(text, allowed)
    if action is not None:
        return (_extract_thought(text), action, None)
    action = _parse_regex_fallback(text, allowed)
    if action is not None:
        return (_extract_thought(text), action, None)
    raise ActionParseError(f'Could not parse a tool action from: {text[:200]}')

class ReActRunner:
    """Collect one-action chat turns, verifier feedback, and episode records."""

    def __init__(self, env_factory: Callable[[], SupportsANMEnv] | SupportsANMEnv, client: Any, config: ReActConfig | None=None) -> None:
        """Bind an environment/factory and a client exposing complete()."""
        self.env_factory = env_factory
        self.client = client
        self.config = config or ReActConfig()

    def run_episode(self, scenario_id: str, **kwargs: Any) -> dict[str, Any]:
        """Return a JSON-compatible episode record and close its environment.

        Optional keyword arguments are seed, episode_id, and scenario_metadata.
        The record contains messages, proposals, rewards, usage, and recovery.
        """
        env = self.env_factory() if callable(self.env_factory) else self.env_factory
        runner = ReActRunner(env, self.client, self.config)
        try:
            return runner._run_episode(scenario_id, **kwargs)
        finally:
            close = getattr(env, 'close', None)
            if callable(close):
                close()

    def _run_episode(self, scenario_id: str, *, seed: int | None=None, episode_id: str | None=None, scenario_metadata: Mapping[str, Any] | None=None) -> dict[str, Any]:
        started = time.time()
        env = self.env_factory() if callable(self.env_factory) else self.env_factory
        initial_observation = env.reset(scenario_id=scenario_id, seed=seed)
        tools = copy.deepcopy(list(getattr(env, 'tool_schemas', [])))
        if not tools:
            raise RuntimeError('Environment exposed no OpenAI tool schemas')
        allowed_tools = [t['function']['name'] for t in tools]
        system_prompt = str(getattr(env, 'system_prompt', ''))
        messages: list[dict[str, Any]] = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': initial_observation}]
        steps: list[dict[str, Any]] = []
        token_totals = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        response_models: list[str] = []
        finish_reasons: list[str] = []
        gate_rejections = 0
        violation_count = 0
        parse_errors = 0
        process_reward = 0.0
        accumulated_reward = 0.0
        error: str | None = None
        stop = bool(getattr(env, 'recovered', False))
        adaptive_max_tokens = int(self.config.sampling.max_tokens)
        empty_length_rescues = 0
        for logical_step in range(1, self.config.max_steps + 1):
            if stop:
                break
            for proposal in range(1, self.config.max_fail_retries + 2):
                request_messages = copy.deepcopy(messages)
                call_started = time.perf_counter()
                request_sampling = replace(self.config.sampling, max_tokens=adaptive_max_tokens)
                try:
                    response = self.client.complete(request_messages, tools, request_sampling)
                except Exception as exc:
                    error = _safe_exception_label(exc)
                    steps.append({'step': logical_step, 'proposal': proposal, 'outcome': 'llm_error', 'error': error, 'latency_s': round(time.perf_counter() - call_started, 6)})
                    stop = True
                    break
                prompt_tokens = int(response.usage.get('prompt_tokens', 0) or 0)
                completion_tokens = int(response.usage.get('completion_tokens', 0) or 0)
                token_totals['prompt_tokens'] += prompt_tokens
                token_totals['completion_tokens'] += completion_tokens
                token_totals['total_tokens'] += int(response.usage.get('total_tokens', prompt_tokens + completion_tokens) or 0)
                finish_reasons.append(response.finish_reason)
                if response.response_model:
                    response_models.append(response.response_model)
                try:
                    (thought, action, native_call_id) = parse_action(response, allowed_tools)
                except ActionParseError as exc:
                    empty_length = response.finish_reason == 'length' and (not response.content.strip()) and (not response.tool_calls)
                    if empty_length and empty_length_rescues < self.config.max_empty_length_rescues:
                        # An empty length-limited answer can exhaust its budget in reasoning.
                        nxt = adaptive_max_tokens * 2
                        cap = self.config.sampling.max_tokens_cap
                        if cap is not None:
                            nxt = min(nxt, int(cap))
                        if nxt > adaptive_max_tokens:
                            adaptive_max_tokens = nxt
                            empty_length_rescues += 1
                    parse_errors += 1
                    messages.append({'role': 'assistant', 'content': response.content or '(empty response)'})
                    feedback = '[PARSE ERROR] The previous response did not contain one valid tool call. Emit exactly one native tool call or one JSON object with "tool_name" and "params".'
                    messages.append({'role': 'user', 'content': feedback})
                    steps.append({
                        'step': logical_step,
                        'proposal': proposal,
                        'outcome': 'parse_error',
                        'error': str(exc),
                        'response': _response_metadata(response),
                        'request_max_tokens': request_sampling.max_tokens,
                        'next_max_tokens': adaptive_max_tokens,
                        'latency_s': round(time.perf_counter() - call_started, 6),
                    })
                    if proposal <= self.config.max_fail_retries:
                        continue
                    break
                call_id = native_call_id or f'call_{logical_step}_{proposal}_{uuid.uuid4().hex[:10]}'
                assistant_message = {'role': 'assistant', 'content': response.content or None, 'tool_calls': [{'id': call_id, 'type': 'function', 'function': {'name': action['tool_name'], 'arguments': json.dumps(action['params'], ensure_ascii=False, separators=(',', ':'))}}]}
                messages.append(assistant_message)
                try:
                    (observation, verdict_raw, info_raw, done) = env.step(action['tool_name'], action['params'])
                    verdict = _verdict_text(verdict_raw)
                    info = _json_safe(dict(info_raw or {}))
                except Exception as exc:
                    observation = f'[ENV ERROR] {type(exc).__name__}: {exc}'
                    verdict = 'ERROR'
                    info = {'applied': False, 'error': str(exc)}
                    done = False
                messages.append({'role': 'tool', 'tool_call_id': call_id, 'name': action['tool_name'], 'content': str(observation)})
                applied = bool(info.get('applied', verdict in {'PASS', 'SAFE_PROGRESS'}))
                intercepted = self.config.gated and verdict in {'FAIL', 'ERROR'} and (not applied)
                if intercepted:
                    gate_rejections += 1
                current_violations = info.get('violation_count')
                if current_violations is None:
                    raw_violations = info.get('violations', [])
                    current_violations = len(raw_violations) if isinstance(raw_violations, list) else 0
                violation_count += int(current_violations or 0)
                step_process_reward = float(info.get('process_reward', 0.0) or 0.0)
                process_reward += step_process_reward
                step_total_reward = float(info.get('total_reward', info.get('reward', step_process_reward)) or 0.0)
                accumulated_reward += step_total_reward
                steps.append({
                    'step': logical_step,
                    'proposal': proposal,
                    'thought': thought,
                    'action': _json_safe(action),
                    'verdict': verdict,
                    'applied': applied,
                    'intercepted': intercepted,
                    'violations': info.get('violations', []),
                    'violation_count': int(current_violations or 0),
                    'reward': step_total_reward,
                    'process_reward': step_process_reward,
                    'info': info,
                    'response': _response_metadata(response),
                    'request_max_tokens': request_sampling.max_tokens,
                    'latency_s': round(time.perf_counter() - call_started, 6),
                })
                stop = bool(done or getattr(env, 'recovered', False))
                if stop or verdict in {'PASS', 'SAFE_PROGRESS'} or (not self.config.gated):
                    break
                if proposal <= self.config.max_fail_retries:
                    continue
                break
        recovered = bool(getattr(env, 'recovered', False))
        elapsed = time.time() - started
        tool_proposals = sum(('action' in step for step in steps))
        return {
            'schema_version': 'gridrlvr.episode.v1',
            'episode_id': episode_id or f'{scenario_id}::seed{seed or 0}',
            'scenario_id': scenario_id,
            'scenario': _json_safe(dict(scenario_metadata or {})),
            'seed': seed,
            'mode': 'gated' if self.config.gated else 'ungated',
            'model': getattr(self.client, 'model', None),
            'response_models': sorted(set(response_models)),
            'recovered': recovered,
            'total_steps': len({s['step'] for s in steps if 'step' in s}),
            'total_proposals': len(steps),
            'tool_proposals': tool_proposals,
            'gate_rejections': gate_rejections,
            'unsafe_action_count': gate_rejections,
            'gate_intercept_rate': gate_rejections / tool_proposals if tool_proposals else 0.0,
            'violation_count': violation_count,
            'constraint_violation_observations': violation_count,
            'parse_errors': parse_errors,
            'outcome_reward': 1.0 if recovered else 0.0,
            'process_reward': process_reward,
            'total_reward': accumulated_reward,
            'tokens': token_totals,
            'finish_reasons': finish_reasons,
            'messages': messages,
            'tools': tools,
            'steps': steps,
            'elapsed_s': round(elapsed, 6),
            'error': error,
        }

def _get(obj: Any, key: str, default: Any=None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)

def _safe_exception_label(exc: Exception) -> str:
    status = getattr(exc, 'status_code', None)
    text = str(exc).lower()
    if status == 408 or '408' in text or 'timeout' in text:
        return 'gateway_timeout: HTTP 408'
    if status == 400:
        raw = ' '.join(str(exc).split())
        excerpt = raw[:240]
        return f'{type(exc).__name__}: HTTP 400 | {excerpt}'
    if status is not None:
        return f'{type(exc).__name__}: HTTP {status}'
    return f'{type(exc).__name__}: request failed'

def _content_to_text(content: Any) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, Mapping) and item.get('type') == 'text':
                parts.append(str(item.get('text', '')))
            else:
                parts.append(str(item))
        return '\n'.join(parts)
    return str(content)

def _normalise_native_tool_call(call: Any) -> dict[str, Any]:
    function = _get(call, 'function', {})
    arguments = _get(function, 'arguments', '{}')
    try:
        parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError:
        parsed = arguments
    return {'id': str(_get(call, 'id', '') or ''), 'name': str(_get(function, 'name', '') or ''), 'arguments': parsed}

def _merge_split_tool_call(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(calls) >= 2:
        named = next((c for c in calls if c.get('name')), None)
        argued = next((c for c in calls if c.get('arguments') not in ({}, '', None) and (not c.get('name'))), None)
        if named and argued and (named.get('arguments') in ({}, '', None)):
            return [{'id': named.get('id') or argued.get('id'), 'name': named['name'], 'arguments': argued['arguments']}]
    return calls

def _normalise_usage(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    result: dict[str, int] = {}
    for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
        value = _get(usage, key, None)
        if value is not None:
            result[key] = int(value)
    return result

def _normalise_tool_name(name: str, allowed: set[str]) -> str:
    normal = name.strip().lower().replace('-', '_').replace(' ', '_')
    if normal in allowed or not allowed:
        return normal
    match = get_close_matches(normal, sorted(allowed), n=1, cutoff=0.8)
    return match[0] if match else normal

def _parse_json_fallback(text: str, allowed: set[str]) -> dict[str, Any] | None:
    candidates: list[str] = []
    candidates.extend((match.group(1).strip() for match in re.finditer('```(?:json)?\\s*(.*?)\\s*```', text, re.I | re.S)))
    decoder = json.JSONDecoder()
    for (index, char) in enumerate(text):
        if char != '{':
            continue
        try:
            (_, end) = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        candidates.append(text[index:index + end])
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        action = _action_from_mapping(obj, allowed)
        if action is not None:
            return action
    return None

def _action_from_mapping(obj: Any, allowed: set[str]) -> dict[str, Any] | None:
    if not isinstance(obj, Mapping):
        return None
    if obj.get('type') == 'function' and isinstance(obj.get('function'), Mapping):
        obj = obj['function']
    name = obj.get('tool_name') or obj.get('action') or obj.get('name')
    if not name:
        return None
    name = _normalise_tool_name(str(name), allowed)
    params = obj.get('params')
    if params is None:
        params = obj.get('parameters')
    if params is None:
        params = obj.get('arguments')
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            return None
    if params is None:
        params = {key: value for (key, value) in obj.items() if key not in {'tool_name', 'action', 'name', 'thought', 'reasoning'}}
    if not isinstance(params, Mapping):
        return None
    return {'tool_name': name, 'params': dict(params)}

def _parse_regex_fallback(text: str, allowed: set[str]) -> dict[str, Any] | None:
    for tool in sorted(allowed, key=len, reverse=True):
        match = re.search(f'\\b{re.escape(tool)}\\s*\\((.*?)\\)', text, re.I | re.S)
        if not match:
            continue
        params: dict[str, Any] = {}
        for part in re.split(',\\s*', match.group(1).strip()):
            if '=' not in part:
                continue
            (key, value) = part.split('=', 1)
            params[key.strip().strip('\'"')] = _parse_scalar(value.strip())
        return {'tool_name': tool, 'params': params}
    return None

def _parse_scalar(value: str) -> Any:
    value = value.strip().strip('\'"')
    if value.lower() in {'true', 'false'}:
        return value.lower() == 'true'
    if value.lower() in {'none', 'null'}:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value

def _extract_thought(text: str) -> str:
    stripped = text.lstrip()
    if stripped.startswith('{'):
        return ''
    indices = [idx for marker in ('```', '\n{') if (idx := text.find(marker)) >= 0]
    prose = text[:min(indices)] if indices else text
    return prose.strip()[:4000]

def _verdict_text(verdict: Any) -> str:
    value = getattr(verdict, 'value', verdict)
    text = str(value)
    if '.' in text:
        text = text.rsplit('.', 1)[-1]
    return text.upper()

def _response_metadata(response: CompletionResult) -> dict[str, Any]:
    return {'finish_reason': response.finish_reason, 'usage': response.usage, 'response_model': response.response_model, 'response_id': response.raw_id}

def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        if isinstance(value, Mapping):
            return {str(k): _json_safe(v) for (k, v) in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(v) for v in value]
        if hasattr(value, 'item'):
            try:
                return value.item()
            except Exception:
                pass
        if hasattr(value, 'value'):
            return _json_safe(value.value)
        return str(value)
__all__ = ['ActionParseError', 'CompletionResult', 'OpenAICompatibleClient', 'ReActConfig', 'ReActRunner', 'SamplingConfig', 'load_api_key', 'parse_action']
