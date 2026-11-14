"""Collect ANM tool trajectories with assistant-only verl response masks."""

from __future__ import annotations
import ast
import inspect
import json
import logging
import os
import re
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4
from ._compat import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput
from .reward_fn import trajectory_reward
_TOOL_CALL_BLOCK = re.compile('<tool_call>\\s*(.*?)\\s*</tool_call>', re.DOTALL | re.IGNORECASE)
_FUNCTION_BLOCK = re.compile('<function=([^>]+)>(.*?)</function>', re.DOTALL | re.IGNORECASE)
_PARAMETER_BLOCK = re.compile('<parameter=([^>]+)>\\s*(.*?)\\s*</parameter>', re.DOTALL | re.IGNORECASE)
_FENCED_JSON = re.compile('```(?:json)?\\s*(\\{.*?\\})\\s*```', re.DOTALL | re.IGNORECASE)
logger = logging.getLogger(__name__)

def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for (k, v) in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    enum_value = getattr(value, 'value', None)
    if enum_value is not None:
        return _json_safe(enum_value)
    if hasattr(value, 'item'):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)

def _parse_scalar(text: str) -> Any:
    value = text.strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value

def _from_mapping(payload: Mapping[str, Any]) -> Optional[tuple[str, dict[str, Any]]]:
    function = payload.get('function')
    if isinstance(function, Mapping):
        payload = function
    name = payload.get('name', payload.get('tool_name'))
    arguments = payload.get('arguments', payload.get('params', {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if isinstance(name, str) and isinstance(arguments, Mapping):
        return (name.strip(), dict(arguments))
    return None

def parse_tool_call(text: str) -> Optional[tuple[str, dict[str, Any]]]:
    """Return the first decoded tool name/arguments, or None for unparsed text.

    Accept tool_call JSON blocks, function/parameter tags, fenced JSON, and
    bare JSON objects. Names and arguments are validated by the environment.
    """
    for block in _TOOL_CALL_BLOCK.findall(text):
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, Mapping):
            parsed = _from_mapping(payload)
            if parsed is not None:
                return parsed
        function_match = _FUNCTION_BLOCK.search(block)
        if function_match:
            (name, body) = function_match.groups()
            params = {key.strip(): _parse_scalar(value) for (key, value) in _PARAMETER_BLOCK.findall(body)}
            return (name.strip(), params)
    function_match = _FUNCTION_BLOCK.search(text)
    if function_match:
        (name, body) = function_match.groups()
        params = {key.strip(): _parse_scalar(value) for (key, value) in _PARAMETER_BLOCK.findall(body)}
        return (name.strip(), params)
    candidates = _FENCED_JSON.findall(text)
    stripped = text.strip()
    if stripped.startswith('{') and stripped.endswith('}'):
        candidates.append(stripped)
    for candidate in reversed(candidates):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            parsed = _from_mapping(payload)
            if parsed is not None:
                return parsed
    return None

def _verdict_name(verdict: Any) -> str:
    value = getattr(verdict, 'value', verdict)
    text = str(value)
    return text.split('.')[-1].upper()

def _extract_tool_schemas(env: Any) -> list[dict[str, Any]]:
    for name in ('tool_schemas', 'tools'):
        value = getattr(env, name, None)
        if callable(value):
            value = value()
        if isinstance(value, list):
            return [_json_safe(schema) for schema in value]
    getter = getattr(env, 'get_tool_schemas', None)
    if callable(getter):
        return [_json_safe(schema) for schema in getter()]
    raise AttributeError('ANMEnv must expose OpenAI tool schemas via tool_schemas, tools, or get_tool_schemas()')
_FLOAT_LITERAL = re.compile('[-+]?(?:\\d+\\.\\d*|\\.\\d+)(?:[eE][-+]?\\d+)?|[-+]?\\d+[eE][-+]?\\d+')
_FLOAT_PLACEHOLDER = '<f>'

def _float_blind(text: str) -> str:
    return _FLOAT_LITERAL.sub(_FLOAT_PLACEHOLDER, text)

def _observation_embedded(prompt_text: str, observation: str) -> tuple[bool, str]:
    if observation in prompt_text:
        return (True, 'exact')
    if _float_blind(observation) in _float_blind(prompt_text):
        return (True, 'float_normalized')
    return (False, 'absent')

def _observation_match(prompt_text: str, observation: str) -> dict[str, Any]:
    if observation in prompt_text:
        kind = 'exact'
    elif _float_blind(observation) in _float_blind(prompt_text):
        kind = 'float_normalized'
    else:
        kind = 'absent'
    report: dict[str, Any] = {'kind': kind, 'embedded': kind != 'absent', 'observation_chars': len(observation), 'prompt_tail_chars': len(prompt_text)}
    if kind != 'float_normalized':
        return report
    window = prompt_text[-len(observation):] if len(prompt_text) >= len(observation) else prompt_text
    report['length_delta'] = len(prompt_text) - len(observation)
    if report['length_delta'] == 0:
        report['drift_chars'] = sum((1 for (a, b) in zip(window, observation) if a != b))
    before = _FLOAT_LITERAL.findall(window)
    after = _FLOAT_LITERAL.findall(observation)
    if len(before) == len(after):
        worst = 0.0
        for (raw_a, raw_b) in zip(before, after):
            try:
                (value_a, value_b) = (float(raw_a), float(raw_b))
            except ValueError:
                continue
            scale = max(abs(value_a), abs(value_b), 1e-12)
            worst = max(worst, abs(value_a - value_b) / scale)
        report['max_float_rel_drift'] = worst
    return report

def _normalise_token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        if 'input_ids' not in value:
            raise ValueError(f'tokenizer mapping has no input_ids (keys={list(value)})')
        value = value['input_ids']
    elif hasattr(value, 'input_ids'):
        value = value.input_ids
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ValueError(f'expected one tokenized conversation, got batch size {len(value)}')
        value = list(value[0])
    return [int(token_id) for token_id in value]

def _default_env_factory(**kwargs: Any) -> Any:
    from gridrlvr.env import ANMEnv
    if 'scenario_file' in kwargs:
        kwargs['scenarios_path'] = kwargs.pop('scenario_file')
    return ANMEnv(**kwargs)

class ANMAgentLoop(AgentLoopBase):
    """Generate interleaved policy/tool tokens and simulator-based rollout rewards.

    Assistant tokens carry mask one; rendered tool feedback carries mask zero.
    Re-rendering must preserve the sampled prefix before tool tokens append.
    """

    def __init__(
        self, *args: Any,
        env_factory: Optional[Callable[..., Any]] = None,
        scenario_file: str | Path | None = None,
        max_steps: int = 8,
        gated: bool = True,
        gating_policy: str = 'progress_mag',
        outcome_weight: float = 1.0,
        process_reward_weight: float = 0.0,
        violation_penalty: float = 0.0,
        presence_penalty: float = 1.5,
        min_p: float = 0.0,
        sampling_temperature: Optional[float] = None,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> None:
        """Configure environment, tool budget, scoring, and sampling defaults.

        Remaining arguments initialize AgentLoopBase. Prompt/response limits
        come from rollout_config; sampling_temperature overrides train rows.
        """
        kwargs.pop('tools', None)
        super().__init__(*args, **kwargs)
        self.env_factory = env_factory or _default_env_factory
        self.scenario_file = str(scenario_file) if scenario_file is not None else None
        self.max_steps = int(max_steps)
        self.gated = bool(gated)
        self.gating_policy = gating_policy
        self.outcome_weight = float(outcome_weight)
        self.process_reward_weight = float(process_reward_weight)
        self.violation_penalty = float(violation_penalty)
        self.presence_penalty = float(presence_penalty)
        self.min_p = float(min_p)
        self.sampling_temperature = None if sampling_temperature is None else float(sampling_temperature)
        self.enable_thinking = bool(enable_thinking)
        self._prompt_dropped_prefix = 0
        self._prompt_rendered_length = 0
        self._prompt_observation_match: dict[str, Any] = {}
        rollout = getattr(self, 'rollout_config', None)
        self.prompt_length = int(getattr(rollout, 'prompt_length', 16384))
        self.response_length = int(getattr(rollout, 'response_length', 16384))
        multi_turn = getattr(rollout, 'multi_turn', None)
        self.tokenization_sanity_check_mode = str(getattr(multi_turn, 'tokenization_sanity_check_mode', 'strict'))

    def _make_env(self, *, gated: Optional[bool]=None) -> Any:
        env_kwargs: dict[str, Any] = {'max_steps': self.max_steps, 'gated': self.gated if gated is None else bool(gated), 'gating_policy': self.gating_policy}
        if self.scenario_file:
            env_kwargs['scenario_file'] = self.scenario_file
        try:
            return self.env_factory(**env_kwargs)
        except TypeError as exc:
            try:
                signature = inspect.signature(self.env_factory)
            except (TypeError, ValueError):
                raise exc
            accepts_kwargs = any((p.kind == p.VAR_KEYWORD for p in signature.parameters.values()))
            required = [p for p in signature.parameters.values() if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
            if accepts_kwargs or required:
                raise exc
            return self.env_factory()

    async def _initial_tokens(self, messages: list[dict[str, Any]], tool_schemas: list[dict[str, Any]]) -> list[int]:
        ids = list(await self.apply_chat_template(messages, tools=tool_schemas))
        rendered = await self._render_conversation_tokens(messages, tool_schemas)
        if len(ids) > self.prompt_length:
            ids = ids[-self.prompt_length:]
        self._prompt_rendered_length = len(rendered)
        self._prompt_dropped_prefix = max(0, len(rendered) - len(ids))
        if len(ids) > len(rendered) or rendered[self._prompt_dropped_prefix:] != ids:
            logger.warning('ANM prompt renderer disagreement: the engine prompt (%d tokens) is not the tail of the local re-render (%d tokens); _tool_tokens will not be able to diff against it', len(ids), len(rendered))
        if self._prompt_dropped_prefix:
            logger.warning('ANM prompt over budget: %d head tokens dropped (rendered=%d, prompt_length=%d); the system prompt and the rendered tool table live in that head', self._prompt_dropped_prefix, len(rendered), self.prompt_length)
        return ids

    def _assistant_template_content(self, text: str) -> str:
        eos_token = str(getattr(self.tokenizer, 'eos_token', '') or '')
        if eos_token and text.endswith(eos_token):
            return text[:-len(eos_token)]
        return text

    async def _render_conversation_tokens(self, messages: list[dict[str, Any]], tool_schemas: list[dict[str, Any]]) -> list[int]:
        rendered = await self.loop.run_in_executor(None, lambda : self.tokenizer.apply_chat_template(messages, tools=tool_schemas, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs))
        return _normalise_token_ids(rendered)

    async def _tool_tokens(self, messages: list[dict[str, Any]], tool_schemas: list[dict[str, Any]], runtime_ids: list[int], dropped_prefix: Optional[int]=None) -> tuple[list[int], dict[str, Any]]:
        """Append only a template suffix that preserves all sampled token IDs."""
        full_ids = await self._render_conversation_tokens(messages, tool_schemas)
        offset = max(0, int(self._prompt_dropped_prefix if dropped_prefix is None else dropped_prefix))
        window = full_ids[offset:offset + len(runtime_ids)]
        prefix_ok = window == runtime_ids
        first_mismatch = None
        if not prefix_ok:
            first_mismatch = next((index for (index, (actual, expected)) in enumerate(zip(runtime_ids, window, strict=False)) if actual != expected), min(len(runtime_ids), len(window)))
            if self.tokenization_sanity_check_mode == 'strict':
                raise ValueError(f'strict tokenization sanity failed: sampled runtime is not a prefix of the full Qwen template (first_mismatch={first_mismatch}, runtime={len(runtime_ids)}, rendered={len(full_ids)})')
        if not prefix_ok:
            raise ValueError(f'cannot append tool observation without rewriting sampled assistant tokens (mode={self.tokenization_sanity_check_mode}, first_mismatch={first_mismatch})')
        suffix_ids = full_ids[offset + len(runtime_ids):]
        suffix_text = self.tokenizer.decode(suffix_ids, skip_special_tokens=False)
        chat_template = str(getattr(self.tokenizer, 'chat_template', '') or '')
        boundary_ok = bool(suffix_ids)
        if '<tool_response>' in chat_template:
            boundary_ok = boundary_ok and '<tool_response>' in suffix_text and ('<|im_end|>' in suffix_text)
        if not boundary_ok:
            raise ValueError('strict tokenization sanity failed: rendered tool boundary is incomplete')
        return (suffix_ids, {'mode': self.tokenization_sanity_check_mode, 'passed': True, 'runtime_length': len(runtime_ids), 'rendered_length': len(full_ids), 'suffix_length': len(suffix_ids), 'dropped_prefix': offset, 'tool_boundary_passed': boundary_ok})

    @staticmethod
    def _mask_contract(response_ids: list[int], response_mask: list[int], spans: list[dict[str, Any]]) -> dict[str, Any]:
        assistant_ok = all((all((value == 1 for value in response_mask[span['start']:span['end']])) for span in spans if span['kind'] == 'assistant'))
        tool_ok = all((all((value == 0 for value in response_mask[span['start']:span['end']])) for span in spans if span['kind'] == 'tool'))
        return {'length_equal': len(response_ids) == len(response_mask), 'assistant_mask_all_one': assistant_ok, 'tool_mask_all_zero': tool_ok, 'assistant_tokens': sum(response_mask), 'tool_tokens': len(response_mask) - sum(response_mask)}

    @staticmethod
    def _write_audit(record: dict[str, Any]) -> Optional[str]:
        audit_dir = os.environ.get('GRIDRLVR_AUDIT_DIR')
        if not audit_dir:
            return None
        target_dir = Path(audit_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{record['request_id']}.json"
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, ensure_ascii=False) + '\n', encoding='utf-8')
        os.replace(temporary, target)
        return str(target)

    async def _generate(self, request_id: str, prompt_ids: list[int], sampling_params: dict[str, Any]) -> tuple[list[int], Optional[list[float]], int, dict[str, Any]]:
        call_params = dict(sampling_params)
        call_params.setdefault('presence_penalty', self.presence_penalty)
        call_params.setdefault('min_p', self.min_p)
        remaining = max(1, self.response_length - max(0, len(prompt_ids) - self.prompt_length))
        call_params['max_tokens'] = min(int(call_params.get('max_tokens', remaining)), remaining)
        started = time.perf_counter()
        result = await self.server_manager.generate(request_id=request_id, prompt_ids=prompt_ids, sampling_params=call_params)
        elapsed = time.perf_counter() - started
        token_ids = getattr(result, 'token_ids', result)
        if isinstance(token_ids, Mapping):
            token_ids = token_ids.get('token_ids', token_ids.get('output_ids', []))
        log_probs = getattr(result, 'log_probs', None)
        preempted = getattr(result, 'num_preempted', None)
        extra_fields = dict(getattr(result, 'extra_fields', {}) or {})
        self._last_generate_elapsed = elapsed
        return (list(token_ids), list(log_probs) if log_probs is not None else None, int(preempted or 0), extra_fields)

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        """Return tokens, masks, rewards, and trajectory metadata for one row.

        kwargs supplies raw_prompt plus scenario_id/seed directly or through
        extra_info. response_length counts both assistant and tool tokens.
        """
        wall_started = time.time()
        raw_prompt = [dict(message) for message in kwargs['raw_prompt']]
        extra_info = dict(kwargs.get('extra_info') or {})
        scenario_id = str(kwargs.get('scenario_id') or extra_info.get('scenario_id') or 'unknown')
        seed = int(kwargs.get('seed', extra_info.get('seed', 0)) or 0)
        split = str(extra_info.get('split', 'unknown'))
        gated = bool(extra_info.get('gated', self.gated))
        raw_global_step = kwargs.get('global_steps', extra_info.get('global_steps', -1))
        global_step = -1 if raw_global_step is None else int(raw_global_step)
        uid = str(kwargs.get('uid', extra_info.get('uid', '')) or '')
        env = self._make_env(gated=gated)
        reset_observation = env.reset(scenario_id=scenario_id, seed=seed)
        tool_schemas = _extract_tool_schemas(env)
        reset_text = str(reset_observation)
        prompt_tail = str(raw_prompt[-1].get('content', '')) if raw_prompt else ''
        match = _observation_match(prompt_tail, reset_text)
        embedded = bool(match['embedded'])
        match['appended'] = not embedded
        self._prompt_observation_match = match
        if not embedded:
            logger.warning('ANM reset observation not found in the dataset prompt (scenario=%s, observation=%d chars, prompt tail=%d chars); appending it as a new user turn', scenario_id, len(reset_text), len(prompt_tail))
            raw_prompt.append({'role': 'user', 'content': reset_text})
        prompt_ids = await self._initial_tokens(raw_prompt, tool_schemas)
        messages = list(raw_prompt)
        runtime_ids = list(prompt_ids)
        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        logprobs_complete = True
        steps: list[dict[str, Any]] = []
        request_id = uuid4().hex
        recovered = False
        termination_reason = 'max_steps'
        generate_seconds = 0.0
        tool_seconds = 0.0
        total_preempted = 0
        spans: list[dict[str, Any]] = []
        tokenization_checks: list[dict[str, Any]] = []
        generation_extra_fields: dict[str, Any] = {}
        effective_sampling_temperature: Optional[float] = None
        for step_index in range(1, self.max_steps + 1):
            if len(response_ids) >= self.response_length:
                termination_reason = 'response_length'
                break
            turn_sampling_params = dict(sampling_params)
            if split == 'train' and self.sampling_temperature is not None:
                turn_sampling_params['temperature'] = self.sampling_temperature
            effective_sampling_temperature = float(turn_sampling_params.get('temperature', 1.0))
            available = self.response_length - len(response_ids)
            turn_sampling_params['max_tokens'] = min(int(turn_sampling_params.get('max_tokens', available)), available)
            (generated_ids, log_probs, preempted, turn_extra_fields) = await self._generate(request_id, runtime_ids, turn_sampling_params)
            if not generation_extra_fields:
                generation_extra_fields.update(turn_extra_fields)
            else:
                max_global_steps = turn_extra_fields.get('max_global_steps')
                if max_global_steps is not None:
                    generation_extra_fields['max_global_steps'] = max_global_steps
            generate_seconds += getattr(self, '_last_generate_elapsed', 0.0)
            total_preempted += preempted
            if not generated_ids:
                termination_reason = 'empty_generation'
                break
            available = self.response_length - len(response_ids)
            generated_ids = generated_ids[:available]
            runtime_ids.extend(generated_ids)
            assistant_start = len(response_ids)
            response_ids.extend(generated_ids)
            # Only sampled assistant tokens contribute to the policy loss.
            response_mask.extend([1] * len(generated_ids))
            if log_probs is not None and logprobs_complete:
                response_logprobs.extend(log_probs[:len(generated_ids)])
                if len(log_probs) < len(generated_ids):
                    response_logprobs.extend([0.0] * (len(generated_ids) - len(log_probs)))
            else:
                logprobs_complete = False
                response_logprobs.clear()
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            spans.append({'kind': 'assistant', 'start': assistant_start, 'end': len(response_ids), 'token_count': len(generated_ids)})
            parsed = parse_tool_call(text)
            if parsed is None:
                termination_reason = 'no_tool_call'
                break
            (tool_name, params) = parsed
            messages.append({'role': 'assistant', 'content': self._assistant_template_content(text)})
            tool_started = time.perf_counter()
            (observation, verdict, info, done) = env.step(tool_name, params)
            tool_seconds += time.perf_counter() - tool_started
            step_record = {'step': step_index, 'tool_name': tool_name, 'params': _json_safe(params), 'assistant_text': text, 'verdict': _verdict_name(verdict), 'done': bool(done), 'info': _json_safe(info or {})}
            steps.append(step_record)
            tool_message = {'role': 'tool', 'content': str(observation)}
            messages.append(tool_message)
            (tool_ids, tokenization_check) = await self._tool_tokens(messages, tool_schemas, runtime_ids)
            tokenization_check['step'] = step_index
            tokenization_checks.append(tokenization_check)
            available = self.response_length - len(response_ids)
            if len(tool_ids) > available:
                termination_reason = 'response_length'
                break
            runtime_ids.extend(tool_ids)
            tool_start = len(response_ids)
            response_ids.extend(tool_ids)
            response_mask.extend([0] * len(tool_ids))
            spans.append({'kind': 'tool', 'start': tool_start, 'end': len(response_ids), 'token_count': len(tool_ids)})
            if logprobs_complete:
                response_logprobs.extend([0.0] * len(tool_ids))
            recovered = bool(done) and (bool((info or {}).get('recovered', False)) or step_record['verdict'] == 'PASS' or bool(getattr(env, 'recovered', False)))
            if done:
                termination_reason = 'recovered' if recovered else 'environment_done'
                break
        reward = trajectory_reward(recovered=recovered, steps=steps, outcome_weight=self.outcome_weight, process_reward_weight=self.process_reward_weight, violation_penalty=self.violation_penalty)
        reward_extra_info = {'acc': reward['outcome_reward'], 'outcome_reward': reward['outcome_reward'], 'process_reward': reward['process_reward'], 'violation_count': reward['violation_count']}
        extra_fields = {
            'scenario_id': scenario_id,
            'recovered': recovered,
            'termination_reason': termination_reason,
            'anm_trajectory': steps,
            'step_rewards': [
                {
                    'step': step['step'],
                    'verdict': step['verdict'],
                    'process_reward': trajectory_reward(recovered=False, steps=[step])['process_reward'],
                    'violation_count': trajectory_reward(recovered=False, steps=[step])['violation_count'],
                }
                for step in steps
            ],
            'reward_extra_info': reward_extra_info,
            'gated': gated,
            'tokenization_sanity': tokenization_checks,
            'prompt_dropped_prefix': self._prompt_dropped_prefix,
            'prompt_rendered_length': self._prompt_rendered_length,
            'prompt_observation_match': dict(self._prompt_observation_match),
        }
        extra_fields.update(generation_extra_fields)
        for key in ('min_global_steps', 'max_global_steps'):
            if extra_fields.get(key) is None:
                extra_fields[key] = global_step
        mask_contract = self._mask_contract(response_ids, response_mask, spans)
        if not all((mask_contract[key] for key in ('length_equal', 'assistant_mask_all_one', 'tool_mask_all_zero'))):
            raise ValueError(f'response mask contract failed: {mask_contract}')
        request_id_value = request_id
        audit_record: dict[str, Any] = {
            'schema_version': 'gridrlvr.verl_audit.v1',
            'request_id': request_id_value,
            'uid': uid,
            'scenario_id': scenario_id,
            'split': split,
            'global_step': global_step,
            'seed': seed,
            'gated': gated,
            'recovered': recovered,
            'termination_reason': termination_reason,
            'reward': reward,
            'steps': steps,
            'spans': spans,
            'mask_contract': mask_contract,
            'tokenization_sanity': tokenization_checks,
            'prompt_dropped_prefix': self._prompt_dropped_prefix,
            'prompt_rendered_length': self._prompt_rendered_length,
            'prompt_observation_match': dict(self._prompt_observation_match),
            'prompt_length': len(prompt_ids),
            'response_length': len(response_ids),
            'response_logprobs_length': len(response_logprobs) if logprobs_complete else None,
            'num_turns': 1 + 2 * len(steps),
            'actor_logprob_temperature': float(getattr(getattr(self, 'rollout_config', None), 'temperature', 1.0)),
            'sampling_temperature': effective_sampling_temperature,
            'min_global_steps': extra_fields['min_global_steps'],
            'max_global_steps': extra_fields['max_global_steps'],
            'wall_started': wall_started,
            'wall_finished': time.time(),
        }
        if os.environ.get('GRIDRLVR_AUDIT_LEVEL', 'summary') == 'full':
            audit_record.update({'prompt_ids': prompt_ids, 'response_ids': response_ids, 'response_mask': response_mask, 'response_logprobs': response_logprobs if logprobs_complete else None, 'messages': messages, 'tool_schemas': tool_schemas})
        audit_path = self._write_audit(audit_record)
        if audit_path:
            extra_fields['audit_path'] = audit_path
        logger.info(
            'ANM_TRAJECTORY scenario=%s split=%s gated=%s recovered=%s reward=%.3f '
            'verdicts=%s response_tokens=%d assistant_tokens=%d tool_tokens=%d tokenization=%s',
            scenario_id, split, gated, recovered, float(reward['score']),
            [step['verdict'] for step in steps], len(response_ids),
            mask_contract['assistant_tokens'], mask_contract['tool_tokens'],
            all((check['passed'] for check in tokenization_checks)),
        )
        metrics = AgentLoopMetrics(generate_sequences=generate_seconds, tool_calls=tool_seconds, compute_score=0.0, num_preempted=total_preempted)
        output = AgentLoopOutput(prompt_ids=prompt_ids, response_ids=response_ids, response_mask=response_mask, response_logprobs=response_logprobs if logprobs_complete else None, reward_score=float(reward['score']), num_turns=1 + 2 * len(steps), metrics=metrics, extra_fields=extra_fields)
        close = getattr(env, 'close', None)
        if callable(close):
            close()
        return output
