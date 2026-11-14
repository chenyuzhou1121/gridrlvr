"""Collect remote RestoreBench trajectories and simulator-based verl rewards."""

from __future__ import annotations
import asyncio
import inspect
import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from typing import Any, Optional
from uuid import uuid4
from ..env.remote import RestoreBenchRemoteEnv
from ._compat import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput
from .anm_agent_loop import ANMAgentLoop, _extract_tool_schemas, _json_safe, _normalise_token_ids, _observation_match, _verdict_name, parse_tool_call
logger = logging.getLogger(__name__)
PROPOSE_TOOL = 'propose_maneuver'

def _finite_float(value: Any, default: float=0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result or result in (float('inf'), float('-inf')):
        return default
    return result

def _reward_components(step: Mapping[str, Any]) -> Mapping[str, Any]:
    info = step.get('info')
    if not isinstance(info, Mapping):
        return {}
    components = info.get('reward_components')
    return components if isinstance(components, Mapping) else {}

def restorebench_outcome_from_steps(steps: list[Mapping[str, Any]]) -> tuple[bool, float, int]:
    """Return last proposal recovery/reduction and the count of invalid proposals.

    Diagnostic calls are ignored. Reduction is the last proposal's component,
    rather than a sum over the trajectory.
    """
    recovered = False
    violation_reduction = 0.0
    invalid_proposals = 0
    for step in steps:
        if step.get('tool_name') != PROPOSE_TOOL:
            continue
        components = _reward_components(step)
        applied = bool(components.get('applied', str(step.get('verdict', '')).upper() == 'PASS'))
        if not applied:
            invalid_proposals += 1
        info = step.get('info') if isinstance(step.get('info'), Mapping) else {}
        recovered = bool(components.get('recovered', info.get('recovered', False)))
        violation_reduction = _finite_float(components.get('violation_reduction', 0.0))
    return (recovered, violation_reduction, invalid_proposals)

def restorebench_reward(*, recovered: bool, violation_reduction: float=0.0, invalid_proposals: int=0, outcome_weight: float=1.0, violation_reduction_weight: float=0.5, invalid_penalty: float=0.0) -> dict[str, Any]:
    """Return weighted recovery/reduction minus a penalty per invalid proposal."""
    outcome_reward = float(bool(recovered))
    reduction = _finite_float(violation_reduction)
    invalid = int(invalid_proposals)
    process_reward = _finite_float(violation_reduction_weight, 0.5) * reduction
    score = _finite_float(outcome_weight, 1.0) * outcome_reward + process_reward - _finite_float(invalid_penalty) * invalid
    return {'score': float(score), 'outcome_reward': outcome_reward, 'violation_reduction': reduction, 'process_reward': float(process_reward), 'invalid_proposals': invalid, 'recovered': bool(recovered)}

def compute_score_restorebench(data_source: str, solution_str: str, ground_truth: Any, extra_info: Optional[Mapping[str, Any]]=None, *, outcome_weight: float=1.0, violation_reduction_weight: float=0.5, invalid_penalty: float=0.0) -> dict[str, Any]:
    """Score restorebench_trajectory metadata; generated answer text is ignored."""
    del data_source, solution_str, ground_truth
    extra = dict(extra_info or {})
    tool_fields = extra.get('tool_extra_fields')
    if isinstance(tool_fields, Mapping):
        merged = dict(tool_fields)
        merged.update(extra)
        extra = merged
    raw_steps = extra.get('restorebench_trajectory') or []
    steps = [step for step in raw_steps if isinstance(step, Mapping)]
    (recovered, violation_reduction, invalid_proposals) = restorebench_outcome_from_steps(steps)
    if 'recovered' in extra:
        recovered = bool(extra['recovered'])
    result = restorebench_reward(recovered=recovered, violation_reduction=violation_reduction, invalid_proposals=invalid_proposals, outcome_weight=outcome_weight, violation_reduction_weight=violation_reduction_weight, invalid_penalty=invalid_penalty)
    result['acc'] = result['outcome_reward']
    return result

class RestoreBenchAgentLoop(AgentLoopBase):
    """Interleave remote tool feedback with policy tokens under verl budgets.

    Response masks include assistant tokens in the loss and exclude tool
    tokens. Recovery comes from proposal reward components, not PASS alone.
    """

    def __init__(
        self, *args: Any,
        env_factory: Optional[Callable[..., Any]] = None,
        base_url: Optional[str] = None,
        corpus: Optional[str] = None,
        timeout: float = 30.0,
        terminate_on_propose: bool = True,
        max_steps: int = 8,
        outcome_weight: float = 1.0,
        violation_reduction_weight: float = 0.5,
        invalid_penalty: float = 0.0,
        presence_penalty: float = 1.5,
        min_p: float = 0.0,
        sampling_temperature: Optional[float] = None,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> None:
        """Configure HTTP episodes, tool budget, rewards, and sampling defaults.

        Remaining arguments initialize AgentLoopBase. extra_info.corpus can
        override corpus per row; timeout is in seconds.
        """
        kwargs.pop('tools', None)
        super().__init__(*args, **kwargs)
        self.env_factory = env_factory
        self.base_url = base_url
        self.corpus = corpus
        self.timeout = float(timeout)
        self.terminate_on_propose = bool(terminate_on_propose)
        self.max_steps = int(max_steps)
        self.outcome_weight = float(outcome_weight)
        self.violation_reduction_weight = float(violation_reduction_weight)
        self.invalid_penalty = float(invalid_penalty)
        self.presence_penalty = float(presence_penalty)
        self.min_p = float(min_p)
        self.sampling_temperature = None if sampling_temperature is None else float(sampling_temperature)
        self.enable_thinking = bool(enable_thinking)
        self._prompt_dropped_prefix = 0
        self._prompt_rendered_length = 0
        self._prompt_observation_match: dict[str, Any] = {}
        self._active_env: Any = None
        rollout = getattr(self, 'rollout_config', None)
        self.prompt_length = int(getattr(rollout, 'prompt_length', 16384))
        self.response_length = int(getattr(rollout, 'response_length', 16384))
        multi_turn = getattr(rollout, 'multi_turn', None)
        self.tokenization_sanity_check_mode = str(getattr(multi_turn, 'tokenization_sanity_check_mode', 'strict'))

    def _make_env(self, *, corpus: Optional[str]=None) -> Any:
        env = self._build_env(corpus=corpus)
        self._active_env = env
        return env

    def _build_env(self, *, corpus: Optional[str]=None) -> Any:
        corpus = self.corpus if corpus is None else corpus
        if self.env_factory is None:
            return RestoreBenchRemoteEnv(base_url=self.base_url, corpus=corpus, timeout=self.timeout, terminate_on_propose=self.terminate_on_propose)
        try:
            return self.env_factory(corpus=corpus)
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
            logger.warning('RestoreBench prompt renderer disagreement: the engine prompt (%d tokens) is not the tail of the local re-render (%d tokens); _tool_tokens will not be able to diff against it', len(ids), len(rendered))
        if self._prompt_dropped_prefix:
            message = f'RestoreBench prompt over budget: {self._prompt_dropped_prefix} head tokens dropped (rendered={len(rendered)}, prompt_length={self.prompt_length}); the system prompt and the rendered tool table live in that head'
            if os.environ.get('GRIDRLVR_RESTOREBENCH_PROMPT_OVERFLOW', 'warn').lower() == 'error':
                raise ValueError(message)
            logger.warning('%s', message)
        return ids

    def _assistant_template_content(self, text: str) -> str:
        eos_token = str(getattr(self.tokenizer, 'eos_token', '') or '')
        if eos_token and text.endswith(eos_token):
            return text[:-len(eos_token)]
        return text

    async def _render_conversation_tokens(self, messages: list[dict[str, Any]], tool_schemas: list[dict[str, Any]]) -> list[int]:
        rendered = await self.loop.run_in_executor(None, lambda : self.tokenizer.apply_chat_template(messages, tools=tool_schemas, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs))
        return _normalise_token_ids(rendered)

    async def _tool_tokens(self, messages: list[dict[str, Any]], tool_schemas: list[dict[str, Any]], runtime_ids: list[int], dropped_prefix: int=0) -> tuple[list[int], dict[str, Any]]:
        """Return a rendered tool suffix only when sampled tokens stay unchanged."""
        full_ids = await self._render_conversation_tokens(messages, tool_schemas)
        offset = max(0, int(dropped_prefix))
        window = full_ids[offset:offset + len(runtime_ids)]
        prefix_ok = window == runtime_ids
        first_mismatch = None
        if not prefix_ok:
            first_mismatch = min(len(runtime_ids), len(window))
            for index in range(first_mismatch):
                if runtime_ids[index] != window[index]:
                    first_mismatch = index
                    break
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

    def _close_active_env(self) -> None:
        env = self._active_env
        self._active_env = None
        close = getattr(env, 'close', None) if env is not None else None
        if not callable(close):
            return
        try:
            close()
        except Exception as exc:
            logger.warning('RestoreBench environment close failed during teardown: %s', exc)

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        """Return one row's tokens, masks, reward, and metadata; always close HTTP state.

        kwargs supplies raw_prompt and scenario_id/seed/corpus in extra_info.
        Prompt and response token limits come from rollout_config.
        """
        self._active_env = None
        try:
            return await self._run_episode(sampling_params, **kwargs)
        finally:
            self._close_active_env()

    async def _run_episode(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not None and getattr(self, 'loop', None) is not running:
            self.loop = running
        wall_started = time.time()
        raw_prompt = [dict(message) for message in kwargs['raw_prompt']]
        extra_info = dict(kwargs.get('extra_info') or {})
        scenario_id = str(kwargs.get('scenario_id') or extra_info.get('scenario_id') or 'unknown')
        seed = int(kwargs.get('seed', extra_info.get('seed', 0)) or 0)
        split = str(extra_info.get('split', 'unknown'))
        corpus = extra_info.get('corpus', self.corpus)
        corpus = str(corpus) if corpus is not None else None
        raw_global_step = kwargs.get('global_steps', extra_info.get('global_steps', -1))
        global_step = -1 if raw_global_step is None else int(raw_global_step)
        uid = str(kwargs.get('uid', extra_info.get('uid', '')) or '')
        env = self._make_env(corpus=corpus)
        reset_observation = env.reset(scenario_id=scenario_id, seed=seed)
        tool_schemas = _extract_tool_schemas(env)
        reset_text = str(reset_observation)
        prompt_tail = str(raw_prompt[-1].get('content', '')) if raw_prompt else ''
        match = _observation_match(prompt_tail, reset_text)
        embedded = bool(match['embedded'])
        match['appended'] = not embedded
        self._prompt_observation_match = match
        if not embedded:
            logger.warning('RestoreBench reset observation not found in the dataset prompt (scenario=%s corpus=%s, observation=%d chars, prompt tail=%d chars); appending it as a new user turn', scenario_id, corpus, len(reset_text), len(prompt_tail))
            raw_prompt.append({'role': 'user', 'content': reset_text})
        prompt_ids = await self._initial_tokens(raw_prompt, tool_schemas)
        prompt_text = self.tokenizer.decode(prompt_ids)
        chat_template = str(getattr(self.tokenizer, 'chat_template', '') or '')
        tool_block_present = '<tools>' in prompt_text if '<tools>' in chat_template else None
        self._prompt_probe = {
            'n_tokens': len(prompt_ids),
            'rendered_tokens': self._prompt_rendered_length,
            'dropped_prefix': self._prompt_dropped_prefix,
            'tool_schema_names': [(schema.get('function') or schema).get('name') for schema in tool_schemas],
            'tool_block_present': tool_block_present,
            'observation_match': dict(self._prompt_observation_match),
            'head': self.tokenizer.decode(prompt_ids[:600]),
            'tail': self.tokenizer.decode(prompt_ids[-400:]),
        }
        if tool_block_present is False:
            logger.warning('RestoreBench prompt has no <tools> block after budgeting (scenario=%s, tokens=%d, rendered=%d, dropped_prefix=%d): the policy cannot see the tool table and will invent action names', scenario_id, len(prompt_ids), self._prompt_rendered_length, self._prompt_dropped_prefix)
        messages = list(raw_prompt)
        runtime_ids = list(prompt_ids)
        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        logprobs_complete = True
        steps: list[dict[str, Any]] = []
        request_id = uuid4().hex
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
                self._no_tool_call_text = text[:2000]
                break
            (tool_name, params) = parsed
            messages.append({'role': 'assistant', 'content': self._assistant_template_content(text)})
            tool_started = time.perf_counter()
            (observation, verdict, info, done) = env.step(tool_name, params)
            tool_seconds += time.perf_counter() - tool_started
            step_record = {'step': step_index, 'tool_name': tool_name, 'params': _json_safe(params), 'assistant_text': text, 'verdict': _verdict_name(verdict), 'done': bool(done), 'info': _json_safe(info or {})}
            steps.append(step_record)
            messages.append({'role': 'tool', 'content': str(observation)})
            (tool_ids, tokenization_check) = await self._tool_tokens(messages, tool_schemas, runtime_ids, self._prompt_dropped_prefix)
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
            if done:
                (recovered_now, _, _) = restorebench_outcome_from_steps(steps)
                termination_reason = 'recovered' if recovered_now else 'environment_done'
                break
        (recovered, violation_reduction, invalid_proposals) = restorebench_outcome_from_steps(steps)
        reward = restorebench_reward(recovered=recovered, violation_reduction=violation_reduction, invalid_proposals=invalid_proposals, outcome_weight=self.outcome_weight, violation_reduction_weight=self.violation_reduction_weight, invalid_penalty=self.invalid_penalty)
        violation_count = sum((1 for step in steps if step['verdict'] in ('FAIL', 'ERROR')))
        reward_extra_info = {'acc': reward['outcome_reward'], 'outcome_reward': reward['outcome_reward'], 'violation_reduction': reward['violation_reduction'], 'process_reward': reward['process_reward'], 'violation_count': violation_count}
        extra_fields = {
            'scenario_id': scenario_id,
            'corpus': corpus,
            'recovered': recovered,
            'termination_reason': termination_reason,
            'restorebench_trajectory': steps,
            'reward_extra_info': reward_extra_info,
            'tokenization_sanity': tokenization_checks,
            'prompt_dropped_prefix': self._prompt_dropped_prefix,
            'prompt_rendered_length': self._prompt_rendered_length,
            'prompt_observation_match': dict(self._prompt_observation_match),
            'no_tool_call_text': getattr(self, '_no_tool_call_text', None),
            'prompt_probe': getattr(self, '_prompt_probe', None),
        }
        extra_fields.update(generation_extra_fields)
        for key in ('min_global_steps', 'max_global_steps'):
            if extra_fields.get(key) is None:
                extra_fields[key] = global_step
        mask_contract = ANMAgentLoop._mask_contract(response_ids, response_mask, spans)
        if not all((mask_contract[key] for key in ('length_equal', 'assistant_mask_all_one', 'tool_mask_all_zero'))):
            raise ValueError(f'response mask contract failed: {mask_contract}')
        rollout_config = getattr(self, 'rollout_config', None)
        audit_record: dict[str, Any] = {
            'schema_version': 'gridrlvr.verl_audit.v1',
            'request_id': request_id,
            'uid': uid,
            'scenario_id': scenario_id,
            'corpus': corpus,
            'split': split,
            'global_step': global_step,
            'seed': seed,
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
            'no_tool_call_text': getattr(self, '_no_tool_call_text', None),
            'prompt_probe': getattr(self, '_prompt_probe', None),
            'prompt_length': len(prompt_ids),
            'response_length': len(response_ids),
            'response_logprobs_length': len(response_logprobs) if logprobs_complete else None,
            'num_turns': 1 + 2 * len(steps),
            'actor_logprob_temperature': float(getattr(rollout_config, 'temperature', 1.0)),
            'sampling_temperature': effective_sampling_temperature,
            'min_global_steps': extra_fields['min_global_steps'],
            'max_global_steps': extra_fields['max_global_steps'],
            'wall_started': wall_started,
            'wall_finished': time.time(),
        }
        if os.environ.get('GRIDRLVR_AUDIT_LEVEL', 'summary') == 'full':
            audit_record.update({'prompt_ids': prompt_ids, 'response_ids': response_ids, 'response_mask': response_mask, 'response_logprobs': response_logprobs if logprobs_complete else None, 'messages': messages, 'tool_schemas': tool_schemas})
        audit_path = ANMAgentLoop._write_audit(audit_record)
        if audit_path:
            extra_fields['audit_path'] = audit_path
        logger.info(
            'RESTOREBENCH_TRAJECTORY scenario=%s corpus=%s split=%s recovered=%s '
            'reward=%.3f verdicts=%s response_tokens=%d assistant_tokens=%d tool_tokens=%d '
            'tokenization=%s prompt_tokens=%d prompt_rendered=%d prompt_dropped=%d '
            'obs_match=%s tools_visible=%s',
            scenario_id, corpus, split, recovered, float(reward['score']),
            [step['verdict'] for step in steps], len(response_ids),
            mask_contract['assistant_tokens'], mask_contract['tool_tokens'],
            all((check['passed'] for check in tokenization_checks)), len(prompt_ids),
            self._prompt_rendered_length, self._prompt_dropped_prefix,
            self._prompt_observation_match.get('kind'),
            (self._prompt_probe or {}).get('tool_block_present'),
        )
        metrics = AgentLoopMetrics(generate_sequences=generate_seconds, tool_calls=tool_seconds, compute_score=0.0, num_preempted=total_preempted)
        output = AgentLoopOutput(prompt_ids=prompt_ids, response_ids=response_ids, response_mask=response_mask, response_logprobs=response_logprobs if logprobs_complete else None, reward_score=float(reward['score']), num_turns=1 + 2 * len(steps), metrics=metrics, extra_fields=extra_fields)
        return output
