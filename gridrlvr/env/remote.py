"""Provide a stateful HTTP client for the RestoreBench environment service."""

from __future__ import annotations
import http.client
import json
import logging
import os
import threading
import urllib.error
import urllib.request
from typing import Any, Mapping, Optional
logger = logging.getLogger(__name__)
DEFAULT_RESTOREBENCH_ENV_URL = 'http://127.0.0.1:8737'
_URL_CURSOR = 0
_URL_CURSOR_LOCK = threading.Lock()

def _next_url(urls: list[str]) -> str:
    global _URL_CURSOR
    if len(urls) == 1:
        return urls[0]
    with _URL_CURSOR_LOCK:
        url = urls[_URL_CURSOR % len(urls)]
        _URL_CURSOR += 1
    return url

class RestoreBenchRemoteEnv:
    """Expose reset/step/close over one selected RestoreBench service URL.

    A comma-separated URL list distributes new clients in round-robin order;
    all calls for an episode stay on its selected service.
    """

    def __init__(self, base_url: Optional[str]=None, corpus: Optional[str]=None, timeout: float=30.0, terminate_on_propose: bool=True) -> None:
        """Select a URL, corpus name, timeout in seconds, and proposal stop mode."""
        raw = str(base_url or os.environ.get('GRIDRLVR_RESTOREBENCH_ENV_URL') or DEFAULT_RESTOREBENCH_ENV_URL)
        self.base_urls = [part.strip().rstrip('/') for part in raw.split(',') if part.strip()]
        if not self.base_urls:
            self.base_urls = [DEFAULT_RESTOREBENCH_ENV_URL]
        self.base_url = _next_url(self.base_urls)
        self.corpus = corpus
        self.timeout = float(timeout)
        self.terminate_on_propose = bool(terminate_on_propose)
        self.episode_id: Optional[str] = None
        self.system_prompt: str = ''
        self.tool_schemas: list[dict[str, Any]] = []
        self.episode_meta: dict[str, Any] = {}
        self.recovered = False

    def _post(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        body = json.dumps(dict(payload)).encode('utf-8')
        request = urllib.request.Request(self.base_url + path, data=body, headers={'Content-Type': 'application/json'}, method='POST')
        last_error: Optional[BaseException] = None
        for _attempt in range(2):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode('utf-8'))
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as exc:
                last_error = exc
        raise RuntimeError(f'RestoreBench environment request {path} failed after one retry: {last_error}')

    def reset(self, scenario_id: str, seed: int=0) -> str:
        """Create a server episode and return its observation; seed is ignored."""
        del seed
        data = self._post('/reset', {'scenario_id': scenario_id, 'corpus': self.corpus, 'terminate_on_propose': self.terminate_on_propose})
        self.recovered = False
        self.episode_id = str(data.get('episode_id', ''))
        self.system_prompt = str(data.get('system_prompt', ''))
        if os.environ.get('GRIDRLVR_RESTOREBENCH_FORCE_TOOL_PROTOCOL', '1') == '1':
            self.system_prompt += (
                '\n\nResponse protocol (mandatory)\nYou MUST answer with a tool call and nothing else. '
                'Emit exactly one call, wrapped in <tool_call> ... </tool_call>, '
                'whose body is a JSON object with keys "name" and "arguments".  Example:\n'
                '<tool_call>\n{"name": "rank_candidate_maneuvers", "arguments": {}}\n</tool_call>\n'
                'Never answer in prose, never use markdown code fences, and never invent tool names: '
                'use only the tools listed above.'
            )
        self.tool_schemas = [dict(schema) for schema in data.get('tool_schemas') or []]
        self.episode_meta = dict(data.get('meta') or {})
        return str(data.get('observation', ''))

    def step(self, tool_name: str, params: dict[str, Any]) -> tuple[str, str, dict[str, Any], bool]:
        """Return observation, verdict, normalized reward info, and done.

        Recovery and application flags come from the server reward components.
        """
        data = self._post('/step', {'episode_id': self.episode_id, 'tool_name': tool_name, 'params': dict(params)})
        info = dict(data.get('info') or {})
        info['reward_components'] = dict(data.get('reward_components') or {})
        components = info['reward_components']
        if components:
            self.recovered = bool(components.get('recovered', False))
            info['applied'] = bool(components.get('applied', False))
            info['recovered'] = self.recovered
            info['process_reward'] = float(components.get('violation_reduction', 0.0))
            info['violation_count'] = int(components.get('violations_after', 0))
        return (str(data.get('observation', '')), str(data.get('verdict', '')), info, bool(data.get('done', False)))

    def close(self) -> None:
        """Request episode cleanup, log failures, and clear the local episode ID."""
        if not self.episode_id:
            return
        try:
            self._post('/close', {'episode_id': self.episode_id})
        except Exception as exc:
            logger.warning('RestoreBench environment /close failed for episode %s: %s', self.episode_id, exc)
        finally:
            self.episode_id = None
