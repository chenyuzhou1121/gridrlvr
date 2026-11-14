# gridrlvr

gridrlvr implements SimVerify-RL, reinforcement learning for multi-turn power-grid restoration
with rewards verified by an AC power-flow solver. The same solver checks proposed controls
and selects among sampled restoration trajectories at inference time.

## Installation

Use Python 3.11. Run commands from the repository root.

```sh
python3 -m pip install -e .
```

Base dependencies are gym-anm 2.0.1, NumPy, SciPy, CVXPY, OpenAI, and HTTPX.
The `train` extra adds verl 0.9.0, pandas, PyArrow, and PyYAML; `restorebench` adds
RestoreBench, pandapower 3.4.0, and Pydantic for the environment service.
Set `OPENAI_API_KEY` in the shell when the chat endpoint requires authentication.

## Evaluate an ANM scenario

Provide a running OpenAI-compatible chat endpoint and its model identifier.
The endpoint must accept function tools and the sampling extensions sent by
`SamplingConfig`, including `top_k` and `chat_template_kwargs`.

```sh
python3 scripts/evaluate.py --base-url http://127.0.0.1:8000/v1 \
  --model MODEL --scenario-id scenario_s0_l1p2_g1_socnear_min \
  --output-dir outputs/anm
```

Required arguments are `--base-url`, `--model`, `--output-dir`, and either
`--scenario-id` or `--scenarios`. Each scenario produces an indexed `episode_000000.json`
with sampled candidates, independent replays, and selection. `summary.json` lists recovery and paths;
`evaluation.log` records progress. Defaults are three candidates, eight logical
steps per candidate, and two retries after each rejected proposal.

## Scenario inputs

ANM IDs encode `scenario_s<seed>_l<load multiplier>_g<generation multiplier>_soc<mode>`.
Use `p` for a decimal point and `native`, `near_min`, or `near_max` for SoC mode.
Optional suffixes such as `__r90-gp80-sp50-v05` specify integer percentages for
branch ratings, generator P capacity, storage P bounds, and the voltage half-band.
The voltage half-band sets non-slack voltage limits to `1 +/- v_band`.

For catalogs, evaluation accepts JSON or JSONL lists of IDs/records, or a JSON
object keyed by `scenarios` or the requested `--split`. Dataset building accepts
JSON records in a list, a `scenarios` list, or a requested split list. For example,
save this as `scenarios.json`:

```json
{
  "train": [{"id": "scenario_s0_l1p2_g1_socnear_min"}],
  "held_out": [{"id": "scenario_s1_l1_g1_socnative"}]
}
```

Records use `id` or `scenario_id`. Instead of encoding an ID, ANM records can supply
`source_seed`, `load_mul`, `gen_mul`, and `soc_pert`, optionally under `perturbation`.
Frozen records supply both `P_load` and `P_pot` as device-ID-to-MW mappings covering
all loads and non-slack generators. Optional `initial_P_set`, `initial_Q_set`, and
`initial_soc` mappings set controls and storage state. Q uses MVAr; SoC uses simulator
units. `limits` accepts `branch_rate_mul`, `gen_p_mul`, `storage_p_mul`, and `v_band`.
Optional `split`, `difficulty`, and `seed`/`rep_seed` fields populate dataset metadata.

## Build data and train

Training uses a GPU environment for verl/vLLM and a local model with a tool chat template:

```sh
python3 -m pip install -e '.[train]'
python3 -m gridrlvr.verl_adapter.build_dataset --input scenarios.json \
  --split train --output data/train.parquet
python3 -m gridrlvr.verl_adapter.build_dataset --input scenarios.json \
  --split held_out --output data/validation.parquet
python3 scripts/train.py --model-path /path/to/model --train-data data/train.parquet \
  --val-data data/validation.parquet --scenario-file scenarios.json \
  --checkpoint-dir checkpoints
```

Parquet rows contain `prompt`, `agent_name`, `reward_model`, and `extra_info` with scenario/replicate IDs.
The launcher selects the loop and reward hook, logs to `training.log`, and forwards verl settings. `--config`
selects training YAML; explicit launcher flags override it, and positional Hydra
overrides take final precedence. `--print-command` prints without launching.

## RestoreBench service

Supply a corpus with `manifest.json`, `lean/<scenario_id>.json`, and full networks for RestoreBench.

```sh
python3 -m pip install -e '.[restorebench]'
python3 -m gridrlvr.env.restorebench --corpus demo=/path/to/corpus
```

In another shell, evaluate with `--environment restorebench --corpus demo
--env-url http://127.0.0.1:8737` added to the evaluation command and a corpus scenario
ID. Dataset building accepts the same environment/corpus/URL options; each input
record may supply `corpus`. Train with `--environment restorebench --env-url URL`.

## Configuration and layout

`configs/train.yaml` defines data/model paths, optimization, rollout budgets, and
checkpoints. `anm_loop.yaml` defines the scenario catalog, gate, and reward weights;
`restorebench_loop.yaml` defines the service URL, corpus fallback, and proposal rewards.
All YAML options have inline descriptions. `GRIDRLVR_SCENARIO_FILE` and
`GRIDRLVR_RESTOREBENCH_ENV_URL` set loop inputs. `GRIDRLVR_AUDIT_DIR` enables rollout
JSON records; `GRIDRLVR_AUDIT_LEVEL=full` includes tokens/messages. Service options
`GRIDRLVR_RESTOREBENCH_TOOL_TOKEN_BUDGET` (default 2048, nonpositive disables) and
`GRIDRLVR_RESTOREBENCH_TOPOLOGY_FORMAT` (`table` or `records`) control diagnostic output.
Client option `GRIDRLVR_RESTOREBENCH_FORCE_TOOL_PROTOCOL=0` disables added response
instructions; `GRIDRLVR_RESTOREBENCH_PROMPT_OVERFLOW=error` makes training reject overflow.

`gridrlvr/env/` contains simulators, checks, scenarios, and HTTP tools; `agent/`
contains sampling and replay selection; `verl_adapter/` contains data, loops, and
reward hooks. `scripts/` holds launchers. `patches/` contains optional verl patches:
`python3 scripts/apply_verl_patches.py VERL_SOURCE` applies them (`--check` only checks).
`--reward-dimensions` selects ANM GDPO rewards and requires the reward patch.
The offload patch adds `VERL_FSDP2_CPU_OFFLOAD_PIN_MEMORY` (default on; `0` disables).

MIT. See [LICENSE](LICENSE).
