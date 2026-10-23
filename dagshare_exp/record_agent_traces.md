# Recording agent traces (SWE-bench with mini-swe-agent)

The agent tenant replays traces in the C3 format, one job per line:

```json
{"job_id": "astropy__astropy-12907", "source": "swebench",
 "turns": [{"prompt_tokens": 2311, "output_tokens": 143, "tool_name": "grep",
            "tool_time_s": 0.41, "tool_output_tokens": 388}, ...]}
```

The last turn has `tool_name: null`, `tool_time_s: 0` and `tool_output_tokens: 0`.
`tool_name` becomes `this_func_call` in the request, which Continuum's fast-tools rule needs.

## Did Continuum release usable traces?

No. `continuum_exp/swebench_llama70B_{continuum,vllm}/` contain only:

- `results.json`: run summary plus `job_durations` (one end-to-end duration per job id, 500 jobs).
- `verification.json`: SWE-bench resolved/unresolved instance ids.

There are no per-turn token counts, tool names or tool times, so they can't be replayed.
The per-job durations (median about 181 s, p99 about 927 s with Llama-3.1-70B) are a sanity check for recorded traces.

Until real traces exist, use the synthetic ones:

```sh
python dagshare_exp/traces/make_synthetic.py --jobs 200 --seed 0   # -> dagshare_exp/traces/synthetic.jsonl
```

These have 5 to 30 turns, with about 80 % fast tools (`ls`, `cat`, `grep`, `sed`, `find`, `git`, mean ≤ 1 s) and slow tools (`python` 6 s, `pytest` 15 s, `pip` 25 s).
Prompts grow by `output_tokens + tool_output_tokens` each turn and are capped at 24k tokens.

## What mini-swe-agent saves

`mini-extra swebench` writes `<output>/<instance_id>/<instance_id>.traj.json` (`trajectory_format: mini-swe-agent-1`) with `messages`:

- `system`
- `user` (task)
- then repeated `assistant` (one bash block) + `user` (`Observation: ...`)

Assistant messages carry `extra.response`, which is the last stream chunk.
The stock format has no timestamps, and with streaming no `usage`. That is why recording goes through the hooks below.

## Record once

1. Install mini-swe-agent with SWE-bench support. Docker must be running, because each instance runs in its own container:
   ```sh
   pip install -e mini-swe-agent
   ```
2. Start a plain vLLM server. Use fcfs, so recording is not affected by the scheduler under test:
   ```sh
   vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8100 --max-model-len 32768
   ```
3. Run SWE-bench through the recording wrapper. Arguments after `--` go to `mini-extra` unchanged:
   ```sh
   python dagshare_exp/traces/convert_mini_swe.py record -- swebench \
       --model-class vllm --model meta-llama/Llama-3.1-8B-Instruct --port 8100 \
       --subset verified --split test --slice 0:100 --workers 8 --output runs/swe_rec
   ```
   The wrapper patches mini-swe-agent in-process (nothing in `mini-swe-agent/` is edited):
   - It adds `stream_options.include_usage`, so each assistant message stores exact `prompt_tokens` and `completion_tokens`.
   - It times `execute_action`, so each observation message stores `tool_time_s`, the wall time of the command in the container.

   Use `--workers` well below the GPU's capacity, so tool times are not inflated by queueing (they are measured around the command only, so model latency is excluded either way).
4. Convert:
   ```sh
   python dagshare_exp/traces/convert_mini_swe.py convert runs/swe_rec --out dagshare_exp/traces/swebench.jsonl
   ```
   `configs/workloads/main.yaml` points at `traces/swebench.jsonl`.

## Conversion rules (`convert_mini_swe.py convert`)

For each assistant message (one model call = one turn):

| field | source when recorded | fallback for old trajectories |
|---|---|---|
| `prompt_tokens` | `extra.response.usage.prompt_tokens` | sum of earlier message lengths (`--tokenizer` or chars/4, plus 4 per message) |
| `output_tokens` | `usage.completion_tokens` | length of the assistant content |
| `tool_name` | first word of the single ```` ```bash ```` block (same rule as Continuum's `ToolCallParser`); `null` on format errors | same |
| `tool_time_s` | `tool_time_s` on the following observation | `--default-tool-time '{"pytest": 15, "*": 1.0}'` (default 1 s; 0 for format errors) |
| `tool_output_tokens` | next `prompt_tokens` − this `prompt_tokens` − this `output_tokens` | length of the messages up to the next assistant message |

- The final turn's tool fields are zeroed.
- Jobs with fewer than `--min-turns` (default 2) model calls are dropped.
- `job_id` is the SWE-bench `instance_id`. The load generator makes it unique per replay as `<tenant>-<seq>-<job_id>`.
