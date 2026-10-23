# DAGShare: implementation and architecture plan for parallel subagents

## Context
The governing spec is **`milestone3/CLAUDE.md`** (not `CLAUDE_sectionV_update.md`). It asks for a `dagshare` scheduling policy inside a fork of `Hanchenli/vllm-continuum`, plus experiment tooling. The research question: does charging idle (pinned) KV (beta > 0) make sharing fairer than charging only active memory (beta = 0)?

Two new requirements:
1. Runs happen on **Civo Kubernetes with a GPU node**, not RunPod.
2. **Any teammate** can reproduce a run with scripts.

The work is split into independent workstreams, each owned by a subagent. They can run in parallel because every boundary between them is fixed below as a contract. This file is the shared architecture document. **The first action after approval is to write it into the repos:**
- team repo: `docs/IMPLEMENTATION_PLAN.md`, linked from `docs/architecture.html`
- fork: `engine/docs/dagshare/IMPLEMENTATION.md`, which every subagent prompt points to

## Facts already verified
- The fork is based on **vLLM v0.10.2**: `block_pool.py` and `arg_utils.py` match the v0.10.2 files exactly; `kv_cache_manager.py` is modified by Continuum.
  - So the image is `FROM vllm/vllm-openai:v0.10.2` with the fork's Python package copied over the installed `vllm/`, keeping the compiled `.so` files. There is no CUDA build.
  - Stream E must confirm that Continuum changed no `csrc/` (C++/CUDA) files relative to v0.10.2.
- `engine/` is a clone on branch `dagshare`, with uncommitted step-1 plumbing:
  - `tenant_id` added to the 3 request classes in `protocol.py` and to `v1/request.py`, defaulting to `"default"`.
  - `"dagshare"` added to `SchedulerPolicy`.
  - These get committed first, as part of S0.
- Continuum hook sites in `vllm/v1/core/sched/scheduler.py`:
  - Policy mapping in `__init__`, around l.117.
  - Running-loop preemption, around l.374.
  - Waiting peek at l.468 and pops at l.483/497.
  - Preemption when nothing is running, l.634.
  - `_free_blocks` pin decision, l.1358.
  - `unpin_requests_regular()` call at the start of `schedule()`, l.276.
  - `add_request`, l.1269, which calls `tool_call_estimator.request_arrives`.
- `unpin_request` frees KV immediately, so a logical unpin is also the physical free.
- `ToolCallEstimator.set_up_pin` (`vllm/v1/core/estimate_with_func.py`) returns 2.0 only if `this_func_call` is set and that tool's average time is ≤ 2 s.
  - `this_func_call` comes **from the client** in `extra_args`, so the load generator must send it, or nothing is ever pinned.
- Scheduler tests without a GPU: `tests/v1/core/utils.py::create_scheduler(model="facebook/opt-125m", num_blocks=…)`.
- Local tools: `kubectl` and `helm` are installed; `docker` is an alias for podman; there is no `civo` CLI yet.
- The GitHub fork must be created by the user (`azkanab` → Fork `Hanchenli/vllm-continuum`). Until then, branches stay local.

## Architecture

```
team repo (Fall26-Team-03-IlliniConsensus)
├── engine/  (submodule → azkanab/vllm-continuum, branch dagshare)
│   ├── vllm/v1/core/sched/dagshare.py        [A] pure logic
│   ├── vllm/v1/core/sched/scheduler.py       [B] thin hooks
│   ├── vllm/v1/core/sched/request_queue.py   [B] DagShareRequestQueue
│   ├── tests/dagshare/                        [A][B]
│   ├── dagshare_exp/
│   │   ├── loadgen.py, mock_server.py, traces/, prepare_sharegpt.py   [C]
│   │   ├── configs/{presets,workloads,matrices}/*.yaml                 [S0 schema, C fills]
│   │   ├── analyze.py, analysis.ipynb                                  [D]
│   │   └── deploy/ (Dockerfile, k8s/, scripts, Makefile, RUNBOOK.md)   [E]
│   └── .github/workflows/dagshare-image.yml                            [E]
└── docs/architecture.html (existing)
```

Runtime on Civo: one Kubernetes **Job per run** (preset × retention × seed). Its pod has two containers that share an `emptyDir` named `/run` and the results volume:
- **`server`**: `vllm serve` with the preset's env vars and `--scheduling-policy`. It gets 1 GPU and N whole CPUs.
- **`loadgen`**: uses the same image. It waits for `/health`, warms up, runs the workload and writes `requests.jsonl`. When finished it writes `/run/done`; the server wrapper sees that file and exits. Then results are copied to the results PVC under `<matrix>/<preset>_<retention>_seed<k>/`.
- HF weights come from a shared `hf-cache` PVC that is downloaded once by a `prefetch` Job.
- `fetch-results.sh` pulls results to the laptop through a helper pod and `kubectl cp`.

## Contracts (fixed by S0; streams must not change them without the integrator)

### C1. `vllm/v1/core/sched/dagshare.py` public API
This module is pure Python. It imports nothing from vLLM, so it can be unit-tested anywhere.

```python
@dataclass(frozen=True)
class DagShareConfig:            # from_env(env=os.environ) parses spec §6, raises ValueError on bad values
    compute: bool; compute_share: Literal["slot","tokens"]
    alpha: float; beta: float
    fit_check: bool; fit_estimate: Literal["margin","p90"]; fit_margin_blocks: int
    delta: float; charge_mode: Literal["measured","predicted","predicted_corrected"]
    evict_overserved: bool; log_interval_ms: int; run_output_dir: str

@dataclass(frozen=True)
class RetentionRule:             # from_env(policy: str, env=os.environ)
    kind: Literal["none","ttl","until_done"]; seconds: float; fast_tools_only: bool
    def pin_seconds(self, *, this_func_call: str|None, tool_avg_s: float|None, is_last_step: bool|None) -> float

@dataclass(frozen=True)
class ReqView:                   # built by scheduler.py from Request + kv_cache_manager
    request_id: str; tenant_id: str; job_id: str|None; arrival_time: float
    blocks_held: int; scheduled_tokens: int = 0

@dataclass(frozen=True)
class CandidateView:
    request_id: str; tenant_id: str; job_id: str|None; arrival_time: float
    blocks_needed_now: int; blocks_predicted: int

@dataclass(frozen=True)
class PickResult:
    request_id: str|None; tenant_id: str|None; reason: str   # "lowest" | "slack" | "blocked"

class TenantLedger:
    def __init__(self, cfg: DagShareConfig, total_slots: int, total_kv_blocks: int, bytes_per_block: int): ...
    def update(self, now: float, running: Sequence[ReqView], pinned: Sequence[ReqView]) -> None
    def score(self, tenant: str) -> float
    def pick(self, candidates: Sequence[CandidateView], free_blocks: int, now: float) -> PickResult
    def mark_released(self, tenant: str, now: float) -> None
    def preemption_order(self) -> list[str]                  # highest S first
    def snapshot(self) -> dict                               # per tenant C, M_act, M_idle, S

class RunLogger:                 # files under cfg.run_output_dir; never raises into the scheduler
    def __init__(self, cfg: DagShareConfig): ...
    def write_config(self, info: dict) -> None               # config.json
    def maybe_log_ledger(self, now: float, ledger: TenantLedger, stats: dict) -> None   # ledger.jsonl
    def event(self, **fields) -> None                        # events.jsonl
```
- **Units:** time is in seconds (`time.time()`); GB = blocks × bytes_per_block / 1e9.
- **Normalizers:** T_C = `max_num_running_reqs` and T_M = total KV GB.
- **Ledger formula:** S = C/T_C + α·M_act/T_M + β·M_idle/T_M.
- **M_idle:** every pinned request adds GB × dt, including the time the job's next request waits in the queue (spec §4.3).
- **Fit check:** `fit_check=False` (A0) means no DAGShare fit check; vLLM's own `allocate_slots` still guards memory. `fit_check=True` means prompt blocks − prefix hits + margin must fit in free blocks, with pinned blocks counted as used.

### C2. HTTP request fields (loadgen → server)
OpenAI `/v1/completions` or `/v1/chat/completions` bodies carry these extra top-level fields:
- `tenant_id`, `job_id`
- `is_last_step`
- `this_func_call`: the tool this turn will call, or `null` on the last turn
- `last_func_call`: the tool that just returned

### C3. Agent trace JSONL (one job per line)
`{"job_id": str, "source": "swebench"|"synthetic", "turns": [{"prompt_tokens": int, "output_tokens": int, "tool_name": str|null, "tool_time_s": float, "tool_output_tokens": int}]}`

`tool_name` is new compared with the spec. Continuum needs it, through `this_func_call`.

### C4. Output files (all JSONL; `t` = unix seconds)
- **`requests.jsonl`**: `tenant, job_id, turn, send_t, first_token_t, finish_t, prompt_tokens, output_tokens, status`.
- **`ledger.jsonl`**:
  - `t`
  - `tenants: {id: {C, M_act, M_idle, S}}`
  - `gb_running`, `gb_pinned`
  - `n_running`, `n_waiting`, `n_pinned`
- **`events.jsonl`**:
  - `t, event, request_id, job_id, tenant_id, gb, reason`
  - `event` is one of `release`, `pin`, `unpin` or `preempt`.
- **`config.json`**: every env setting, the policy, model and git commit hash (spec §4.6). Deploy also adds the image digest, GPU name and CPU affinity.

### C5. Config YAML
- **Preset**: `configs/presets/<name>.yaml` = `{name, policy, retention, env: {DAGSHARE_*: …}}`.
- **Workload**: `configs/workloads/<name>.yaml` = `{model, max_model_len, gpu_memory_utilization, duration_s, warmup_s, agent: {traces, jobs_in_flight, tool_time_scale}, chat: {source, rate_rps}, tenants: [...]}`.
- **Matrix**: `configs/matrices/<name>.yaml` = `{presets: [...], retentions: [...], seeds: [...], workload}`.
- The `run.sh` script and `analyze.py` consume these as they are.

## Workstreams

| ID | Owner | Files it owns (only these) | Depends on | Done when |
|---|---|---|---|---|
| S0 | integrator (me), first, about 1 h | commit step-1 plumbing; `dagshare.py` stub with C1 signatures raising `NotImplementedError`; `configs/` examples (C5); `docs/dagshare/IMPLEMENTATION.md` (this file) | — | stubs import cleanly; contracts committed |
| A | subagent "engine-core" | `vllm/v1/core/sched/dagshare.py`, `tests/dagshare/test_{config,ledger,pick,retention,logger}.py` | S0 | all C1 behaviour implemented; `pytest tests/dagshare` passes on a laptop |
| B | subagent "engine-hooks" | `scheduler.py`, `request_queue.py`, `estimate_with_func.py` (expose tool avg only), `tests/dagshare/test_scheduler_hooks.py` | S0 (uses C1 stubs/fakes) | dagshare branches at every hook site; continuum and fcfs code paths unchanged; retention routed through `RetentionRule`; tests pass with `create_scheduler` |
| C | subagent "workload" | `dagshare_exp/{loadgen.py, mock_server.py, prepare_sharegpt.py, traces/make_synthetic.py, traces/convert_mini_swe.py, record_agent_traces.md}`, workload YAMLs | S0 (C2, C3, C4) | loadgen runs 60 s against `mock_server.py` locally and writes a valid `requests.jsonl` |
| D | subagent "analysis" | `dagshare_exp/{analyze.py, analysis.ipynb, tests/test_analyze.py, fixtures/}` | S0 (C4, C5) | produces the spec §5 metrics (p50/p99 TTFT and latency, job completion, throughput, compute and memory-time shares, idle GB·s, Jain, pins/unpins/preemptions) with plots per preset |
| E | subagent "deploy" | `dagshare_exp/deploy/**`, `.github/workflows/dagshare-image.yml` | S0 (C4, C5), C's CLI flags | image builds (GitHub Action → GHCR); `make up/prefetch/run/fetch/down` work; RUNBOOK lets a teammate run one preset from scratch |
| I | integrator | merge A → B → C/D/E; README; first Civo smoke run | all | 2-min smoke run per preset on Civo, analyzed |

Each subagent runs with `isolation: "worktree"` on its own branch (`ds/<id>`). It sticks to its file list and must not reformat other files.

### Stream E details (deploy and team scripts)
- **`deploy/Dockerfile`**: `FROM vllm/vllm-openai:v0.10.2`. Copy the fork's `vllm/` Python files into the installed package (`.py` files only) and add `dagshare_exp/`. Set `ENTRYPOINT` to a small wrapper for the server role and the loadgen role.
- **GitHub Action**: build and push `ghcr.io/azkanab/vllm-dagshare:<sha>` and `:dagshare` on each push to `dagshare`. No teammate needs to build locally; `make image` stays as a podman fallback.
- **`deploy/k8s/`** (rendered by a script with `envsubst`, with no Helm dependency):
  - `namespace.yaml`, `hf-cache-pvc.yaml`, `results-pvc.yaml`
  - `secret-hf.yaml.tmpl` for `HF_TOKEN`, since the Llama 3.1 model is gated
  - `prefetch-job.yaml`
  - `run-job.yaml.tmpl` for the two-container pod: a GPU limit, Guaranteed QoS with integer CPUs, and a toleration for the GPU pool taint
  - `fetch-pod.yaml`
- **Scripts in `deploy/`**, wrapped by a `Makefile`:
  - `env.example`: `CIVO_TOKEN`, `CIVO_REGION`, `CLUSTER_NAME`, `GPU_SIZE`, `HF_TOKEN`, `IMAGE`, `NAMESPACE=dagshare-$USER`
  - `00-check.sh`: checks that tools, env vars and the kubeconfig are present
  - `cluster-up.sh`: `civo kubernetes create` with a GPU pool, saves the kubeconfig, and installs the GPU operator or device plugin if Civo doesn't preinstall it
  - `cluster-down.sh`
  - `prefetch.sh`
  - `run.sh --preset A3 --retention ttl --seed 1 [--workload smoke]`
  - `run-matrix.sh configs/matrices/required.yaml`: runs the matrix sequentially and resumes by skipping finished result dirs
  - `fetch-results.sh`
  - `logs.sh`
- **Teammates:** each teammate gets their own namespace, either on a shared cluster through the shared kubeconfig or on their own cluster with their own `CIVO_TOKEN`. `RUNBOOK.md` covers both, the expected cost per hour, and the rule to always run `make down`.
- **CPU pinning:** request integer CPUs for both containers (Guaranteed QoS). `taskset` inside each container uses its own affinity mask, and `config.json` records that mask. If Civo nodes run the static CPU manager the cores are exclusive; otherwise this is documented as a limitation.
- **Civo facts:**
  - **Region:** GPU Kubernetes is offered **only in `LON1`**.
  - **GPU size:** default is the 1× A100 40GB (8 vCPU, 56 GB, $1.09/h). The 1× L40S 48GB (12 vCPU, $1.29/h) is the alternative and leaves more CPUs for pinning.
    - Size slugs aren't documented. `cluster-up.sh` looks them up with `civo sizes ls` (GPU Optimized group), and `GPU_SIZE` in `env` can override the choice.
  - **Cluster create:** `civo kubernetes create $CLUSTER_NAME -n 1 -s $GPU_SIZE --cluster-type k3s --region LON1 --create-firewall --firewall-rules 6443 --wait --save --merge --switch`
  - **GPU operator:** not preinstalled, so `cluster-up.sh` installs it with Helm. Chart `nvidia/gpu-operator` v25.10.1 with:
    - `driver.enabled=true`, `toolkit.enabled=false`, `devicePlugin.enabled=true`, `gfd.enabled=true`
    - `operator.defaultRuntime=containerd`, `validator.cuda.runtimeClassName=nvidia`
    - The script then waits until the node reports `nvidia.com/gpu`, and `00-check.sh` runs `nvidia-smi` in a test pod to confirm the driver supports CUDA ≥ 12.8.
    - Single-H100 nodes need an extra kernel ConfigMap (`NVreg_NvLinkDisable=1`). It's off by default.
  - **Pods:** `runtimeClassName: nvidia` and `limits: nvidia.com/gpu: 1`. Add a GPU taint toleration only if `kubectl describe node` shows one.
  - **Storage:** storage class `civo-volume` (RWO); `hf-cache` is 50Gi and `results` is 20Gi.
    - Volumes keep billing after a cluster is deleted, so `cluster-down.sh` deletes the PVCs and lists any leftover volumes with `civo volume ls`.
  - **Results:** they stay on the PVC and are pulled with `fetch-results.sh`. Upload to Civo object storage (`objectstore.lon1.civo.com`, S3-compatible) is optional and off by default.
  - **Account:** new accounts get $250 credit. Quotas are account-wide, so RUNBOOK tells each teammate to request a GPU quota increase early.
  - **CPU pinning:** assume no static CPU manager on Civo k3s. Use Guaranteed QoS with integer CPUs, one run pod per GPU node, and `taskset` inside each container. Write this up as a known limit in README.
  - **Install:** the `civo` CLI is installed by `00-check.sh` hints (`brew install civo`); `civo apikey save` uses `CIVO_TOKEN`.

## Order and parallelism
1. **S0:** I commit the contracts.
2. **Parallel:** launch A, B, C, D and E as 5 worktree subagents in one message. B codes against the C1 stub and fakes until A merges.
3. **Merge:** A, then B (re-run the B tests against the real A), then C, D and E.
4. **Smoke test:** `make up && make prefetch && make run PRESET=… WORKLOAD=smoke` for vllm, continuum and A0–A3 (each under ttl and until_done), then `make fetch` and `analyze.py`.
5. **Hand-off:** the team runs the full spec §7 matrix with `run-matrix.sh`: vllm-none, continuum-ttl and A0–A3 × {ttl, until_done}, so 10 configurations × 3 seeds.

## Verification
- **Laptop, no GPU:**
  - `pytest tests/dagshare -q`
  - `pytest tests/v1/core/test_scheduler.py -q` still passes, so the baselines are untouched
  - loadgen runs against `mock_server.py`
  - `analyze.py` runs on the fixtures
  - `make render` produces valid manifests (`kubectl apply --dry-run=client`)
- **Baseline identity:** continuum with default settings produces the same sequence of pin and unpin events before and after B, on a small synthetic replay.
- **Civo smoke:**
  - each preset finishes a 2-minute run
  - `ledger.jsonl` keeps growing
  - `events.jsonl` shows pins held during tool waits
  - A3 with `KV_RETENTION=none` matches A2
- **Teammate test:** a second person follows only `RUNBOOK.md` and completes one run.
