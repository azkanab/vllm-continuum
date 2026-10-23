# DAGShare on Civo: runbook

This page takes you from nothing to analyzed results for one preset, then the full matrix.
All commands run from `engine/dagshare_exp/deploy/` on your laptop. Every script has `--help`.

**Rule zero: a GPU cluster bills by the hour until you delete it. When you are done, run `make down`.**

## 1. Prerequisites (do these early, some take a day)

| What | Why | How |
|---|---|---|
| Civo account | the cluster | sign up at civo.com; new accounts get **$250 credit** |
| GPU quota increase | new accounts cannot create GPU nodes | Civo dashboard → Quota → request at least 1 GPU in **LON1** (GPU Kubernetes exists only in LON1). Quotas are per account, so every teammate with their own account must ask. |
| Civo API key | scripts use it | Dashboard → Settings → Profile → Security → API key |
| Hugging Face account with Llama 3.1 access | the model is gated | accept the license on https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct, wait for approval, then create a **read** token |
| Laptop tools | | `brew install civo kubectl helm gettext yq` (gettext provides `envsubst`) |

The image is built by GitHub Actions on every push to `dagshare` (`.github/workflows/dagshare-image.yml`) and pushed to `ghcr.io/azkanab/vllm-dagshare:{<sha>,dagshare}`. Nobody needs to build it locally (`make image` is a podman/docker fallback). The first push creates a **private** package: the repo owner should make it public once (GitHub → Packages → vllm-dagshare → Package settings → Change visibility). Otherwise each user sets `GHCR_USER`/`GHCR_TOKEN` (a PAT with `read:packages`) in `.env`.

## 2. One-time setup

```bash
cd engine/dagshare_exp/deploy
cp env.example .env           # fill CIVO_TOKEN and HF_TOKEN; .env is gitignored
make check LOCAL=1            # tools + env vars
```

Choose one of the two options below.

**Own cluster** (you pay, nobody waits for you):
```bash
make up                       # ~10-15 min: creates the k3s GPU node, installs the NVIDIA GPU operator
make check-gpu                # nvidia-smi in a pod, driver must support CUDA >= 12.8
```
`GPU_SIZE` empty means `cluster-up.sh` picks the smallest A100 (else L40S) Kubernetes size from `civo sizes ls`. See it with `./cluster-up.sh --print-size`.

**Shared cluster** (one person ran `make up`; everyone else gets its kubeconfig):
```bash
civo kubernetes config <cluster-name> --region LON1 --save --merge   # needs access to that Civo account
# or: the owner sends the kubeconfig file, you merge it into ~/.kube/config
# in .env: KUBE_CONTEXT=<context name>, leave CIVO_TOKEN empty
make check
```
Each person works in their own namespace (`NAMESPACE=dagshare-$USER`) with their own PVCs and results. There is one GPU, so runs from different people queue: a run pod stays `Pending` until the GPU is free. `make status` shows who is running.

Scripts only talk to `KUBE_CONTEXT` (default: `CLUSTER_NAME`), never to your current kubectl context.

## 3. Download the model (once per namespace)

```bash
make prefetch                 # ~5-10 min; also pulls the ~10 GB image onto the node
```

## 4. Run one smoke preset end to end

```bash
make run PRESET=A3 WORKLOAD=smoke                 # retention from the preset (ttl:2)
make run PRESET=A3 RETENTION=until_done WORKLOAD=smoke
make run PRESET=vllm WORKLOAD=smoke               # baseline, retention none
```
Or directly: `./run.sh --preset A3 --retention ttl --seed 1 --workload smoke`.

What happens: one Kubernetes Job per run. Its pod has a `server` container (`vllm serve` with the preset's settings, 1 GPU, `SERVER_CPUS` whole CPUs) and a `loadgen` container (waits for `/health`, warms up, runs the workload). When the load generator finishes, the server shuts down and the Job completes. The script follows the logs (also saved under `deploy/logs/`) and exits non-zero if the run failed.

Results go to the results PVC under `<matrix>/<preset>_<none|ttl2|until_done60>_seed<k>/`; single runs use matrix name `adhoc`. Each run dir has `requests.jsonl`, `ledger.jsonl`, `events.jsonl`, `config.json`, `server.log`, `loadgen.log`, and `deploy_info.json` (CPU affinity, GPU name, image digest, git SHA).

A smoke run takes about 2 min of load plus 3-5 min of model load.

## 5. Run a matrix

```bash
make matrix MATRIX=smoke      # 10 runs x 1 seed, about 1 h
make matrix MATRIX=required   # 10 configurations x 3 seeds, 20 min each, about 13-15 h
```
The matrix resolves the image tag to a digest once, so every run uses the same image. Runs whose dir already has `requests.jsonl` are skipped, so if it stops (laptop sleep, failure), run the same command again to resume. Failed runs do not stop the matrix; the summary at the end lists them. Rerun one with `make run ... MATRIX_NAME=required`.

Keep the laptop awake (`caffeinate -i make matrix MATRIX=required`). If the laptop disconnects, the current Job still finishes on the cluster.

## 6. Fetch and analyze

```bash
make fetch                    # copies the PVC to engine/results/
cd ../.. && python -m dagshare_exp.analyze results --out results/analysis
```
`./fetch-results.sh --matrix smoke` copies only one matrix.

## 7. Tear down. ALWAYS.

```bash
make fetch                    # first, results live only on the PVC
make down                     # deletes namespace + PVCs + cluster, then lists leftover volumes
make down NS_ONLY=1           # shared-cluster users: delete only your namespace
```
Check the volume list it prints. Civo volumes **keep billing after the cluster is gone**; delete leftovers with `civo volume rm <name>`.

## 8. Costs

| Item | Price |
|---|---|
| 1x A100 40GB node (8 vCPU) | about $1.09/h |
| 1x L40S 48GB node (12 vCPU) | about $1.29/h |
| volumes (hf-cache 50 GB + results 20 GB per namespace) | $0.11/GB-month, about $7.70/month, billed until deleted |

The smoke matrix costs about $1.50; the required matrix about $15-20. The $250 credit covers several full matrices, but only if clusters are deleted after use.

## 9. Make targets

`make check | check-gpu | up | down | image | render | prefetch | run | matrix | fetch | logs | status`.
`make render` only writes manifests to `.rendered/` and validates them; it touches no cluster.

## 10. Troubleshooting

| Symptom | Check / fix |
|---|---|
| `make up` waits forever for `nvidia.com/gpu` | `kubectl --context $CLUSTER_NAME -n gpu-operator get pods`. The driver daemonset compiles the driver (about 10 min). `CrashLoopBackOff` on `nvidia-driver-daemonset`: see its logs; `nvidia-operator-validator` stays `Init` until the driver is ready. |
| Run pod `Pending`, reason `Unschedulable` | GPU busy (another run or teammate: `make status`) or CPU/memory requests too large for the node. Lower `SERVER_CPUS`/`SERVER_MEM` in `.env` (A100 node: 8 vCPU, 56 GB). |
| Pod `ContainerCreating` for minutes | image pull, about 10 GB. Normal on the first run of a node; `make prefetch` pulls it early. |
| `ImagePullBackOff` | the GHCR package is private: make it public, or set `GHCR_USER`/`GHCR_TOKEN`. Or the tag does not exist yet: check the Actions run. |
| prefetch fails with 401/403 | `HF_TOKEN` wrong, or your HF account has not been granted Llama 3.1 access. Then `make prefetch` again. |
| server `OOMKilled` | raise `SERVER_MEM`. CUDA `out of memory` in `server.log`: lower `gpu_memory_utilization` or `max_model_len` in the workload. |
| run fails with `DeadlineExceeded` | the Job's limit is duration + warmup + `DEADLINE_MARGIN_S` (default 1800 s). Raise the margin if model load is slow. |
| loadgen error | `make logs` or `./logs.sh --preset A3 -c loadgen`; partial outputs are kept under `<run>/failed/` on the PVC. |
| driver CUDA < 12.8 in `make check-gpu` | the operator installed an old driver; set `driver.version` via a `--set` in `cluster-up.sh`. |
| GPU node tainted | `kubectl describe node`; set `GPU_TOLERATION_KEY` in `.env`. |

## 11. Known limits

- **CPU pinning:** Civo k3s does not run the static CPU manager, so cores are not exclusive. Pods are Guaranteed QoS with integer CPUs, there is one run pod per GPU node, and each container runs under `taskset` with its own affinity mask, which is recorded in `deploy_info.json` and `loadgen_info.json`.
- Configs (workloads, traces) come from the **image**, so commit and push config changes and wait for the image build before running. Presets are passed as env vars from your local files.
- `RWO` volumes: all pods of a namespace must land on the same node. That holds with the one-node cluster.
- Warmup requests use tenant `warmup`; analysis should ignore that tenant.
