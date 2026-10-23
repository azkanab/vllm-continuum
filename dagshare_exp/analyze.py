"""Analyze DAGShare experiment results (spec section 5, contract C4).

Usage: python -m dagshare_exp.analyze <results_root> --out <dir>

Definitions:
- TTFT = first_token_t - send_t; e2e latency = finish_t - send_t (ok requests only).
- Agent job completion time = last finish_t - first send_t over a job whose requests all
  finished ok.
- Throughput counts ok, non-warmup requests over [min send_t, max finish_t].
- compute_share / memtime_share: final ledger.jsonl snapshot; C_t / sum C and
  (M_act_t + M_idle_t) / sum(M_act + M_idle). idle_gbs = final M_idle (GB*s).
- jain_service: Jain's index over tenants of x_t = compute_share_t + memtime_share_t,
  J = (sum x)^2 / (n * sum x^2). 1 = equal service, 1/n = one tenant gets everything.
  NaN when there is no ledger (vllm, continuum).
- prefill_recomputes = number of preempt events.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("dagshare_exp.analyze")

TENANTS = ("agent", "chat")
IGNORED_TENANTS = ("warmup",)  # deploy/warmup.py traffic, not part of the experiment
AGENT_TENANT = "agent"
CHAT_TENANT = "chat"
OK_STATUSES = {"ok", "success", "finished", "completed", "done"}
PRESET_ORDER = ["vllm", "continuum", "A0", "A1", "A2", "A3", "A4", "A5"]
RETENTION_ORDER = ["none", "ttl", "until_done"]
ID_COLS = ["run", "preset", "retention", "retention_kind", "seed"]
COMPARISONS = [
    ("A2_vs_A3", "A2", "A3", "ttl"),
    ("A2_vs_A3", "A2", "A3", "until_done"),
    ("continuum_vs_A3", "continuum", "A3", "ttl"),
    ("A0_vs_A3", "A0", "A3", "until_done"),
]
STANDARD_METRICS = [
    "n_requests", "n_unfinished", "duration_s", "throughput_req_s", "throughput_tok_s",
    "n_jobs", "job_ct_p50", "job_ct_p99", "job_ct_mean", "jain_service",
    "n_pin", "n_unpin", "n_preempt", "n_release", "prefill_recomputes",
    *[f"{m}_{t}" for t in TENANTS for m in (
        "n_done", "ttft_p50", "ttft_p99", "e2e_p50", "e2e_p99",
        "compute_share", "memtime_share", "idle_gbs", "S_final")],
]
# Validated categorical slots (dataviz reference palette, light mode).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
TENANT_COLORS = {AGENT_TENANT: SERIES_COLORS[0], CHAT_TENANT: SERIES_COLORS[1]}
RETENTION_COLORS = dict(zip(RETENTION_ORDER, SERIES_COLORS))
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def parse_run_name(name: str) -> dict | None:
    """Split '<preset>_<retention>_seed<k>' (retention may contain underscores)."""
    if "_seed" not in name:
        return None
    head, seed = name.rsplit("_seed", 1)
    if not seed.isdigit() or "_" not in head:
        return None
    m = re.match(r"^(?P<preset>.+?)_(?P<ret>(?:none|ttl|until_done).*)$", head)
    preset, retention = (m["preset"], m["ret"]) if m else head.split("_", 1)
    return {"run": name, "preset": preset, "retention": retention,
            "retention_kind": retention_kind(retention), "seed": int(seed)}


def retention_kind(token: str) -> str:
    for kind in ("until_done", "ttl", "none"):
        if token.startswith(kind):
            return kind
    return token


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("%s:%d: skipping bad JSON line", path, lineno)
    return rows


def load_requests(run_dir: Path) -> pd.DataFrame | None:
    path = run_dir / "requests.jsonl"
    if not path.exists():
        log.warning("%s: no requests.jsonl, skipping request metrics", run_dir.name)
        return None
    df = pd.DataFrame(read_jsonl(path))
    if df.empty:
        return df
    for col in ("send_t", "first_token_t", "finish_t", "prompt_tokens", "output_tokens"):
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan
    if "warmup" in df:
        df = df[~df["warmup"].fillna(False).astype(bool)]
    status = df.get("status", pd.Series("ok", index=df.index)).fillna("").astype(str).str.lower()
    df = df.assign(ok=status.isin(OK_STATUSES) & df["finish_t"].notna())
    return df.assign(ttft=df["first_token_t"] - df["send_t"], e2e=df["finish_t"] - df["send_t"])


def load_ledger(run_dir: Path) -> list[dict] | None:
    path = run_dir / "ledger.jsonl"
    if not path.exists():
        return None
    rows = [r for r in read_jsonl(path) if isinstance(r.get("tenants"), dict)]
    for r in rows:
        r["tenants"] = {t: v for t, v in r["tenants"].items() if t not in IGNORED_TENANTS}
    return rows or None


def load_events(run_dir: Path) -> pd.DataFrame | None:
    path = run_dir / "events.jsonl"
    if not path.exists():
        return None
    return pd.DataFrame(read_jsonl(path))


def pct(values, q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(np.percentile(arr, q)) if arr.size else math.nan


def jain(values) -> float:
    x = np.asarray(values, dtype=float)
    if x.size == 0 or np.isnan(x).any() or not (x ** 2).sum():
        return math.nan
    return float(x.sum() ** 2 / (x.size * (x ** 2).sum()))


def request_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_requests": 0, "n_unfinished": 0}
    if "tenant" not in df:
        df = df.assign(tenant=None)
    out: dict = {"n_requests": len(df), "n_unfinished": int((~df["ok"]).sum())}
    ok = df[df["ok"]]
    tenants = sorted(set(TENANTS) | set(df["tenant"].dropna().unique()))
    for t in tenants:
        sub = ok[ok["tenant"] == t]
        out[f"n_done_{t}"] = len(sub)
        for q in (50, 99):
            out[f"ttft_p{q}_{t}"] = pct(sub["ttft"], q) if len(sub) else math.nan
            out[f"e2e_p{q}_{t}"] = pct(sub["e2e"], q) if len(sub) else math.nan
    out.update(job_metrics(df))
    if len(ok):
        window = ok["finish_t"].max() - ok["send_t"].min()
        out["duration_s"] = float(window)
        out["throughput_req_s"] = len(ok) / window if window > 0 else math.nan
        out["throughput_tok_s"] = ok["output_tokens"].sum() / window if window > 0 else math.nan
    else:
        out.update(duration_s=math.nan, throughput_req_s=math.nan, throughput_tok_s=math.nan)
    return out


def job_metrics(df: pd.DataFrame) -> dict:
    nan = {"n_jobs": 0, "job_ct_p50": math.nan, "job_ct_p99": math.nan, "job_ct_mean": math.nan}
    if "job_id" not in df:
        return nan
    agent = df[df["tenant"] == AGENT_TENANT]
    if agent.empty:
        return nan
    g = agent.groupby("job_id")
    jobs = pd.DataFrame({"all_ok": g["ok"].all(), "start": g["send_t"].min(),
                         "end": g["finish_t"].max()})
    ct = (jobs["end"] - jobs["start"])[jobs["all_ok"]]
    if ct.empty:
        return nan
    return {"n_jobs": len(ct), "job_ct_p50": pct(ct, 50), "job_ct_p99": pct(ct, 99),
            "job_ct_mean": float(ct.mean())}


def ledger_metrics(rows: list[dict]) -> dict:
    final = rows[-1]["tenants"]
    tenants = sorted(set(TENANTS) | set(final))
    get = lambda t, k: float(final.get(t, {}).get(k, 0.0) or 0.0)
    c_tot = sum(get(t, "C") for t in final)
    m_tot = sum(get(t, "M_act") + get(t, "M_idle") for t in final)
    out: dict = {}
    for t in tenants:
        out[f"compute_share_{t}"] = get(t, "C") / c_tot if c_tot else math.nan
        out[f"memtime_share_{t}"] = (get(t, "M_act") + get(t, "M_idle")) / m_tot if m_tot else math.nan
        out[f"idle_gbs_{t}"] = get(t, "M_idle") if t in final else math.nan
        out[f"S_final_{t}"] = get(t, "S") if t in final else math.nan
    out["jain_service"] = jain([out[f"compute_share_{t}"] + out[f"memtime_share_{t}"]
                                for t in final])
    return out


def event_metrics(ev: pd.DataFrame) -> dict:
    counts = ev["event"].value_counts() if "event" in ev else pd.Series(dtype=int)
    out = {f"n_{e}": int(counts.get(e, 0)) for e in ("pin", "unpin", "preempt", "release")}
    out["prefill_recomputes"] = out["n_preempt"]
    return out


def analyze_run(run_dir: Path) -> dict | None:
    ids = parse_run_name(run_dir.name)
    if ids is None:
        return None
    row = dict(ids)
    steps = [
        ("requests", lambda: load_requests(run_dir), request_metrics),
        ("ledger", lambda: load_ledger(run_dir), ledger_metrics),
        ("events", lambda: load_events(run_dir), event_metrics),
    ]
    for name, load, compute in steps:
        try:
            data = load()
            if data is None:
                continue
            row.update(compute(data))
        except Exception as e:  # noqa: BLE001 - one bad run must not stop the rest
            log.warning("%s: %s metrics failed: %s", run_dir.name, name, e)
    return row


def summarize(results_root: Path) -> pd.DataFrame:
    rows = []
    for d in sorted(p for p in Path(results_root).iterdir() if p.is_dir()):
        row = analyze_run(d)
        if row is None:
            log.warning("skipping %s: name is not <preset>_<retention>_seed<k>", d.name)
            continue
        rows.append(row)
    df = pd.DataFrame(rows, columns=ID_COLS) if not rows else pd.DataFrame(rows)
    extra = sorted(c for c in df.columns if c not in ID_COLS and c not in STANDARD_METRICS)
    df = df.reindex(columns=ID_COLS + STANDARD_METRICS + extra)
    df[STANDARD_METRICS + extra] = df[STANDARD_METRICS + extra].astype(float)
    return sort_runs(df) if len(df) else df


def sort_runs(df: pd.DataFrame) -> pd.DataFrame:
    key = {p: i for i, p in enumerate(PRESET_ORDER)}
    rkey = {r: i for i, r in enumerate(RETENTION_ORDER)}
    order = df.assign(_p=df["preset"].map(lambda p: key.get(p, len(key))),
                      _r=df["retention_kind"].map(lambda r: rkey.get(r, len(rkey))))
    sort_cols = ["_p", "preset", "_r", "retention"] + (["seed"] if "seed" in df else [])
    return order.sort_values(sort_cols).drop(columns=["_p", "_r"]).reset_index(drop=True)


def metric_columns(summary: pd.DataFrame) -> list[str]:
    return [c for c in summary.columns
            if c not in ID_COLS and pd.api.types.is_numeric_dtype(summary[c])]


def aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(columns=["preset", "retention", "retention_kind", "n_seeds"])
    metrics = metric_columns(summary)
    g = summary.groupby(["preset", "retention", "retention_kind"], sort=False)
    mean = g[metrics].mean().add_suffix("_mean")
    std = g[metrics].std().add_suffix("_std")
    out = pd.concat([g["seed"].nunique().rename("n_seeds"), mean, std], axis=1)
    cols = ["n_seeds"] + [f"{m}_{s}" for m in metrics for s in ("mean", "std")]
    return sort_runs(out[cols].reset_index())


def bootstrap_diff_ci(a, b, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if not a.size or not b.size:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    diffs = (rng.choice(b, (n_boot, b.size)).mean(axis=1)
             - rng.choice(a, (n_boot, a.size)).mean(axis=1))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def comparisons(summary: pd.DataFrame) -> pd.DataFrame:
    """Rows of mean(B) - mean(A) with a seed-bootstrap 95% CI; exploratory (few seeds)."""
    cols = ["comparison", "retention_kind", "preset_a", "retention_a", "preset_b", "retention_b",
            "metric", "mean_a", "mean_b", "diff_b_minus_a", "ci95_low", "ci95_high",
            "n_a", "n_b", "note"]
    if summary.empty:
        return pd.DataFrame(columns=cols)
    metrics = metric_columns(summary)
    out = []
    for name, pa, pb, kind in COMPARISONS:
        a = summary[(summary["preset"] == pa) & (summary["retention_kind"] == kind)]
        b = summary[(summary["preset"] == pb) & (summary["retention_kind"] == kind)]
        if a.empty or b.empty:
            log.warning("comparison %s (%s): missing runs for %s or %s", name, kind, pa, pb)
            continue
        for m in metrics:
            va, vb = a[m].dropna(), b[m].dropna()
            lo, hi = bootstrap_diff_ci(va, vb)
            ma = float(va.mean()) if len(va) else math.nan
            mb = float(vb.mean()) if len(vb) else math.nan
            out.append({"comparison": name, "retention_kind": kind,
                        "preset_a": pa, "retention_a": ",".join(sorted(a["retention"].unique())),
                        "preset_b": pb, "retention_b": ",".join(sorted(b["retention"].unique())),
                        "metric": m, "mean_a": ma, "mean_b": mb, "diff_b_minus_a": mb - ma,
                        "ci95_low": lo, "ci95_high": hi, "n_a": len(va), "n_b": len(vb),
                        "note": "exploratory (few seeds)"})
    return pd.DataFrame(out, columns=cols)


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")


def _grouped_bar(agg: pd.DataFrame, metric: str, title: str, ylabel: str, path: Path) -> bool:
    import matplotlib.pyplot as plt
    mcol, scol = f"{metric}_mean", f"{metric}_std"
    if mcol not in agg or agg[mcol].isna().all():
        log.warning("plot %s: no data for %s", path.name, metric)
        return False
    presets = [p for p in dict.fromkeys(agg["preset"])]
    kinds = [k for k in RETENTION_ORDER if k in set(agg["retention_kind"])]
    kinds += [k for k in dict.fromkeys(agg["retention_kind"]) if k not in kinds]
    width = 0.8 / max(len(kinds), 1)
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(presets) + 2), 4))
    for i, kind in enumerate(kinds):
        sub = agg[agg["retention_kind"] == kind].set_index("preset")
        xs = [j + (i - (len(kinds) - 1) / 2) * width for j, p in enumerate(presets) if p in sub.index]
        ys = [sub.loc[p, mcol] for p in presets if p in sub.index]
        es = [sub.loc[p, scol] if scol in sub else 0 for p in presets if p in sub.index]
        ax.bar(xs, ys, width * 0.92, yerr=np.nan_to_num(es), label=kind,
               color=RETENTION_COLORS.get(kind, SERIES_COLORS[(i + 3) % len(SERIES_COLORS)]),
               edgecolor="white", linewidth=1, error_kw={"ecolor": MUTED, "elinewidth": 1})
    ax.set_xticks(range(len(presets)), presets)
    _style(ax, title, ylabel)
    ax.legend(title="retention", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def _memtime_stacked(agg: pd.DataFrame, path: Path) -> bool:
    import matplotlib.pyplot as plt
    tenants = [t for t in TENANTS if f"memtime_share_{t}_mean" in agg]
    sub = agg.dropna(subset=[f"memtime_share_{t}_mean" for t in tenants], how="all")
    if sub.empty or not tenants:
        log.warning("plot %s: no ledger data", path.name)
        return False
    labels = [f"{p}\n{r}" for p, r in zip(sub["preset"], sub["retention"])]
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(labels) + 2), 4))
    bottom = np.zeros(len(sub))
    for t in tenants:
        vals = sub[f"memtime_share_{t}_mean"].fillna(0).to_numpy()
        ax.bar(range(len(sub)), vals, 0.7, bottom=bottom, label=t,
               color=TENANT_COLORS.get(t, SERIES_COLORS[2]), edgecolor="white", linewidth=2)
        bottom += vals
    ax.set_xticks(range(len(sub)), labels, fontsize=8)
    ax.set_ylim(0, 1)
    _style(ax, "Memory-time share per tenant (M_act + M_idle)", "share")
    ax.legend(title="tenant", frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def _ledger_timeseries(results_root: Path, summary: pd.DataFrame, path: Path) -> bool:
    import matplotlib.pyplot as plt
    if summary.empty:
        return False
    seed = int(summary["seed"].min())
    runs = summary[summary["seed"] == seed]
    series = []
    for run in runs["run"]:
        try:
            rows = load_ledger(Path(results_root) / run)
        except Exception as e:  # noqa: BLE001
            log.warning("%s: ledger read failed: %s", run, e)
            rows = None
        if rows:
            series.append((run, rows))
    if not series:
        log.warning("plot %s: no ledger.jsonl in seed %d runs", path.name, seed)
        return False
    ncol = min(4, len(series))
    nrow = math.ceil(len(series) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.6 * nrow), squeeze=False)
    for ax, (run, rows) in zip(axes.flat, series):
        t0 = float(rows[0]["t"])
        tenants = sorted({t for r in rows for t in r["tenants"]},
                         key=lambda t: (t not in TENANT_COLORS, t))
        for i, t in enumerate(tenants):
            xs = [float(r["t"]) - t0 for r in rows if t in r["tenants"]]
            ys = [r["tenants"][t].get("S", math.nan) for r in rows if t in r["tenants"]]
            ax.plot(xs, ys, linewidth=2, label=t,
                    color=TENANT_COLORS.get(t, SERIES_COLORS[(i + 2) % len(SERIES_COLORS)]))
        _style(ax, run, "S")
        ax.set_xlabel("time since start (s)", color=MUTED)
    for ax in list(axes.flat)[len(series):]:
        ax.set_visible(False)
    axes.flat[0].legend(title="tenant", frameon=False)
    fig.suptitle(f"Ledger S over time per tenant (seed {seed})", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def make_plots(results_root: Path, summary: pd.DataFrame, agg: pd.DataFrame,
               out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    out_dir.mkdir(parents=True, exist_ok=True)
    plots = [
        ("chat_ttft_p99.png", lambda p: _grouped_bar(
            agg, f"ttft_p99_{CHAT_TENANT}", "Chat p99 time to first token", "seconds", p)),
        ("agent_job_completion.png", lambda p: _grouped_bar(
            agg, "job_ct_mean", "Agent job completion time (mean)", "seconds", p)),
        ("jain_service.png", lambda p: _grouped_bar(
            agg, "jain_service", "Jain index over tenants (compute + memory-time share)",
            "Jain index", p)),
        ("memtime_share.png", lambda p: _memtime_stacked(agg, p)),
        ("ledger_S.png", lambda p: _ledger_timeseries(results_root, summary, p)),
    ]
    made = []
    for name, fn in plots:
        path = out_dir / name
        try:
            if fn(path):
                made.append(path)
        except Exception as e:  # noqa: BLE001
            log.warning("plot %s failed: %s", name, e)
    return made


def run(results_root: Path, out_dir: Path, plots: bool = True) -> dict:
    """Write summary.csv, aggregate.csv, comparisons.csv and plots; return the frames."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(Path(results_root))
    agg = aggregate(summary)
    comp = comparisons(summary)
    summary.to_csv(out_dir / "summary.csv", index=False)
    agg.to_csv(out_dir / "aggregate.csv", index=False)
    comp.to_csv(out_dir / "comparisons.csv", index=False)
    made = make_plots(Path(results_root), summary, agg, out_dir / "plots") if plots else []
    return {"summary": summary, "aggregate": agg, "comparisons": comp, "plots": made}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results_root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    res = run(args.results_root, args.out, plots=not args.no_plots)
    log.info("%d runs, %d configs, %d comparison rows, %d plots -> %s",
             len(res["summary"]), len(res["aggregate"]), len(res["comparisons"]),
             len(res["plots"]), args.out)


if __name__ == "__main__":
    main()
