"""Multi-scale profiling helper for Lab5 task2 hotspot location.

Scales
------
- coarse : full OJ-equivalent run (BS2, performance_public.jsonl, 10 reqs).
           Emits per-request TTFT / mean-TPOT / total / token-count / finish-reason,
           plus aggregate wall, generated tokens, throughput, peak memory, and the
           full per-decode-step latency list. No profiler overhead -> reproduces the
           749.8 s number and splits it into prefill vs decode.
- medium : short window (1 req, small prompt): 1 prefill + N decode steps wrapped in
           torch.profiler (CPU+CUDA activity). Emits key_averages sorted by
           self CUDA time, an op-category breakdown (fused dequant-GEMM / H2D copy /
           attention / INT4 unpack / other), CPU total vs GPU active (overlap hint),
           and a chrome trace. Designed to be run under `nsys profile` so the two
           capture together.
- fine   : same short window, NO torch.profiler (ncu wraps the process externally).
           Just runs enough decode steps for ncu to capture the v2 kernel.

The engine config mirrors the deployed ~/lab5/config.yaml + OJ argv:
  int4_hybrid / flash / rebind_batch / async / ring_indexed / compact_active_slots,
  BS2 (coarse) or BS1 (medium/fine), max_sequence_length 2048, seed 0,
  synchronize_metrics=False. CUDA Graph is left at its default (False) so the
  measured path is exactly the current 749.8 s path.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

from hpc101_infer import (
    EngineConfig,
    GenerationRequest,
    InferenceEngine,
    Runner,
    SamplingParams,
)


# ----------------------------- dataset loading ----------------------------- #

def load_requests(path, default_max_new):
    reqs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sp = r.get("sampling_params") or {}
            stop = r.get("stop_token_ids")
            reqs.append(
                GenerationRequest(
                    prompt=r.get("prompt"),
                    input_ids=r.get("input_ids"),
                    max_new_tokens=r.get("max_new_tokens", default_max_new),
                    stop_token_ids=tuple(stop) if stop else None,
                    sampling_params=SamplingParams(**sp),
                )
            )
    return reqs


def mem():
    return (
        torch.cuda.memory_allocated() / 1024**3,
        torch.cuda.memory_reserved() / 1024**3,
    )


def build_engine(model_path, bs, msl, kv_backend="ring_indexed", offload="async"):
    torch.cuda.reset_peak_memory_stats()
    config = EngineConfig(
        dtype=torch.bfloat16,
        device="cuda",
        max_batch_size=bs,
        scheduler_batch_size=bs,
        max_sequence_length=msl,
        seed=0,
        synchronize_metrics=False,
        attention_backend="flash",
        linear_backend="int4_hybrid",
        scheduler_backend="rebind_batch",
        weight_offload=offload,
        kv_cache_backend=kv_backend,
        compact_active_slots=True,
    )
    engine = InferenceEngine.from_pretrained(model_path, config)
    return engine, config


# ------------------------------- scale: coarse ----------------------------- #

def scale_coarse(model_path, public_path, out_dir):
    print("\n===== SCALE 1 (COARSE): full BS2 public run, phase breakdown =====",
          flush=True)
    gc.collect(); torch.cuda.empty_cache()
    t_load0 = time.perf_counter()
    engine, cfg = build_engine(model_path, bs=2, msl=2048, kv_backend="ring_indexed")
    t_load = time.perf_counter() - t_load0
    a0, r0 = mem()
    print(f"[coarse] engine loaded in {t_load:.2f}s  "
          f"alloc={a0:.3f}GiB reserved={r0:.3f}GiB  cuda_graph={cfg.cuda_graph} "
          f"weight_offload={cfg.weight_offload} kv={cfg.kv_cache_backend} "
          f"linear={cfg.linear_backend} attn={cfg.attention_backend}",
          flush=True)

    reqs = load_requests(public_path, 48)
    print(f"[coarse] requests={len(reqs)}  "
          f"prompt_lens={[len(r.input_ids) if r.input_ids else len(r.prompt.split()) for r in reqs]}",
          flush=True)

    runner = Runner(engine)
    t_run0 = time.perf_counter()
    outputs = runner.run(iter(reqs))
    t_run = time.perf_counter() - t_run0
    peak_a = torch.cuda.max_memory_allocated() / 1024**3
    peak_r = torch.cuda.max_memory_reserved() / 1024**3

    # Per-request phase breakdown.
    total_tokens = 0
    total_prefill = 0.0
    all_tpot = []
    per_req = []
    for i, o in enumerate(outputs):
        m = o.metrics
        ntok = o.generated_tokens
        total_tokens += ntok
        ttft = m.prefill_latency_s
        total_prefill += ttft
        de_lats = list(m.decode_latencies_s)
        all_tpot.extend(de_lats)
        mean_tpot = (sum(de_lats) / len(de_lats)) if de_lats else float("nan")
        per_req.append({
            "req": i,
            "prompt_tokens": o.prompt_tokens,
            "n_gen": ntok,
            "ttft_s": ttft,
            "mean_tpot_s": mean_tpot,
            "total_s": m.total_latency_s,
            "n_decode_steps": len(de_lats),
            "finish": o.finish_reason,
        })
        print(f"  req{i}: ptok={o.prompt_tokens:5d} ngen={ntok:3d} ttft={ttft:7.3f}s "
              f"mean_tpot={mean_tpot:7.4f}s total={m.total_latency_s:8.3f}s "
              f"steps={len(de_lats):3d} finish={o.finish_reason}",
              flush=True)

    n_steps = len(all_tpot)
    sum_decode = sum(all_tpot)
    wall = t_run
    throughput = total_tokens / wall if wall > 0 else 0.0
    print(f"\n[coarse] AGGREGATE", flush=True)
    print(f"  wall_run        = {wall:8.3f}s   (OJ elapsed was 749.796s for this set)",
          flush=True)
    print(f"  generated_tokens= {total_tokens}", flush=True)
    print(f"  throughput      = {throughput:.4f} tok/s   (OJ scored 0.426 tok/s)",
          flush=True)
    print(f"  n_decode_steps  = {n_steps}", flush=True)
    print(f"  sum_prefill_TTFT= {total_prefill:8.3f}s  "
          f"({100*total_prefill/wall:.1f}% of wall)" if wall else "", flush=True)
    print(f"  sum_decode_time = {sum_decode:8.3f}s  "
          f"({100*sum_decode/wall:.1f}% of wall)" if wall else "", flush=True)
    if all_tpot:
        print(f"  TPOT min/med/mean/max = "
              f"{min(all_tpot):.4f}/{statistics.median(all_tpot):.4f}/"
              f"{statistics.mean(all_tpot):.4f}/{max(all_tpot):.4f}s",
              flush=True)
    print(f"  peak_alloc={peak_a:.3f}GiB peak_reserved={peak_r:.3f}GiB", flush=True)

    out = {
        "scale": "coarse",
        "wall_run_s": wall,
        "generated_tokens": total_tokens,
        "throughput_toks": throughput,
        "n_decode_steps": n_steps,
        "sum_prefill_ttft_s": total_prefill,
        "sum_decode_time_s": sum_decode,
        "prefill_pct_of_wall": 100 * total_prefill / wall if wall else 0,
        "decode_pct_of_wall": 100 * sum_decode / wall if wall else 0,
        "tpot_min": min(all_tpot) if all_tpot else 0,
        "tpot_median": statistics.median(all_tpot) if all_tpot else 0,
        "tpot_mean": statistics.mean(all_tpot) if all_tpot else 0,
        "tpot_max": max(all_tpot) if all_tpot else 0,
        "peak_alloc_gib": peak_a,
        "peak_reserved_gib": peak_r,
        "cuda_graph": cfg.cuda_graph,
        "weight_offload": cfg.weight_offload,
        "kv_cache_backend": cfg.kv_cache_backend,
        "linear_backend": cfg.linear_backend,
        "per_request": per_req,
        "all_decode_latencies_s": all_tpot,
    }
    p = Path(out_dir) / "coarse_phase_breakdown.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"[coarse] wrote {p}", flush=True)


# ------------------------------- scale: medium ----------------------------- #

_OP_CATEGORIES = [
    ("fused_dequant_gemm", ("fused_dequant_gemm", "triton_", "_fused_")),
    ("h2d_copy", ("aten::copy_", "aten::_to_copy", "aten::copy")),
    ("attention", ("_flash", "flash", "bmm", "softmax", "_attention", "sliding")),
    ("int4_unpack", ("aten::bitwise", "aten::rshift", "aten::and", "aten::to",
                     "aten::view", "aten::reshape", "aten::slice")),
]


def _categorize(name):
    n = name.lower()
    for label, pats in _OP_CATEGORIES:
        if any(p in n for p in pats):
            return label
    return "other"


def scale_medium(model_path, small_path, out_dir, n_decode=12):
    """Short window under torch.profiler. Emit key_averages + category breakdown +
    chrome trace. Also prints CPU-total vs GPU-active as an overlap hint."""
    print("\n===== SCALE 2 (MEDIUM): torch.profiler on 1 prefill + "
          f"{n_decode} decode steps =====", flush=True)
    gc.collect(); torch.cuda.empty_cache()
    engine, cfg = build_engine(model_path, bs=1, msl=2048, kv_backend="ring_indexed")
    print(f"[medium] engine ready cuda_graph={cfg.cuda_graph} "
          f"offload={cfg.weight_offload}", flush=True)

    reqs = load_requests(small_path, 48)
    req = reqs[0]
    runner = Runner(engine)

    # warmup (1 generate) so CUDA caches / autotune are stable, then profile a 2nd.
    print("[medium] warmup generate ...", flush=True)
    _ = runner.run(iter([req]))
    gc.collect(); torch.cuda.empty_cache()

    activities = [torch.profiler.ProfilerActivity.CPU,
                  torch.profiler.ProfilerActivity.CUDA]
    print(f"[medium] profiled run (1 prefill + {n_decode} decode) ...", flush=True)
    t0 = time.perf_counter()
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
    ) as prof:
        # limit decode tokens so the window is small but representative
        req_p = GenerationRequest(
            prompt=req.prompt, input_ids=req.input_ids,
            max_new_tokens=n_decode,
            stop_token_ids=req.stop_token_ids,
            sampling_params=req.sampling_params,
        )
        outs = runner.run(iter([req_p]))
    t1 = time.perf_counter()
    print(f"[medium] profiled window wall = {t1-t0:.3f}s  "
          f"tokens={len(outs[0].token_ids)}", flush=True)

    # key_averages by self CUDA time (self_device_time_total is in MICROSECONDS)
    ka = prof.key_averages(group_by_input_shape=False)
    ka_cuda = sorted(ka, key=lambda e: e.self_device_time_total, reverse=True)
    print("\n[medium] ---- top 30 ops by self CUDA time (values in seconds) ----", flush=True)
    print(f"{'op':55s} {'self_cuda_s':>12s} {'calls':>8s} {'category':>16s}",
          flush=True)
    cat_cuda = {}
    for e in ka_cuda[:30]:
        if e.self_device_time_total <= 0:
            continue
        cat = _categorize(e.key)
        cat_cuda[cat] = cat_cuda.get(cat, 0.0) + e.self_device_time_total
        print(f"{e.key[:55]:55s} {e.self_device_time_total/1e6:12.5f} "
              f"{e.count:8d} {cat:>16s}", flush=True)

    # full category breakdown (all entries, not just top 30) — convert µs -> s
    cat_cuda_full = {}
    cpu_total_us = 0.0
    gpu_total_us = 0.0
    for e in ka:
        if e.self_device_time_total > 0:
            cat = _categorize(e.key)
            cat_cuda_full[cat] = cat_cuda_full.get(cat, 0.0) + e.self_device_time_total
            gpu_total_us += e.self_device_time_total
        if e.self_cpu_time_total > 0:
            cpu_total_us += e.self_cpu_time_total
    gpu_total_s = gpu_total_us / 1e6
    cpu_total_s = cpu_total_us / 1e6
    print("\n[medium] ---- CUDA time by op category (all entries, seconds) ----", flush=True)
    for cat, t_us in sorted(cat_cuda_full.items(), key=lambda kv: -kv[1]):
        t_s = t_us / 1e6
        print(f"  {cat:20s} {t_s:12.5f}s  ({100*t_s/gpu_total_s:5.1f}% of GPU self)"
              if gpu_total_s else f"  {cat:20s} {t_s:12.5f}s", flush=True)
    print(f"\n[medium] GPU self_time_total = {gpu_total_s:.5f}s", flush=True)
    print(f"[medium] CPU self_time_total = {cpu_total_s:.5f}s", flush=True)
    print(f"[medium] overlap hint: GPU_active={gpu_total_s:.5f}s vs "
          f"window_wall={t1-t0:.3f}s  "
          f"(GPU_util_on_window={100*gpu_total_s/(t1-t0):.1f}%)", flush=True)
    print("[medium] if GPU_util << 100%, decode is CPU-launch-bound "
          "(consistent with cuda_graph=False).", flush=True)
    print("[medium] inspect nsys trace for copy-stream vs compute-stream "
          "concurrency (async overlap = H4).", flush=True)

    # chrome trace
    trace_path = Path(out_dir) / "medium_chrome_trace.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"[medium] wrote chrome trace {trace_path}", flush=True)

    # also dump key_averages table as text
    txt_path = Path(out_dir) / "medium_key_averages.txt"
    txt_path.write_text(prof.key_averages(
        group_by_input_shape=True).table(sort_by="self_device_time_total",
                                          row_limit=80))
    print(f"[medium] wrote key_averages {txt_path}", flush=True)


# -------------------------------- scale: fine ------------------------------ #

def scale_fine(model_path, small_path, n_decode=6):
    """No torch.profiler. ncu wraps this process externally. Run a short window so
    the v2 fused dequant-GEMM is launched a handful of times."""
    print(f"\n===== SCALE 3 (FINE): short window for ncu ({n_decode} decode) =====",
          flush=True)
    gc.collect(); torch.cuda.empty_cache()
    engine, cfg = build_engine(model_path, bs=1, msl=2048, kv_backend="ring_indexed")
    print(f"[fine] engine ready cuda_graph={cfg.cuda_graph}", flush=True)
    reqs = load_requests(small_path, 48)
    req = reqs[0]
    runner = Runner(engine)
    # one warmup + one short generation
    _ = runner.run(iter([GenerationRequest(
        prompt=req.prompt, input_ids=req.input_ids, max_new_tokens=2,
        stop_token_ids=req.stop_token_ids, sampling_params=req.sampling_params)]))
    print("[fine] warmup done; launching profiled window ...", flush=True)
    outs = runner.run(iter([GenerationRequest(
        prompt=req.prompt, input_ids=req.input_ids, max_new_tokens=n_decode,
        stop_token_ids=req.stop_token_ids, sampling_params=req.sampling_params)]))
    print(f"[fine] done; tokens={list(outs[0].token_ids)}", flush=True)


# ----------------------------------- main ---------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--small", required=True)
    ap.add_argument("--public", default="")
    ap.add_argument("--scale", required=True,
                    choices=["coarse", "medium", "medium-nsys", "fine"])
    ap.add_argument("--out-dir", default="/tmp/lab5-profile")
    ap.add_argument("--n-decode", type=int, default=12)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"PID={os.getpid()}  scale={args.scale}  out_dir={out_dir}", flush=True)
    print(f"torch={torch.__version__}  device={torch.cuda.get_device_name(0)}  "
          f"free={torch.cuda.mem_get_info()[0]/1024**3:.2f}GiB", flush=True)

    if args.scale == "coarse":
        scale_coarse(args.model, args.public, args.out_dir)
    elif args.scale == "medium":
        scale_medium(args.model, args.small, args.out_dir, n_decode=args.n_decode)
    elif args.scale == "medium-nsys":
        # Same short window as `medium` but WITHOUT torch.profiler, so nsys is the
        # sole CUPTI consumer. nsys wraps this process. Default more decode steps
        # for a richer timeline.
        scale_fine(args.model, args.small, n_decode=max(args.n_decode, 20))
    elif args.scale == "fine":
        scale_fine(args.model, args.small, n_decode=args.n_decode)


if __name__ == "__main__":
    main()
