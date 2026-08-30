#!/usr/bin/env python3
"""First-launch corruption: per-row pattern + double-launch test."""
import os
import torch
import torch_npu

torch.npu.config.allow_internal_format = False
torch.npu.set_device(int(os.environ.get("ASCEND_DEVICE_ID", "0")))

import custom_ops_lib  # noqa: E402

g = torch.Generator(device="cpu")


def run_case(B, H, seed, verbose_rows=0):
    g.manual_seed(seed)
    lo, hi = -1.0, 1.0
    x = (torch.rand(B, H, generator=g) * (hi - lo) + lo).to(torch.float16).npu()
    r = (torch.rand(B, H, generator=g) * (hi - lo) + lo).to(torch.float16).npu()
    w = (torch.rand(H, generator=g) * 2.0).clamp(min=0.01).to(torch.float16).npu()
    y, res = custom_ops_lib.fused_add_rmsnorm(x, r, w, 1e-6)
    R = r.float() + x.float()
    ms = torch.mean(R * R, dim=-1, keepdim=True)
    rms = torch.sqrt(ms + 1e-6)
    gy = ((R / rms) * w.float()).to(torch.float16)
    gr = R.to(torch.float16)
    err = (y.float() - gy.float()).abs()
    rel = err / gy.float().abs().clamp(min=1e-12)
    bad = ((err > 1e-3) & (rel > 1e-3)).sum().item()
    badr = ((res.float() - gr.float()).abs() > 1e-3).sum().item()
    print(f"{B}x{H} seed={seed}: y_bad={bad} res_bad={badr}")
    if verbose_rows:
        badmat = ((err > 1e-3) & (rel > 1e-3)).cpu()
        badrows = badmat.sum(dim=1)
        nz = (badrows > 0).nonzero().flatten()
        print("  rows with errors:", nz[:16].tolist(), "..." if len(nz) > 16 else "",
              f"({len(nz)} rows)")
    return bad == 0 and badr == 0


run_case(32, 4096, 101, verbose_rows=1)
run_case(32, 4096, 201, verbose_rows=1)
run_case(2048, 4096, 202)
print("done")
