#!/usr/bin/env python3
"""Sequence repro: run several shapes in ONE process like the checker does."""
import os
import sys

import torch
import torch_npu

torch.npu.config.allow_internal_format = False
torch.npu.set_device(int(os.environ.get("ASCEND_DEVICE_ID", "0")))

import custom_ops_lib  # noqa: E402

g = torch.Generator(device="cpu")


def run_case(B, H, seed):
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
    err_y = (y.float() - gy.float()).abs()
    rel = err_y / gy.float().abs().clamp(min=1e-12)
    bad = (err_y > 1e-3) & (rel > 1e-3)
    nbad = bad.sum().item()
    err_r = (res.float() - gr.float()).abs()
    nbad_r = (err_r > 1e-3).sum().item()
    print(f"{B}x{H} seed={seed}: y_bad={nbad} res_bad={nbad_r} "
          f"max_y_err={err_y.max().item():.4e}")
    del x, r, w, y, res, R, ms, rms, gy, gr


run_case(32, 4096, 101)
run_case(256, 1024, 102)
run_case(1, 4096, 103)
run_case(1997, 3037, 104)
run_case(2048, 4096, 105)
print("done")
