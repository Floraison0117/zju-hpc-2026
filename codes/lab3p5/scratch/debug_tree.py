#!/usr/bin/env python3
"""Debug helper: run the op on a small shape, print per-row max |y-golden| and
the implied rstd error pattern. Usage: python3 scratch/debug_tree.py B H"""
import os
import sys

import torch
import torch_npu

torch.npu.config.allow_internal_format = False
torch.npu.set_device(int(os.environ.get("ASCEND_DEVICE_ID", "0")))

import custom_ops_lib  # noqa: E402

B = int(sys.argv[1]) if len(sys.argv) > 1 else 8
H = int(sys.argv[2]) if len(sys.argv) > 2 else 1024

g = torch.Generator(device="cpu")
g.manual_seed(42)
lo, hi = -1.0, 1.0
x = (torch.rand(B, H, generator=g) * (hi - lo) + lo).to(torch.float16).npu()
r = (torch.rand(B, H, generator=g) * (hi - lo) + lo).to(torch.float16).npu()
w = (torch.rand(H, generator=g) * 2.0).clamp(min=0.01).to(torch.float16).npu()

y, res = custom_ops_lib.fused_add_rmsnorm(x, r, w, 1e-6)

R = r.float() + x.float()
ms = torch.mean(R * R, dim=-1, keepdim=True)
rms = torch.sqrt(ms + 1e-6)
gy = ((R / rms) * w.float()).to(torch.float16)

y = y.cpu()
res = res.cpu()
gy = gy.cpu()
Rc = R.cpu()

err = (y.float() - gy.float()).abs()
rel = err / gy.float().abs().clamp(min=1e-12)
print("shape", B, "x", H)
print("residual_out max err:", (res.float() - Rc.to(torch.float16).float()).abs().max().item())
for b in range(B):
    print(f"row {b}: max_abs_err={err[b].max().item():.4e}  max_rel={rel[b].max().item():.4e}  "
          f"nbad(>1e-3)={(rel[b] > 1e-3).sum().item()}")
# implied rstd per row from y vs golden (median ratio on large-|y| elements)
for b in range(min(B, 4)):
    m = gy[b].float().abs() > 0.2
    if m.sum() > 8:
        ratio = (y[b].float()[m] / gy[b].float()[m]).median().item()
        print(f"row {b}: y/golden median ratio = {ratio:.6f}")
