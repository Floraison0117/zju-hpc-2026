#!/usr/bin/env python3
# coding=utf-8
# scratch: hidden-shape robustness tests for FusedAddRmsNorm.
# NOT part of checker/; mirrors the checker's golden + tolerance exactly.
import os
import sys
import torch
import torch_npu

torch.npu.config.allow_internal_format = False
torch.npu.set_device(int(os.environ.get("ASCEND_DEVICE_ID", "0")))
import custom_ops_lib


def golden(x, residual, weight, eps):
    R = residual.float() + x.float()
    ms = torch.mean(R * R, dim=-1, keepdim=True)
    rms = torch.sqrt(ms + eps)
    Y = (R / rms) * weight.float()
    return Y.to(torch.float16), R.to(torch.float16)


def verify_result(real, golden_t):
    tol = 1e-3
    out = real.detach().cpu().to(torch.float64).reshape(-1)
    g = golden_t.detach().cpu().to(torch.float64).reshape(-1)
    eps = 1e-12
    denom = torch.where(g.abs() < eps, torch.tensor(eps), g.abs())
    abs_err = (out - g).abs()
    rel_err = abs_err / denom
    pass_check = (abs_err <= tol) | (rel_err <= tol)
    error_ratio = float((~pass_check).sum().item()) / g.numel()
    return error_ratio <= 0.0, float(abs_err.max()), float(rel_err.max())


def run_one(name, B, H, eps=1e-6, data_range="S", seed=42):
    lo, hi = {"S": (-1.0, 1.0), "M": (1.0, 10.0), "L": (-1000.0, 1000.0)}[data_range]
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = (torch.rand(B, H, generator=g) * (hi - lo) + lo).to(torch.float16)
    r = (torch.rand(B, H, generator=g) * (hi - lo) + lo).to(torch.float16)
    w = (torch.rand(H, generator=g) * 2.0).clamp(min=0.01).to(torch.float16)
    gy, gres = golden(x, r, w, eps)
    y, res = custom_ops_lib.fused_add_rmsnorm(x.npu(), r.npu(), w.npu(), eps)
    ok_y, ae_y, re_y = verify_result(y.cpu(), gy)
    ok_r, ae_r, re_r = verify_result(res.cpu(), gres)
    status = "PASS" if (ok_y and ok_r) else "FAIL"
    print(f"[{name:22s}] {B}x{H} range={data_range}: {status}  "
          f"y(abs={ae_y:.3e},rel={re_y:.3e}) res(abs={ae_r:.3e},rel={re_r:.3e})")
    return ok_y and ok_r


def main():
    all_ok = True
    cases = [
        # B=1 around the 32B FP16 boundary
        ("B1 H15", 1, 15, 1e-6, "S"),
        ("B1 H16", 1, 16, 1e-6, "S"),
        ("B1 H17", 1, 17, 1e-6, "S"),
        ("B1 H31", 1, 31, 1e-6, "S"),
        ("B1 H32", 1, 32, 1e-6, "S"),
        ("B1 H33", 1, 33, 1e-6, "S"),
        ("B1 H4095", 1, 4095, 1e-6, "S"),
        ("B1 H4097", 1, 4097, 1e-6, "S"),
        # H around 1024
        ("H1023", 64, 1023, 1e-6, "S"),
        ("H1024", 64, 1024, 1e-6, "S"),
        ("H1025", 64, 1025, 1e-6, "S"),
        # B relative to AIV count (40)
        ("B32", 32, 1024, 1e-6, "S"),
        ("B40", 40, 1024, 1e-6, "S"),
        ("B41", 41, 1024, 1e-6, "S"),
        ("B250", 250, 1024, 1e-6, "S"),   # not divisible by 40
        # larger shapes / chunked path (H > 4096)
        ("B512 H8192", 512, 8192, 1e-6, "S"),
        ("B16 H16384", 16, 16384, 1e-6, "S"),
        ("B4096 H512", 4096, 512, 1e-6, "S"),
        # extreme values
        ("B256 H1024 L", 256, 1024, 1e-6, "L"),
        ("B1997 H3037 L", 1997, 3037, 1e-6, "L"),
        # tiny H
        ("B7 H3", 7, 3, 1e-6, "S"),
        ("B8 H8", 8, 8, 1e-6, "S"),
    ]
    for name, B, H, eps, rng in cases:
        try:
            ok = run_one(name, B, H, eps, rng)
            all_ok &= ok
        except Exception as e:
            print(f"[{name:22s}] EXCEPTION: {e}")
            all_ok = False
    print("ALL PASS" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
