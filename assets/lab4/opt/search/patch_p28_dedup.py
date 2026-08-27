#!/usr/bin/env python3
# Iter28-dedup: eliminate in-kernel global re-reads of the stress-energy
# arrays (rho, Sx..Szz) in rhs_kernel / rhs_kernel_int.
#
# Evidence: the RHS kernel reads rho[idx] 3x (lines 711/738/946), Sxx..Szz 2x
# (699-700 then 715-720) and Sx/Sy/Sz 2x (404 then 1049-1051) per point. The
# pointers are non-const non-restrict -> nvcc cannot CSE the loads across the
# kernel's stores (aliasing). The full RHS->RK fusion (Iter28) is blocked by
# the sommerfeld boundary damping between RHS and RK (RK uses the damped
# f_rhs), so this in-kernel round-trip elimination is the remaining safe
# sub-target: load each array once into a local and reuse (P2-style,
# bit-exact by construction - same array values).
import hashlib, os, sys, shutil

FORMAL = os.path.expanduser("~/lab4-gpu")
EXCLUDE_DIRS = {"evidence", "build", "__pycache__", "GW250118"}

S_ANCHOR = """    double S = chin1 * (gupxx * Sxx[idx] + gupyy * Syy[idx] + gupzz * Szz[idx] + 
               TWO * (gupxy * Sxy[idx] + gupxz * Sxz[idx] + gupyz * Syz[idx]));"""

S_OUT = """    // Iter28: load stress-energy arrays once and reuse (in-kernel
    // round-trip elimination; bit-exact, same array values)
    double val_rho = rho[idx];
    double val_Sxx = Sxx[idx]; double val_Sxy = Sxy[idx]; double val_Sxz = Sxz[idx];
    double val_Syy = Syy[idx]; double val_Syz = Syz[idx]; double val_Szz = Szz[idx];

    double S = chin1 * (gupxx * val_Sxx + gupyy * val_Syy + gupzz * val_Szz + 
               TWO * (gupxy * val_Sxy + gupxz * val_Sxz + gupyz * val_Syz));"""

SRC_ANCHOR = """    double src_xx = alpn1 * (l_Rxx - EIGHT*PI*Sxx[idx]) - fxx; // fxx is D_i D_j Lap
    double src_yy = alpn1 * (l_Ryy - EIGHT*PI*Syy[idx]) - fyy;
    double src_zz = alpn1 * (l_Rzz - EIGHT*PI*Szz[idx]) - fzz;
    double src_xy = alpn1 * (l_Rxy - EIGHT*PI*Sxy[idx]) - fxy;
    double src_xz = alpn1 * (l_Rxz - EIGHT*PI*Sxz[idx]) - fxz;
    double src_yz = alpn1 * (l_Ryz - EIGHT*PI*Syz[idx]) - fyz;"""

SRC_OUT = """    double src_xx = alpn1 * (l_Rxx - EIGHT*PI*val_Sxx) - fxx; // fxx is D_i D_j Lap
    double src_yy = alpn1 * (l_Ryy - EIGHT*PI*val_Syy) - fyy;
    double src_zz = alpn1 * (l_Rzz - EIGHT*PI*val_Szz) - fzz;
    double src_xy = alpn1 * (l_Rxy - EIGHT*PI*val_Sxy) - fxy;
    double src_xz = alpn1 * (l_Rxz - EIGHT*PI*val_Sxz) - fxz;
    double src_yz = alpn1 * (l_Ryz - EIGHT*PI*val_Syz) - fyz;"""

F_ANCHOR = "    double f = F2o3 * val_trK * val_trK - trA2 - F16*PI*rho[idx] + EIGHT*PI*S;"
F_OUT = "    double f = F2o3 * val_trK * val_trK - trA2 - F16*PI*val_rho + EIGHT*PI*S;"

TRK_ANCHOR = "    trK_rhs[idx] = -chin1 * trK_rhs_val + alpn1 * (F1o3 * val_trK * val_trK + trA2 + FOUR * PI * (rho[idx] + S));"
TRK_OUT = "    trK_rhs[idx] = -chin1 * trK_rhs_val + alpn1 * (F1o3 * val_trK * val_trK + trA2 + FOUR * PI * (val_rho + S));"

HAM_ANCHOR = "    ham_Res[idx] = chin1 * ham_val + F2o3 * val_trK * val_trK - trA2 - F16 * PI * rho[idx];"
HAM_OUT = "    ham_Res[idx] = chin1 * ham_val + F2o3 * val_trK * val_trK - trA2 - F16 * PI * val_rho;"

MOV_ANCHOR = """    movx_Res[idx] = movx_Res[idx] - F2o3 * Kx - F8 * PI * Sx[idx];
    movy_Res[idx] = movy_Res[idx] - F2o3 * Ky - F8 * PI * Sy[idx];
    movz_Res[idx] = movz_Res[idx] - F2o3 * Kz - F8 * PI * Sz[idx];"""

MOV_OUT = """    movx_Res[idx] = movx_Res[idx] - F2o3 * Kx - F8 * PI * val_Sx;
    movy_Res[idx] = movy_Res[idx] - F2o3 * Ky - F8 * PI * val_Sy;
    movz_Res[idx] = movz_Res[idx] - F2o3 * Kz - F8 * PI * val_Sz;"""


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    if len(sys.argv) != 2:
        print("usage: patch_p28_dedup.py <candidate_root>"); sys.exit(2)
    cand = sys.argv[1]
    if os.path.exists(cand):
        print(f"ERROR: {cand} exists"); sys.exit(3)

    for f in ("bssn_rhs_gpu.cu", "bssn_rhs_gpu_int.cu"):
        p = os.path.join(FORMAL, "src", f)
        s = open(p).read()
        print(f"{f}: {sha(p)[:16]}")
    fb = sha(os.path.join(FORMAL, "src", "bssn_rhs_gpu.cu"))
    assert fb.startswith("6ef7bf1f"), "bssn_rhs_gpu.cu drifted (expect 26a deployed)"

    shutil.copytree(FORMAL, cand, ignore=shutil.ignore_patterns(*EXCLUDE_DIRS))
    print(f"copied formal -> {cand}")

    for f in ("bssn_rhs_gpu.cu", "bssn_rhs_gpu_int.cu"):
        p = os.path.join(cand, "src", f)
        s = open(p).read()
        for anchor, out, label in ((S_ANCHOR, S_OUT, "S calc"),
                                   (SRC_ANCHOR, SRC_OUT, "src_*"),
                                   (F_ANCHOR, F_OUT, "f calc"),
                                   (TRK_ANCHOR, TRK_OUT, "trK_rhs"),
                                   (HAM_ANCHOR, HAM_OUT, "ham_Res"),
                                   (MOV_ANCHOR, MOV_OUT, "mov_*")):
            n = s.count(anchor)
            assert n == 1, f"{f} {label} anchor: {n}"
            s = s.replace(anchor, out)
        open(p, "w").write(s)
        assert s.count("val_rho") == 4, f"{f} val_rho: {s.count('val_rho')}"
        assert s.count("rho[idx]") == 1, f"{f} rho[idx] (expect 1 = the val_rho load): {s.count('rho[idx]')}"
        assert s.count("val_Sxx") == 3, f"{f} val_Sxx count: {s.count('val_Sxx')}"
        print(f"patched {f} ({sha(p)[:16]})")
    print("PATCH OK")


if __name__ == "__main__":
    main()
