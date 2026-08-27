#!/usr/bin/env python3
# Iter30-pool: re-enable the GPUManager device-memory pool.
#
# Evidence (phase0 ledger 2.5): cudaMalloc+cudaFree = 16.7s over the run
# (223,313 + 222,513 calls, 2.1% of Program Cost); the pool infrastructure in
# gpu_manager.cu exists but is commented out. Re-enabling it with a byte cap
# (bounds retained memory on the 10 GiB MIG slice) and cudaMemset on reuse
# (preserves the allocate-zeroes semantics) converts most malloc/free pairs
# into pool hits.
#
# Risk notes: reuse assumes callers free buffers only after their kernels
# complete (per-stream sync at step boundaries); any race would surface as
# trajectory RMS != 0 in the L1/L2 gates. The pool cap prevents unbounded
# growth on size-mismatched alloc/free pairs.
import hashlib, os, sys, shutil

FORMAL = os.path.expanduser("~/lab4-gpu")
EXCLUDE_DIRS = {"evidence", "build", "__pycache__", "GW250118"}

IMPL_ANCHOR = """struct GPUManager::Impl {
    std::unordered_map<size_t, std::vector<double*>> memory_pool;
    std::mutex pool_mutex;
"""
IMPL_OUT = """struct GPUManager::Impl {
    std::unordered_map<size_t, std::vector<double*>> memory_pool;
    std::mutex pool_mutex;
    size_t pool_bytes = 0;               // Iter30: bytes currently retained
    static constexpr size_t POOL_CAP_BYTES = 1u << 30; // 1 GiB cap (10 GiB MIG)
"""

ALLOC_ANCHOR = """double* GPUManager::allocate_device_memory(size_t num_elements) {
    // std::lock_guard<std::mutex> lock(pimpl->pool_mutex);
    // if (pimpl->memory_pool.find(num_elements) != pimpl->memory_pool.end() && 
    //     !pimpl->memory_pool[num_elements].empty()) {
    //     double* d_ptr = pimpl->memory_pool[num_elements].back();
    //     pimpl->memory_pool[num_elements].pop_back();
    //     // CUDA_CHECK(cudaMemset(d_ptr, 0, num_elements * sizeof(double)));
    //     return d_ptr;
    // }
    double* d_ptr = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_ptr, num_elements * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_ptr, 0, num_elements * sizeof(double)));
    return d_ptr;
}"""

ALLOC_OUT = """double* GPUManager::allocate_device_memory(size_t num_elements) {
    // Iter30: reuse pooled buffers when available (zeroed on reuse to keep
    // the allocate-zeroes semantics); fall back to cudaMalloc on pool miss.
    {
        std::lock_guard<std::mutex> lock(pimpl->pool_mutex);
        auto it = pimpl->memory_pool.find(num_elements);
        if (it != pimpl->memory_pool.end() && !it->second.empty()) {
            double* d_ptr = it->second.back();
            it->second.pop_back();
            pimpl->pool_bytes -= num_elements * sizeof(double);
            CUDA_CHECK(cudaMemset(d_ptr, 0, num_elements * sizeof(double)));
            return d_ptr;
        }
    }
    double* d_ptr = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_ptr, num_elements * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_ptr, 0, num_elements * sizeof(double)));
    return d_ptr;
}"""

FREE_ANCHOR = """void GPUManager::free_device_memory(double* d_ptr, size_t num_elements) {
    if (d_ptr == nullptr) return;
    // std::lock_guard<std::mutex> lock(pimpl->pool_mutex);
    // pimpl->memory_pool[num_elements].push_back(d_ptr);
    CUDA_CHECK(cudaFree(d_ptr));
}"""

FREE_OUT = """void GPUManager::free_device_memory(double* d_ptr, size_t num_elements) {
    if (d_ptr == nullptr) return;
    // Iter30: return to the pool when under the byte cap, else free directly
    // (bounds retained device memory on the 10 GiB MIG slice).
    size_t bytes = num_elements * sizeof(double);
    {
        std::lock_guard<std::mutex> lock(pimpl->pool_mutex);
        if (pimpl->pool_bytes + bytes <= Impl::POOL_CAP_BYTES) {
            pimpl->memory_pool[num_elements].push_back(d_ptr);
            pimpl->pool_bytes += bytes;
            return;
        }
    }
    CUDA_CHECK(cudaFree(d_ptr));
}"""

CLEAR_ANCHOR = """void GPUManager::clear_pool() {
    // std::lock_guard<std::mutex> lock(pimpl->pool_mutex);
    // for (auto& pair : pimpl->memory_pool) {
    //     for (double* d_ptr : pair.second) {
    //         cudaFree(d_ptr);
    //     }
    // }
    pimpl->memory_pool.clear();
}"""

CLEAR_OUT = """void GPUManager::clear_pool() {
    // Iter30: release all pooled buffers (called from the destructor before
    // cudaDeviceReset, so the frees are safe there).
    std::lock_guard<std::mutex> lock(pimpl->pool_mutex);
    for (auto& pair : pimpl->memory_pool) {
        for (double* d_ptr : pair.second) cudaFree(d_ptr);
    }
    pimpl->memory_pool.clear();
    pimpl->pool_bytes = 0;
}"""


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    if len(sys.argv) != 2:
        print("usage: patch_p30_pool.py <candidate_root>"); sys.exit(2)
    cand = sys.argv[1]
    if os.path.exists(cand):
        print(f"ERROR: {cand} exists"); sys.exit(3)

    p = os.path.join(FORMAL, "src", "gpu_manager.cu")
    fh = sha(p)
    print(f"formal gpu_manager.cu {fh[:16]}")
    assert fh.startswith("435c9b69"), "gpu_manager.cu drifted from deployed baseline"

    shutil.copytree(FORMAL, cand, ignore=shutil.ignore_patterns(*EXCLUDE_DIRS))
    print(f"copied formal -> {cand}")

    p = os.path.join(cand, "src", "gpu_manager.cu")
    s = open(p).read()
    for anchor, out, label in ((IMPL_ANCHOR, IMPL_OUT, "Impl"),
                               (ALLOC_ANCHOR, ALLOC_OUT, "allocate"),
                               (FREE_ANCHOR, FREE_OUT, "free"),
                               (CLEAR_ANCHOR, CLEAR_OUT, "clear_pool")):
        n = s.count(anchor)
        assert n == 1, f"{label} anchor: {n}"
        s = s.replace(anchor, out)
    open(p, "w").write(s)
    assert s.count("POOL_CAP_BYTES") == 2
    assert s.count("pool_bytes") >= 4
    assert "cudaMemset(d_ptr, 0, num_elements * sizeof(double))" in s
    print(f"patched gpu_manager.cu ({sha(p)[:16]})")
    print("PATCH OK")


if __name__ == "__main__":
    main()
