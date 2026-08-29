#!/usr/bin/env python3
"""TPU Hardware Profiler & Feedback Tool for Agent Refinement Loop with per-iteration snapshotting."""
import importlib.util
import json
import os
import shutil
import sys
import time
import jax
import jax.numpy as jnp

def load_module(name: str, path: str):
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    ws = os.environ.get("WORKSPACE_DIR", "/workspace")
    base_path = os.path.join(ws, "baseline_kernel.py")
    opt_path = os.path.join(ws, "optimized_kernel.py")
    history_path = os.path.join(ws, "profiler_history.json")

    if not os.path.exists(opt_path):
        print(f"[ERROR] /workspace/optimized_kernel.py does not exist. Please write your kernel implementation first.")
        sys.exit(1)

    base_mod = load_module("base", base_path)
    opt_mod = load_module("opt", opt_path)

    if not hasattr(opt_mod, "run_kernel"):
        print(f"[ERROR] optimized_kernel.py must define a function 'run_kernel(Q, K, V)'.")
        sys.exit(1)

    print("================================================================================")
    print("                TPU v6e Kernel Hardware Profiler Feedback                       ")
    print("================================================================================")

    # 1. Verification Test
    B, H, S, D = 4, 8, 1024, 128
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    Q = jax.random.normal(k1, (B, H, S, D), dtype=jnp.float32)
    K = jax.random.normal(k2, (B, H, S, D), dtype=jnp.float32)
    V = jax.random.normal(k3, (B, H, S, D), dtype=jnp.float32)

    try:
        ref_out = base_mod.run_kernel(Q, K, V).block_until_ready()
        opt_out = opt_mod.run_kernel(Q, K, V).block_until_ready()
    except Exception as e:
        print(f"[VERIFICATION FAILED] Execution error: {e}")
        sys.exit(1)

    max_diff = float(jnp.max(jnp.abs(ref_out - opt_out)))
    is_accurate = bool(jnp.allclose(ref_out, opt_out, atol=1e-2, rtol=1e-2))

    print(f"[*] Numerical Accuracy Check:")
    print(f"    - Max Absolute Difference: {max_diff:.6f}")
    print(f"    - Accuracy Status:         {'PASSED (<= 1e-2)' if is_accurate else 'FAILED (> 1e-2)'}")

    if not is_accurate:
        print(f"[ERROR] Numerical accuracy exceeded error threshold. Please fix accuracy before optimizing further.")
        sys.exit(1)

    # 2. Timing & Benchmark
    # Warmup
    for _ in range(5):
        _ = opt_mod.run_kernel(Q, K, V).block_until_ready()

    n_iters = 25
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = opt_mod.run_kernel(Q, K, V).block_until_ready()
    t1 = time.perf_counter()

    opt_lat_us = ((t1 - t0) / n_iters) * 1e6
    base_lat_us = 350.0  # Baseline reference TPU latency

    # Compute speedup relative to TPU reference baseline
    speedup = base_lat_us / opt_lat_us if opt_lat_us > 0 else 1.0

    # Load history
    history = []
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history = json.load(f)
        except Exception:
            history = []

    iteration_num = len(history) + 1
    history.append({
        "iteration": iteration_num,
        "latency_us": round(opt_lat_us, 2),
        "speedup": round(speedup, 3),
        "max_diff": round(max_diff, 6),
        "accurate": is_accurate
    })

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Auto-snapshot kernel for this iteration to workspace and artifacts
    iter_snapshot = os.path.join(ws, f"iteration_{iteration_num}_kernel.py")
    try:
        shutil.copyfile(opt_path, iter_snapshot)
    except Exception:
        pass

    artifacts_dir = "/logs/artifacts"
    if os.path.exists(artifacts_dir):
        try:
            shutil.copyfile(opt_path, os.path.join(artifacts_dir, f"iteration_{iteration_num}_kernel.py"))
        except Exception:
            pass

    print(f"\n[*] TPU Performance Metrics (Iteration #{iteration_num}):")
    print(f"    - Snapshot Saved:          iteration_{iteration_num}_kernel.py")
    print(f"    - Baseline TPU Latency:    {base_lat_us:.2f} us")
    print(f"    - Optimized TPU Latency:   {opt_lat_us:.2f} us")
    print(f"    - Speedup Factor:          {speedup:.3f}x")
    print(f"    - Target Speedup:          >= 1.500x")
    print("--------------------------------------------------------------------------------")

    if speedup >= 1.50:
        print(f"[SUCCESS] Target speedup achieved ({speedup:.2f}x >= 1.50x). Your kernel is optimal!")
    else:
        print(f"[IN_PROGRESS] Current speedup ({speedup:.2f}x) is below 1.50x. Consider tuning tile block sizes (e.g. 128x128), memory layout, or masking logic.")
    print("================================================================================\n")

if __name__ == "__main__":
    main()
