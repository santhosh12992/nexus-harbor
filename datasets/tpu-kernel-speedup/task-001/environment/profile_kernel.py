#!/usr/bin/env python3
"""TPU Hardware Profiler & Feedback Tool for Agent Refinement Loop with AST deduplication and robust verification."""
import ast
import hashlib
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

def get_ast_hash(code_str: str) -> str:
    """Returns MD5 hash of the normalized Python AST, ignoring comments and whitespace."""
    try:
        tree = ast.parse(code_str)
        dumped = ast.dump(tree, include_attributes=False)
        return hashlib.md5(dumped.encode("utf-8")).hexdigest()
    except Exception:
        return hashlib.md5(code_str.encode("utf-8")).hexdigest()

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

    backend = jax.default_backend()
    backend_label = "Physical Cloud TPU v6e" if backend == "tpu" else "Local JAX CPU Emulation Mode"

    print("================================================================================")
    print(f"       TPU Kernel Profiler Feedback [{backend_label}]")
    print("================================================================================")

    # 1. Multi-Shape Numerical Parity & NaN/Inf Verification
    test_shapes = [
        (4, 8, 1024, 128),
        (2, 4, 512, 128),
        (1, 8, 2048, 128),
    ]

    print("[*] Numerical Accuracy Check across Multi-Shape Configurations:")
    max_diff_overall = 0.0
    for b, h, s, d in test_shapes:
        key = jax.random.PRNGKey(s)
        k1, k2, k3 = jax.random.split(key, 3)
        Q_t = jax.random.normal(k1, (b, h, s, d), dtype=jnp.float32)
        K_t = jax.random.normal(k2, (b, h, s, d), dtype=jnp.float32)
        V_t = jax.random.normal(k3, (b, h, s, d), dtype=jnp.float32)

        try:
            ref_out = base_mod.run_kernel(Q_t, K_t, V_t).block_until_ready()
            opt_out = opt_mod.run_kernel(Q_t, K_t, V_t).block_until_ready()
        except Exception as e:
            print(f"[VERIFICATION FAILED] Execution error for shape ({b}, {h}, {s}, {d}): {e}")
            sys.exit(1)

        # Check for NaN / Inf
        if bool(jnp.isnan(opt_out).any()) or bool(jnp.isinf(opt_out).any()):
            print(f"[VERIFICATION FAILED] Output contains NaN or Inf values for shape ({b}, {h}, {s}, {d})!")
            sys.exit(1)

        cur_diff = float(jnp.max(jnp.abs(ref_out - opt_out)))
        max_diff_overall = max(max_diff_overall, cur_diff)
        is_shape_accurate = bool(jnp.allclose(ref_out, opt_out, atol=1e-2, rtol=1e-2))

        print(f"    - Shape (B={b}, H={h}, S={s}, D={d}): Max Diff = {cur_diff:.6f} -> {'PASSED' if is_shape_accurate else 'FAILED'}")
        if not is_shape_accurate:
            print(f"[ERROR] Numerical accuracy exceeded error threshold for shape ({b}, {h}, {s}, {d}). Ensure kernel handles dynamic sequence lengths.")
            sys.exit(1)

    is_accurate = max_diff_overall <= 1e-2

    # 2. Timing & Benchmark (Primary Shape B=4, H=8, S=1024, D=128 with anti-memoization perturbation)
    B, H, S, D = 4, 8, 1024, 128
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    Q = jax.random.normal(k1, (B, H, S, D), dtype=jnp.float32)
    K = jax.random.normal(k2, (B, H, S, D), dtype=jnp.float32)
    V = jax.random.normal(k3, (B, H, S, D), dtype=jnp.float32)

    # Warmup
    for _ in range(5):
        _ = opt_mod.run_kernel(Q, K, V).block_until_ready()

    n_iters = 25
    if backend == "cpu":
        t0_b = time.perf_counter()
        for i in range(n_iters):
            _ = base_mod.run_kernel(Q + (i * 1e-7), K + (i * 1e-7), V + (i * 1e-7)).block_until_ready()
        t1_b = time.perf_counter()
        base_lat_us = ((t1_b - t0_b) / n_iters) * 1e6
    else:
        base_lat_us = 350.0  # Baseline reference TPU latency

    t0 = time.perf_counter()
    for i in range(n_iters):
        _ = opt_mod.run_kernel(Q + (i * 1e-7), K + (i * 1e-7), V + (i * 1e-7)).block_until_ready()
    t1 = time.perf_counter()

    opt_lat_us = ((t1 - t0) / n_iters) * 1e6

    # Compute speedup relative to baseline
    speedup = base_lat_us / opt_lat_us if opt_lat_us > 0 else 1.0

    # 3. AST-aware Deduplication & History Tracking
    with open(opt_path, "r") as f:
        opt_code = f.read()
    current_ast_hash = get_ast_hash(opt_code)

    history = []
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history = json.load(f)
        except Exception:
            history = []

    is_duplicate = False
    if history and history[-1].get("ast_hash") == current_ast_hash:
        is_duplicate = True
        iteration_num = len(history)
        history[-1]["latency_us"] = round(opt_lat_us, 2)
        history[-1]["speedup"] = round(speedup, 3)
        history[-1]["max_diff"] = round(max_diff_overall, 6)
        history[-1]["accurate"] = is_accurate
    else:
        iteration_num = len(history) + 1
        history.append({
            "iteration": iteration_num,
            "ast_hash": current_ast_hash,
            "latency_us": round(opt_lat_us, 2),
            "speedup": round(speedup, 3),
            "max_diff": round(max_diff_overall, 6),
            "accurate": is_accurate
        })

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    artifacts_dir = "/logs/artifacts"
    if not is_duplicate:
        # Auto-snapshot kernel for this iteration to workspace and artifacts
        iter_snapshot = os.path.join(ws, f"iteration_{iteration_num}_kernel.py")
        try:
            shutil.copyfile(opt_path, iter_snapshot)
        except Exception:
            pass

        if os.path.exists(artifacts_dir):
            try:
                shutil.copyfile(opt_path, os.path.join(artifacts_dir, f"iteration_{iteration_num}_kernel.py"))
            except Exception:
                pass

    print(f"\n[*] Performance Metrics [{backend_label}] (Iteration #{iteration_num}):")
    if is_duplicate:
        print(f"    - Snapshot Status:         RETAINED iteration_{iteration_num}_kernel.py (AST unchanged)")
    else:
        print(f"    - Snapshot Saved:          iteration_{iteration_num}_kernel.py")
    print(f"    - Baseline Latency:        {base_lat_us:.2f} us")
    print(f"    - Optimized Latency:       {opt_lat_us:.2f} us")
    print(f"    - Speedup Factor:          {speedup:.3f}x")
    print(f"    - Target Speedup:          >= 1.500x")
    print("--------------------------------------------------------------------------------")

    if speedup >= 1.5:
        print(f"[SUCCESS] Target speedup achieved ({speedup:.2f}x >= 1.50x). Your kernel is optimal!")
    else:
        print(f"[REFINEMENT NEEDED] Current speedup is {speedup:.2f}x. Optimize block tiling (e.g. 128x128), reduce memory traffic, and re-run.")
    print("================================================================================\n")

if __name__ == "__main__":
    main()
