# TPU Kernel Optimization Task: FlashAttention on TPU v6e

## Objective
Optimize the reference attention kernel provided in `baseline_kernel.py`.
Your solution must be written to `optimized_kernel.py` and implement the `run_kernel(Q, K, V)` function.

## Multi-Iteration Self-Refinement Protocol
You have access to a hardware profiling feedback tool: `python3 /workspace/profile_kernel.py`.

You MUST use an iterative optimization loop:
1. **Initial Implementation**: Write a baseline Pallas kernel to `optimized_kernel.py` and run `python3 /workspace/profile_kernel.py` to inspect initial execution latency and accuracy.
2. **Iterative Refinement**:
   - Inspect the latency and speedup returned by `profile_kernel.py`.
   - Edit `optimized_kernel.py` to optimize block tiling (e.g. 128x128 block dimensions), padding, and memory layouts.
   - Run `python3 /workspace/profile_kernel.py` again to evaluate the performance delta across iterations.
3. **Target Constraints**:
   - **Mathematical Accuracy**: Maximum absolute difference <= 1e-2 against `baseline_kernel.py`.
   - **Speedup Target**: Achieve at least 1.5x hardware speedup.
4. Conclude only after verifying your final speedup with `python3 /workspace/profile_kernel.py`.
