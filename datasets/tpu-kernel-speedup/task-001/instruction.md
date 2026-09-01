# TPU Kernel Optimization Task: FlashAttention on TPU v6e

## Objective
Optimize the reference attention kernel provided in `workspace/base.py` (and `baseline_kernel.py`).
Your solution must implement the `run_kernel(Q, K, V)` function.

## Multi-Iteration Self-Refinement Workflow
You are equipped with the `tpu-coworker` / `maxkernel-loop` optimization package.
Execute the iterative self-refinement optimization loop:
1. **Initial Pallas Implementation**: Create an initial Pallas attention kernel with 2D block tiling.
2. **Iterative Refinement**:
   - Run compilation and hardware profiling feedback via `python3 /workspace/profile_kernel.py` (or `maxkernel/scripts/tpu_client.py`).
   - Refine block dimensions (e.g., 128x128), memory tiling, and online softmax scaling to maximize TPU VMU throughput.
   - Achieve at least 1.5x hardware speedup with <= 1e-2 maximum absolute difference against `baseline_kernel.py`.
3. **Conclusion**: Save your best optimized kernel to `/workspace/optimized_kernel.py` (or `workspace/state.json`) and conclude with 'Optimization complete. Final kernel is written to optimized_kernel.py'.
