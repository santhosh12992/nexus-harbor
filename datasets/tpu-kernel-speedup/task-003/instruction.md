# TPU Kernel Optimization Task: Fused SwiGLU on TPU v6e

## Objective
Optimize the reference SwiGLU (Swish Gated Linear Unit) activation kernel provided in `workspace/base.py` (and `baseline_kernel.py`).
Your solution must implement the `run_kernel(X, W_gate, W_up)` function.

## Multi-Iteration Self-Refinement Workflow
You are equipped with the `tpu-coworker` / `maxkernel-loop` optimization package.
Execute the iterative self-refinement optimization loop:
1. **Initial Pallas Implementation**: Create an initial Pallas kernel that performs concatenated linear projection with in-register SiLU activation and elementwise multiplication.
2. **Iterative Refinement**:
   - Run compilation and hardware profiling feedback via `python3 /workspace/profile_kernel.py` (or `maxkernel/scripts/tpu_client.py`).
   - Optimize VMU block sizes, matmul tiling, and memory bandwidth utilization.
   - Achieve at least 1.5x hardware speedup with <= 1e-2 maximum absolute difference against `baseline_kernel.py`.
3. **Conclusion**: Save your best optimized kernel to `/workspace/optimized_kernel.py` (or `workspace/state.json`) and conclude with 'Optimization complete. Final kernel is written to optimized_kernel.py'.
