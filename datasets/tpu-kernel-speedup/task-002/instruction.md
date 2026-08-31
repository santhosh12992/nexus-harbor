# TPU Kernel Optimization Task: Fused RMSNorm on TPU v6e

## Objective
Optimize the reference RMSNorm (Root Mean Square Layer Normalization) kernel provided in `baseline_kernel.py`.
Your solution must be written to `optimized_kernel.py` and implement the `run_kernel(X, W, eps)` function.

## Mathematical Formulation
$$\text{RMSNorm}(X, W) = \frac{X}{\sqrt{\frac{1}{D} \sum_{i=1}^D X_i^2 + \epsilon}} \odot W$$
where $X \in \mathbb{R}^{B \times S \times D}$ is the input activation tensor, $W \in \mathbb{R}^D$ is the learnable weight vector, and $\epsilon=10^{-6}$.

## Multi-Iteration Self-Refinement Protocol
You have access to a hardware profiling feedback tool: `python3 /workspace/profile_kernel.py`.

You MUST use an iterative optimization loop:
1. **Initial Implementation**: Write a baseline kernel to `optimized_kernel.py` and run `python3 /workspace/profile_kernel.py` to inspect initial execution latency and accuracy.
2. **Iterative Refinement**:
   - Inspect the latency and speedup returned by `profile_kernel.py`.
   - Edit `optimized_kernel.py` to fuse variance reduction with normalization, optimize row-wise block tiling (e.g. 128 rows per tile), and reduce memory roundtrips.
   - Run `python3 /workspace/profile_kernel.py` again to evaluate the performance delta across iterations.
3. **Target Constraints**:
   - **Mathematical Accuracy**: Maximum absolute difference <= 1e-2 against `baseline_kernel.py`.
   - **Speedup Target**: Achieve at least 1.5x hardware speedup.
4. Conclude only after verifying your final speedup with `python3 /workspace/profile_kernel.py`.
