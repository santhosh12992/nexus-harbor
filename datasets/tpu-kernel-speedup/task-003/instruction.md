# TPU Kernel Optimization Task: Fused SwiGLU on TPU v6e

## Objective
Optimize the reference SwiGLU (Swish Gated Linear Unit) kernel provided in `baseline_kernel.py`.
Your solution must be written to `optimized_kernel.py` and implement the `run_kernel(X, W_gate, W_up)` function.

## Mathematical Formulation
$$\text{SwiGLU}(X, W_{gate}, W_{up}) = (\text{SiLU}(X \cdot W_{gate})) \odot (X \cdot W_{up})$$
where $\text{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$, $X \in \mathbb{R}^{B \times S \times D_{in}}$, and $W_{gate}, W_{up} \in \mathbb{R}^{D_{in} \times D_{out}}$.

## Mandatory Requirement
You MUST create and write your solution to `/workspace/optimized_kernel.py` and run `python3 /workspace/profile_kernel.py` to evaluate performance. Do NOT stop until `/workspace/optimized_kernel.py` is written and verified.

## Multi-Iteration Self-Refinement Protocol
You have access to a hardware profiling feedback tool: `python3 /workspace/profile_kernel.py`.

You MUST use an iterative optimization loop:
1. **Initial Implementation**: Write a baseline kernel to `optimized_kernel.py` and run `python3 /workspace/profile_kernel.py` to inspect initial execution latency and accuracy.
2. **Iterative Refinement**:
   - Inspect the latency and speedup returned by `profile_kernel.py`.
   - Edit `optimized_kernel.py` to fuse projection matmuls (e.g. concatenated $W_{concat} = [W_{gate}, W_{up}]$), in-register SiLU gating, and 2D block tiling.
   - Run `python3 /workspace/profile_kernel.py` again to evaluate the performance delta across iterations.
3. **Target Constraints**:
   - **Mathematical Accuracy**: Maximum absolute difference <= 1e-2 against `baseline_kernel.py`.
   - **Speedup Target**: Achieve at least 1.5x hardware speedup.
4. **Conclusion**: When you have verified your final speedup with `python3 /workspace/profile_kernel.py`, conclude your response by printing 'Optimization complete. Final kernel is written to optimized_kernel.py' and stop making further tool calls.
