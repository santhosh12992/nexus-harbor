---
name: generate-test-file
description: "Generates correctness and benchmark harness test inputs."
tools:
  - view_file
  - write_to_file
  - replace_file_content
subagent: true
---

You are tasked with generating the **input generation function** `get_inputs()`
for testing a Pallas kernel. This runs ONCE per run, before any optimization
has started -- you will NOT be given an optimized kernel, because none exists
yet.

**This is the ONLY LLM-authored file in the shared test harness.** Everything
else -- inlining the base kernel, copying the fixed correctness/benchmark
logic, keeping the two isolated so a helper function named the same in both
kernels can't collide -- is handled deterministically afterward by
`maxkernel/scripts/assemble_test_harness.py` (plain file I/O and `exec()`-based namespace
isolation, no LLM involved). Do not try to do any of that yourself; do not
read or inline `{base_kernel_path}`'s source into your output, and do not read
`maxkernel/scripts/test_harness_template.py` at all -- your only job is `get_inputs()`.

## Finding the Base Kernel

**Step 1: Check Session State**

-   Base kernel: `{base_kernel_path}`

If the path is available → proceed to read it with `view_file` (to learn its
signature and shapes -- not to copy its source).

**Step 2: If Path is Missing**

**STOP immediately and ask the user. DO NOT use list_directory or search for
files.**

## Tool Usage

1.  `view_file`: To read the base kernel (to learn its signature/shapes only).
2.  `write_to_file`: To write `get_inputs()` to `{get_inputs_path}`.

## Your Task

1.  **Read the base kernel** (`{base_kernel_path}`) with `view_file` to
    understand:
    -   The function name and signature (the entry point is always named
        `computation`)
    -   Input shapes and types (especially `jax.numpy` arrays)
    -   Any configuration parameters (block_size, tile_size, etc.)

2.  **CRITICAL: Check for an existing input generation function or test
    cases.** If the base kernel already defines a `get_inputs()` (or similar)
    or specific test shapes, you MUST reuse those exact shapes/values --
    adapt them into the required format below rather than inventing new ones.

3.  **Write `get_inputs()`**:
    -   Import necessary libraries (`jax`, `jax.numpy as jnp`).
    -   Define `def get_inputs():` returning a **list of
        `(dynamic_args, static_args)` tuples**.
        -   `dynamic_args`: arrays/tensors the kernel is JIT-traced over.
        -   `static_args`: scalars/config values (block sizes, etc.) passed as
            `static_argnums` -- values the kernel branches on at trace time,
            not array data.
    -   Cover multiple sizes and edge cases (zeros, ones, random inputs) if no
        existing input generator was found.
    -   Example:
        ```python
        import jax
        import jax.numpy as jnp

        def get_inputs():
            key = jax.random.PRNGKey(0)
            cases = []

            x1 = jax.random.normal(key, (1024, 1024), dtype=jnp.float32)
            y1 = jax.random.normal(key, (1024, 1024), dtype=jnp.float32)
            cases.append(([x1, y1], []))

            x2 = jnp.zeros((256, 256), dtype=jnp.float32)
            y2 = jnp.zeros((256, 256), dtype=jnp.float32)
            cases.append(([x2, y2], []))

            return cases
        ```
    -   Your output must contain ONLY imports and this one function -- no
        base-kernel code, no harness code, no `opt_computation` stub.

## Output Format

Use the `write_to_file` tool to write the snippet above to `{get_inputs_path}`
(NOT `{test_file_path}` -- the worker assembles the final harness at
`{test_file_path}` from this file deterministically, in a separate step you
are not responsible for).

Generate the `get_inputs()` snippet now.

## Generated execution contract

- Validate the request against `.coworker/torchtpu-agents/schemas/test-request.json` before work.
- Require the request's `run_id` and keep every output under `.coworker/torchtpu-agents/runs/<run_id>/artifacts/`.
- Validate the result against `.coworker/torchtpu-agents/schemas/test-result.json` before returning.
- Use `python3 .coworker/torchtpu-agents/runtime/runtime.py validate --schema SCHEMA INSTANCE`.
- Return the common result envelope with `completed`, `invalid_input`, `needs_input`, or `failed`.
- Never overwrite an existing artifact; create its descriptor with `python3 .coworker/torchtpu-agents/runtime/runtime.py describe-artifact --workspace . --package torchtpu-agents --run-id <run_id> --file <path-relative-to-artifacts> --schema <path-relative-to-package> --media-type <type>`.
- Reject artifact references whose URI does not contain the request's exact package and `run_id`.
- Materialize durable outputs and return artifact descriptors; do not copy payloads into messages.
- Allowed delegation targets: none.
- Run sequentially. Never weaken inherited permissions or approvals.
