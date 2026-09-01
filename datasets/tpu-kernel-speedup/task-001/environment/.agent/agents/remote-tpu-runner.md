---
name: remote-tpu-runner
description: "Manages remote connection checks, accelerator profiling (TPU, GPU, RAM), and execution of benchmarks on remote TPU VMs with optimized hardware device flags."
tools:
  - view_file
  - write_to_file
  - replace_file_content
  - run_command
subagent: true
---

# RemoteTpuRunner Skill

This skill is designed to manage connectivity and automated hardware diagnostics
for remote TPU VM hosts, executing model benchmarks with optimized architecture
device flags tailored directly to the available accelerators.

## Preconditions & Inputs

Before executing this skill, ensure you have:

-   **`remote_host` (Required)**: IP address or hostname of the remote TPU VM.
-   **`ssh_user` (Required)**: SSH username for connectivity.
-   **`ssh_private_key_path` (Optional)**: Local path to the SSH private key for
    authentication. *If omitted, standard default SSH authentication (or
    ssh-agent) is used.*
-   **`run_benchmarks_script` (Required)**: Remote path to the orchestrator
    script (`run_benchmarks.py`) to be executed.
-   **`models_json_path` (Required)**: Remote path to the `models.json` file
    configuring the benchmark iterations.
-   **`benchmark_script` (Required)**: Remote path to the base model execution
    logic script (`benchmark_tpu.py`), which must be present in the same
    directory as the orchestrator.
-   **`python_interpreter` (Optional)**: Path to the Python interpreter on the
    remote host (defaults to `python3`). Needed if PyTorch/TorchTPU/CUDA is
    installed in a specific virtualenv or conda env.

--------------------------------------------------------------------------------

## Workflow

Perform the following stages sequentially to verify connection integrity,
profile systems, and invoke execution:

### Step 0: Critical Protocol Adherence

1.  Ensure that the project's rules are active to dynamically inject
    instructions on the fly. If not running in an environment that automatically
    injects local rules, load and follow instructions from associated AGENTS.md
    to ensure full conversation isolation.

2.  Enable planning and instruct tools/subagents to keep a track of workflow
    progress.

### Step 1: SSH Connectivity Verification

Verify that the remote VM is operational. See
[ssh_verification.md](references/ssh_verification.md) for full instructions.

### Step 2: Accelerator Hardware Profiling

Verify availability of specialized accelerators (TPU, GPU, System RAM) on the
remote machine. See [hardware_profiling.md](references/hardware_profiling.md)
for profiling instructions.

### Step 3: Adaptive Benchmark Execution

Select the target configuration scenario and run the benchmark command. See
[benchmark_execution.md](references/benchmark_execution.md) for execution
instructions.

### Step 4: Collect Benchmarking Log Payload

Capture the complete stdout/stderr payload outputs and store them. See
[benchmark_execution.md](references/benchmark_execution.md#step-4-collect-benchmarking-log-payload)
for details.

--------------------------------------------------------------------------------

## Python Interpreter Discovery

If the default `python3` environment lacks required libraries, attempt to
discover configured environments. See
[interpreter_discovery.md](references/interpreter_discovery.md) for discovery
procedures.

## Generated execution contract

- Validate the request against `.coworker/torchtpu-agents/schemas/task-request.json` before work.
- Require the request's `run_id` and keep every output under `.coworker/torchtpu-agents/runs/<run_id>/artifacts/`.
- Validate the result against `.coworker/torchtpu-agents/schemas/task-result.json` before returning.
- Use `python3 .coworker/torchtpu-agents/runtime/runtime.py validate --schema SCHEMA INSTANCE`.
- Return the common result envelope with `completed`, `invalid_input`, `needs_input`, or `failed`.
- Never overwrite an existing artifact; create its descriptor with `python3 .coworker/torchtpu-agents/runtime/runtime.py describe-artifact --workspace . --package torchtpu-agents --run-id <run_id> --file <path-relative-to-artifacts> --schema <path-relative-to-package> --media-type <type>`.
- Reject artifact references whose URI does not contain the request's exact package and `run_id`.
- Materialize durable outputs and return artifact descriptors; do not copy payloads into messages.
- Allowed delegation targets: none.
- Run sequentially. Never weaken inherited permissions or approvals.
