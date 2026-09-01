---
name: torch-tpu-inference
description: "Orchestrates the deployment of Hugging Face models for HTTP inference on Google Cloud TPU. Manages the persistent service lifecycle: compatibility checking, deployment, health monitoring, client verification, and teardown."
tools:
  - view_file
  - write_to_file
  - replace_file_content
  - run_command
subagent: true
---

# Torch TPU Inference Orchestrator

This skill orchestrates the end-to-end deployment of HuggingFace models for HTTP
inference on Cloud TPU VMs.

## Required Inputs

-   `model_id`: The Hugging Face model identifier (e.g.,
    `meta-llama/Llama-3.2-1B`).
-   `ssh_user`: The SSH username for the remote TPU VM.
-   `remote_host`: The IP address or hostname of the remote TPU VM.

## Step-by-Step Inference Workflow

You MUST follow this checklist in order. Check off each step and proceed to the
next.

-   [ ] **Step 0.1: Remote Runtime Setup**: Read/verify SSH connection details
    (check `~/.ssh/config` or SSH configuration). If missing, ask the user.
-   [ ] **Step 0.2: Remote Hardware Diagnostics**: Read `remote_tpu_runner`
    skill via `view_file` and execute it to verify SSH and probe TPU accelerator
    availability and HBM.
-   [ ] **Step 0.3: Pre-flight Compatibility Check**: Read `preflight_check`
    skill via `view_file` and complete preflight compatibility checks on the
    model.
-   [ ] **Step 1: Create deployments.json**: Following the instructions in
    `torch_inference_coder` create a declarative service manifest
    `deployments.json` defining the model ID, target engine, sequence lengths,
    and port.
-   [ ] **Step 2: Inference Code Generation**: Generate the required serving
    wrapper script as per the instructions in `torch-inference-coder` skill and
    upload `deployments.json` along with the script to the remote VM.
-   [ ] **Step 3: Deploy Model & Monitor Health**: Using `torch_inference_coder`
    instructions, launch the serving engine as a background daemon on the remote
    VM. Capture the PID. Poll the engine's `/health` endpoint until it returns
    HTTP `200 OK`.
-   [ ] **Step 4: Output Verification**: Send a test inference request to the
    API server over HTTP to verify functional correctness (coherent generation).
-   [ ] **Step 5: Hand-off**: Provide the user with the endpoint details and the
    PID of the daemon. Keep the service running unless the user explicitly
    requests teardown.

> **CRITICAL**: Do NOT tear down the inference server after verification unless
> explicitly commanded by the user.

## Dependencies

-   [remote-tpu-runner](../remote_tpu_runner/SKILL.md)
-   [preflight-check](../preflight_check/SKILL.md)
-   [torch-inference-coder](../torch_inference_coder/SKILL.md)

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
