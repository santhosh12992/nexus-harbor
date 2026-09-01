---
name: preflight-check
description: "Validates Hugging Face model compatibility, architecture, tokenizer, and licensing properties locally before undertaking TPU compilation or remote benchmark runs."
tools:
  - view_file
  - write_to_file
  - replace_file_content
  - run_command
subagent: true
---

# Pre-flight Compatibility Check Skill

This skill handles the local preflight validation of a Hugging Face model to
verify its compatibility with `torch_tpu` (PyTorch XLA on TPU VMs). This helps
prevent spending resource/hardware budget on incompatible architectures or
running into runtime crashes.

The core validator script is located at
[preflight_check.py](scripts/preflight_check.py).

## Workflow

Before generating benchmark scripts or executing remote TPU/CPU runs, you
**MUST** validate the compatibility of **every single candidate model**
(locally, or on the remote VM host as a backup option if local execution fails)
using this skill. **Never skip or bypass running the preflight check for any
candidate model**, even if it belongs to a standard library or architecture.

### Running a Pre-flight Check

Execute the preflight check script for each model using direct `python3`
execution locally or remotely via SSH:

-   **Local Execution (Primary)**:

    ```bash
    python3 learning/agents/tern/torch_agent/_agents/skills/preflight_check/scripts/preflight_check.py --model={model_id} --allow_gpu_imports
    ```

-   **Remote Execution (Backup option if local execution fails due to missing
    dependencies)**:

    ```bash
    ssh ... "python3 ~/benchmarks/preflight_check.py --model={model_id} --allow_gpu_imports"
    ```

*(Note: If running in a Google3 bazel-native build environment, `bazel run
//learning/agents/tern/torch_agent/_agents/skills/preflight_check/scripts:preflight_check
-- --model={model_id} --allow_gpu_imports` may also be used.)*

## Rules

When running the tool, monitor the stdout and exit code. Take the following
actions based on results:

### A. Access & Gating Check

-   If the check flags a model as requiring gated access (e.g., raising a
    `GatedRepoError` or missing valid access token), and the credentials are not
    provided/valid, the orchestrator **must immediately skip** this model,
    evaluate the next candidate in the search buffer (discovered via
    `smart-hf-search`), and run the local `preflight-check` on it to verify
    compatibility.

### B. Format & Library Tier Validation

The `preflight_check` tool classifies the model's Hugging Face library into one
of three tiers based on metadata and file extensions:

-   **Tier None (Incompatible / Unsupported)**:
    -   *Identification*: Detected via specific `library_name` tags
        (`tensorflow`, `jax`, `onnx`, `openvino`, `coreml`, `paddlepaddle`,
        `tflite`, `gguf`, `wav2letter`, `scikit-learn`) or if the repo files
        contain only non-PyTorch binaries (e.g. only `.gguf` or `.tflite`,
        without any PyTorch/Safetensors weights).
    -   *Action/Utility*: The check fails immediately and you **must abort** the
        migration. These models do not support PyTorch-XLA/`torch_tpu` device
        mapping.
-   **Tier B (Medium/Low Priority)**:
    -   *Identification*: Detected via NLP pipeline packaging or agentic wrapper
        framework tags (`spacy`, `stanza`, `flair`, `allennlp`,
        `simpletransformers`, `bertopic`, `haystack`, `ml-agents`).
    -   *Action/Utility*: The check warns that the library is compatible but
        runs with medium/low benefit due to CPU-bound overhead or high-level
        pipeline scaffolding.
-   **Tier A (High Benefit - Target)**:
    -   *Identification*: Standard model libraries (e.g. `transformers`,
        `diffusers`, `timm`, `sentence-transformers`, `peft`, `trl`) not
        classified in Tier None or Tier B.
    -   *Action/Utility*: Fully supported and designated as high-priority
        targets for TPU VM acceleration.

### C. Configuration & Processor Check

-   **Quantization**: If the model has quantization settings using GPU-specific
    kernels (e.g., AWQ, GPTQ), it is currently marked incompatible for automated
    TPU routes.
-   **Processor / Tokenizer**: The check verifies that the tokenizer or
    processor loads successfully. If it is non-instantiable due to mismatching
    config mapping, the check fails.
-   **Vocabulary Check**: Validates that the tokenizer vocabulary size does not
    exceed the model's embedding layers config parameters (to prevent index
    errors).
-   **Specialized Models**: If the model is tagged, named, or configured with
    specialized keywords (such as protein folding, document layout, table
    structures, GNNs), a preflight warning is raised indicating custom wrapper
    handling is required.

### D. Incompatible Imports Scan in Custom Code

-   For custom models (`trust_remote_code=True`), the script scans the
    downloaded python codebase.
-   If it flags forbidden imports such as:
    -   `triton`, `flash_attn`, `apex`, `deepspeed`, `xformers`, `mamba_ssm`:
        Since `--allow_gpu_imports` is used by default, the check will print a
        warning but pass. You **may proceed** with the migration, but you should
        be aware that TPU compilation or execution might fail if these code
        paths are triggered on the remote VM.

## Tool Options

-   `--model`: Target Hugging Face model ID (e.g.
    `meta-llama/Llama-3.1-8B-Instruct`).
-   `--token`: Optional Hugging Face token for API authentication.
-   `--force`: Bypass all preflight checking errors (marked as `[FORCE
    BYPASS]`).

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
