---
name: torch-tpu-migration-orchestrator
description: "Orchestrates the migration of Hugging Face model inference to Google Cloud TPU using the torch_tpu API and benchmarks performance against a CPU baseline. Acts as the orchestrator agent."
tools:
  - view_file
  - write_to_file
  - replace_file_content
  - run_command
  - invoke_subagent
subagent: true
---

<!-- disableFinding(LINE_OVER_80) -->

# TorchTPU Hugging Face Migration Orchestrator

This skill guides the end-to-end orchestration of migrating Hugging Face PyTorch
model inference to Google Cloud TPU using the `torch_tpu` API, verifying
correctness, and benchmarking performance against a CPU/GPU baseline.

## Step-by-Step Migration Workflow

Always adhere strictly to the following phases. **Never bypass steps** for
perceived efficiency.

### 📋 Mandatory Orchestration Workflow Checklist

You **MUST** follow this step-by-step checklist during execution. At each phase
of your work, check off the step you are executing and ensure you complete the
next step:

-   [ ] **Step 0.1: Remote Runtime Setup**: Read/verify SSH connection details
    (check `~/.ssh/config` or SSH configuration).
-   [ ] **Step 0.2: Hugging Face Requirements**: Skip gating/auth prompts for
    public models.
-   [ ] **Step 0.3: Remote Hardware Diagnostics**: Read `remote_tpu_runner`
    skill via `view_file`, verify SSH, and run hardware profiling.
-   [ ] **Step 0.4: Model Discovery**: Read `smart_hf_search` skill via
    `view_file` and discover candidate models.
-   [ ] **Step 0.5: Pre-flight Compatibility Check**: **MANDATORY**: Read
    `preflight_check` skill via `view_file` AND execute `preflight_check.py` for
    **every single candidate model** (locally or remotely). **NEVER SKIP THIS
    STEP.**
-   [ ] **Step 1: Configure Models JSON**: Write candidate model list to
    `models.json`.
-   [ ] **Step 2: Stage Benchmark Scripts**: Read `torch_coder` skill via
    `view_file` and use it to copy/generate `benchmark_tpu.py` and
    `run_benchmarks.py`, then scp files to remote VM.
-   [ ] **Step 3: Execute Remote Benchmarks**: Execute `run_benchmarks.py` on
    the remote TPU VM.
-   [ ] **Step 4: Download Results & Host Cleanup**: Scp `results.json` back,
    run SSH host cleanup (`rm -rf ~/benchmarks ~/models.json ~/results.json`),
    and compile final report artifact.

### Step 0: Environment Setup & Information Gathering

### Step 0.1: Remote Runtime Setup

Before executing the migration, ensure the user has provided the following
details:

-   **`remote_host` (Required)**: IP or hostname of the target runtime TPU VM.
-   **`ssh_user` (Required)**: SSH username for the remote host.
-   **`ssh_private_key_path` (Optional)**: Local path to the SSH private key for
    authorization.

#### Identify the right VM to use for benchmarking

Follow these instructions when the user hasn't provided information of the
remote VMs that have the required accelerators.

1.  To begin first check the `~/.ssh/config` file to check if the info was
    added.
2.  If `~/.ssh/config` contains any remote host info, **ALWAYS** cross check
    with the user if this IP/Host has the necessary accelerators required for
    benchmarking.
3.  If `~/.ssh/config` does not contain any info or the user confirms the VM
    details you found don't have access to the accelerators:

    3.1. **STOP IMMEDIATELY AND ASK** the user to share the remote VM details.
    3.2. Update the ~/.ssh/config file with these VM details to ensure future
    sessions can automatically find this information. Make sure to also add a
    comment `# Added by TorchTPU Agent` to the entry.

> [!!CAUTION!!]
>
> ### MANDATORY HARD STOP
>
> If valid remote VM host details (`remote_host`, `ssh_user`) are not found in
> `~/.ssh/config` and have not been provided by the user: 1. **YOU MUST
> IMMEDIATELY STOP EXECUTION AND ASK THE USER FOR VM DETAILS.** 2. **DO NOT**
> proceed to model discovery, script generation, benchmarking, or report
> creation until SSH connectivity to the remote VM with physical accelerators
> has been verified.

### Step 0.2: **MANDATORY** HuggingFace Pre-requisites setup

Before you discovery model and profile hardware, please make to complete these
steps to ensure seamless access to models:

1.  Use the `ask_question` tool to ask the user if they would like to setup
    huggingface authentication to access gated models. If the user agrees,
    proceed to step 2. Else, explicitly print a CAUTION message for them stating
    that gated models will not be accessible.

2.  **IMPORTANT** DO NOT run the commands yourself. Instead request the user to
    login to HuggingFace using the `huggingface_hub` CLI for gated model access
    on both local machine and remote VM.

    Ask user to install huggingface_hub and run directly. They can use any
    python interpreter already available on their remote VM for this step:

    ```bash
    pip install --upgrade huggingface_hub
    hf auth login
    ```

### Step 0.3: Remote Hardware Diagnostics

Before discovering or validating models, you MUST inspect the target remote VM's
hardware to identify the available accelerator and memory capacity. Use the
**[remote-tpu-runner Skill](../remote_tpu_runner/SKILL.md)** to: 1. Verify the
remote connection. 2. Execute the inspect/diagnostics script to retrieve the
specs JSON (containing `tpu.type`, `tpu.hbm_gb`, `gpu.available`, and
`ram_gb`). 3. Parse and identify the `{tpu_type}` (e.g. `v5e-4`) and Memory
Limit (`hbm_gb`) from the output.

### Step 0.4: Model Discovery & Capacity Integration

If the user request specifies criteria (e.g., "trending transformer models")
instead of a specific model ID, utilize the
**[smart-hf-search Skill](../smart_hf_search/SKILL.md)** to find candidates.

To prevent Out-Of-Memory (OOM) compilation crashes on the remote TPU
accelerator, you **MUST** apply safe parameter limits on the search query based
on the `hbm_gb` limit resolved in Step 0.1.

#### Real TPU Memory-to-Parameter Boundary Calculations (Relaxed but Conservative)

Models loaded in `bfloat16` format require $2 \times \text{Params}$ GB memory.
To ensure safe compilation and KV cache serving headroom buffers during
PyTorch/XLA execution:

Actual TPU Accelerator Configuration                  | Total HBM Capacity | Safe Search Parameter Limit (`--max_params`) | Weight Footprint (bfloat16) | Notes / Serving Limit
:---------------------------------------------------- | :----------------: | :------------------------------------------: | :-------------------------: | :--------------------
**TPU v5e-1 (1 chip)**                                | `16 GB`            | **`7B`**                                     | `~14 GB`                    | Maximally relaxed. Fits 7B architectures (e.g. Mistral 7B / Qwen 7B) with restricted sequence bounds.
**TPU v6e-1 (1 chip)**                                | `32 GB`            | **`15B`**                                    | `~30 GB`                    | Relaxed limit. Fits standard 7B-14B scales (e.g. Llama 3.1 8B / Qwen 2.5 14B).
**TPU v5e-4 (4 chips)**                               | `64 GB`            | **`30B`**                                    | `~60 GB`                    | Relaxed limit. Fits higher-scale models (e.g. Gemma 2 27B / Llama 2 13B).
**TPU v6e-4 (4 chips) / TPU v4-8 (8 cores, 4 chips)** | `128 GB`           | **`50B`**                                    | `~100 GB`                   | Relaxed limit. Accommodates Mixture-of-Experts and advanced LLMs (e.g. Mixtral 8x7B / Qwen 32B).
**TPU v5p-8 (8 cores, 4 chips)**                      | `380 GB`           | **`130B`**                                   | `~260 GB`                   | Relaxed limit. Fits large-scale serving foundation targets (e.g. Llama 3 70B / Command-R+).

#### Inter-Skill Event Mapping

When parsing candidates returned in JSON format by the search skill, route
metadata fields as follows:

-   **`library` & `pipeline` metadata**: Match the candidate's pipeline tags
    (`text-generation`, `fill-mask`, `feature-extraction`) to determine model
    architecture parameters for the subsequent phases.

#### High-Diversity Model Gathering & Interleaving (Generic & Scalable)

When the user requests a verified list of `{C}` compatible Hugging Face models
from a broad representation of `{L}` PyTorch libraries (e.g., `{C}=20`
compatible models from `{L}=9` distinct library tags including `transformers`,
`diffusers`, `timm`, `peft`, `trl`, `sentence-transformers`, `pyannote.audio`,
`open_clip`, and `speechbrain` / `espnet`; or general large scales such as
`{C}=500` models from `{L}=20` libraries):

1.  **Perform Redundant Search by Library Tag**:

    *   Trigger the **[smart-hf-search Skill](../smart_hf_search/SKILL.md)**
        once for each of the `{L}` distinct library tags.
    *   Since local pre-flight checks can reject models due to gating issues,
        incompatible formats, custom forbidden imports, or unsupported GPU
        dependencies, you **MUST query a redundant/buffer number of candidates
        `{K}` per library tag** to ensure you have a robust buffer of backup
        models to choose from.
    *   *Sizing Guideline*: Choose `{K}` such that `{K} >= 1.5 * (target models
        expected per library)`. For example, if aiming for `{C}=20` across 9
        libraries, choose `{K}=10` or `{K}=15` per tag. If scaling up to
        `{C}=500` models across 20 libraries (averaging ~25 models per library),
        select a buffer of `{K}=40` or `{K}=50` raw matches per tag.

2.  **Interleave the Candidates for Maximum Tag Diversity**:

    *   To prevent a single domain or library (e.g., `transformers`) from
        filling your entire quota of verified models early on and destroying tag
        diversity, you **MUST NOT** perform pre-flight checks sequentially
        library-by-library.
    *   Incorporate all raw candidates and **interleave them using a round-robin
        strategy** across the libraries:
        *   Round 1: 1st model from Library 1, 1st model from Library 2, ...,
            1st model from Library N.
        *   Round 2: 2nd model from Library 1, 2nd model from Library 2, ...,
            2nd model from Library N.
        *   Continue this pattern until all retrieved candidate lists are
            completely exhausted.

3.  **Progressive Pre-flight Filtering**:

    *   Execute the **[preflight-check Skill](../preflight_check/SKILL.md)**
        sequentially on this interleaved list of candidate models.
    *   Track the models that successfully pass all validation steps.
    *   Feed models one-by-one through the validator until the requested target
        quota `{C}` of verified, compatible models is successfully accumulated,
        or the interleaved candidate list is fully exhausted.

#### Handling Batch Model Family Resolution (e.g. `important models`)

When the user requests benchmarking a predefined list of model families (e.g.,
`important models` list in the prompt) rather than concrete model IDs:

1.  **Parse the list** of model families directly from the prompt.
2.  **Build the search tool** locally to run it without Blaze overhead:

    ```bash
    bazel build //learning/agents/tern/torch_agent/_agents/skills/smart_hf_search/scripts:smart_hf_search.par
    ```

3.  **Resolve families in parallel**: Create a temporary Python scratch script
    (in the task's scratch directory) to call `smart_hf_search.par` concurrently
    (e.g., using `ThreadPoolExecutor` with 10 workers) for each family. Use
    `--k=3` to retrieve multiple candidates for redundancy:

    ```bash
    python3 {workspace_root}/bazel-bin/learning/agents/tern/torch_agent/_agents/skills/smart_hf_search/scripts/smart_hf_search.par \
      --search={family} \
      --sort=downloads \
      --k=3 \
      --json
    ```

4.  **Filter by HBM Capacity**: Parse the JSON outputs to get concrete model
    IDs. Immediately discard any resolved models whose parameter size exceeds
    the safe TPU HBM limit resolved during diagnostics (Step 0.3) to avoid OOMs
    during compilation. Keep the remaining compatible candidates ordered by
    popularity for each family.

### Step 0.5: Pre-flight Compatibility Check

Before generating benchmark scripts or executing remote runs, you **MUST**
validate the target model's compatibility and pipeline properties locally, or on
the target remote VM host as a backup option if local execution fails.

1.  **Load Skill**: You **MUST** read the
    **[preflight-check Skill](../preflight_check/SKILL.md)** using `view_file`
    to understand how to validate Hugging Face model compatibility. Never bypass
    loading this skill.
2.  **Execute Preflight Validation**: Follow the workflow instructions in
    `preflight_check/SKILL.md` to run preflight compatibility checks on each
    candidate model (locally, or via SSH on the remote host as a backup option
    if local execution fails) before proceeding to full benchmarking.

### Step 1: Create a file with the list of models to remote VM

We will start by creating the list of compatible models to benchmarking on the
remote VM under the file `~/benchmarks/models.json`. Here is an example file
that benchmarks models on `cpu` and `tpu`:

```json
{
    "models": [
        {
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "library": "transformers",
            "pipeline_tag": "text-generation",
            "device": "cpu",
            "clean_cache": false
        },
        {
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "library": "transformers",
            "device": "tpu",
            "pipeline_tag": "text-generation",
            "clean_cache": true
        },
        {
            "model_id": "stabilityai/stable-diffusion-3-5-large",
            "library": "diffusers",
            "device": "cpu",
            "pipeline_tag": "text-to-image",
            "clean_cache": false
        },
        {
            "model_id": "stabilityai/stable-diffusion-3-5-large",
            "library": "diffusers",
            "device": "tpu",
            "pipeline_tag": "text-to-image",
            "clean_cache": true
        }
    ]
}
```

Similarly, you will need to create a separate file with `cuda` (and optionally
`cpu`) as device for benchmarking on GPU VM.

**TIP**: As shown in the above sample JSON, the models file can contain multiple
entries for the same model_id and library, but with different devices and
clean_cache values. If a single remote VM has multiple accelerators that you are
interested in benchmarking, you can add multiple entries for the same model_id
and library with different device values. This allows the same remote VM to
benchmark the models on multiple accelerators. For more details on what each of
these fields represent, check the `benchmark_tpu.py` file within torch_coder
skill.

**NOTE**: If you need to benchmark the models on multiple accelerators, make
sure to copy the file onto every VM that has your intended accelerators.

### Step 2: Benchmark Script Generation

Utilize the **[torch-coder Skill](../torch_coder/SKILL.md)** to generate or
modify the python execution benchmarking script (e.g., `benchmark_tpu.py`):

1.  Identify the model's architecture/task (mapping to Causal LM, SpeechSeq2Seq,
    Feature Embedding, or Vision-Language patterns).
2.  Incorporate standard PyTorch-to-TPU device placement dynamically using
    `api.tpu_device()`.
3.  Ensure proper warmup phases are added to exclude compilation/graph overhead
    from speedup calculations.

### Step 3: Remote Running & Verification

Use the **[remote-tpu-runner Skill](../remote_tpu_runner/SKILL.md)** to run
targets on the remote VM host:

1.  Stream both the `run_benchmarks.py` orchestrator script and the
    `benchmark_tpu.py` script to the remote VM.
2.  Execute the orchestrator script `run_benchmarks.py` pointing it to the
    `models.json` file on the remote VM. This script will automatically handle
    iterating over the models, running them on the appropriate devices,
    pre-fetching weights, handling errors, and saving results to `results.json`.
3.  Collect execution stdout/stderr payloads and the `results.json` file into
    the localized workspace environment.

### Step 4: Cleanup and Speedup Report Generation

1.  **MANDATORY Remote Host Cleanup**: Immediately after fetching `results.json`
    from the remote host back into the local workspace, you **MUST** run an SSH
    cleanup command to delete all temporary benchmark scripts, configuration
    files, and temporary directories created on the remote VM (e.g. `ssh ... "rm
    -rf ~/benchmarks ~/results.json ~/models.json"`). **Never skip this step**,
    as leaving stale files on the target host risks contamination and disk bloat
    for future benchmarking runs.
2.  **Report Generation**: Compile the comparative performance timing, speedup
    ratios, and error analysis reports locally into `benchmark_report.md`.

--------------------------------------------------------------------------------

## Verification & Benchmarking Rules

*   **Warmup Execution**: You **must** trigger a warmup run on the remote host
    before recording execution times. XLA compiles graphs on-the-fly; first run
    timings represent compile latencies rather than actual runtime performance.
*   **Determinism and Correctness Check**: For text generation models where
    sampling is turned off (`do_sample=False`), enforce strict CPU vs TPU
    semantic comparisons. For sampling-enabled configurations, compare outputs
    using semantic coherence verification.

--------------------------------------------------------------------------------

## Dependencies

-   [remote-tpu-runner (VM Connection & Execution)](../remote_tpu_runner/SKILL.md)
-   [smart-hf-search (Hugging Face Search)](../smart_hf_search/SKILL.md)
-   [preflight-check (Compatibility check)](../preflight_check/SKILL.md)
-   [torch-coder (Benchmark Generation)](../torch_coder/SKILL.md)

## Generated execution contract

- Validate the request against `.coworker/torchtpu-agents/schemas/task-request.json` before work.
- Require the request's `run_id` and keep every output under `.coworker/torchtpu-agents/runs/<run_id>/artifacts/`.
- Validate the result against `.coworker/torchtpu-agents/schemas/task-result.json` before returning.
- Use `python3 .coworker/torchtpu-agents/runtime/runtime.py validate --schema SCHEMA INSTANCE`.
- Return the common result envelope with `completed`, `invalid_input`, `needs_input`, or `failed`.
- Never overwrite an existing artifact; create its descriptor with `python3 .coworker/torchtpu-agents/runtime/runtime.py describe-artifact --workspace . --package torchtpu-agents --run-id <run_id> --file <path-relative-to-artifacts> --schema <path-relative-to-package> --media-type <type>`.
- Reject artifact references whose URI does not contain the request's exact package and `run_id`.
- Materialize durable outputs and return artifact descriptors; do not copy payloads into messages.
- Allowed delegation targets: preflight-check, remote-tpu-runner, smart-hf-search, torch-coder, torch-inference-coder, torch-tpu-inference.
- Run sequentially. Never weaken inherited permissions or approvals.
