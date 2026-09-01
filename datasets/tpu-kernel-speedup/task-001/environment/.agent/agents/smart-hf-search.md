---
name: smart-hf-search
description: "Search for Hugging Face models and libraries based on various criteria like popularity, trending status, parameter count, and architecture. Use this to find models to migrate or benchmark."
tools:
  - view_file
  - write_to_file
  - replace_file_content
  - run_command
subagent: true
---

# Smart Hugging Face Search Skill

A skill to search for Hugging Face models and libraries using the
`smart_hf_search` tool located at
[smart_hf_search.py](scripts/smart_hf_search.py).

## Workflow & Execution Rules

When executing this skill:

1.  Read and adhere to [AGENTS.md](../../AGENTS.md).
2.  Use the `smart_hf_search` script via `bazel run` or standard `python3`.
3.  Map the user's natural language request to the appropriate command-line
    flags.

### Running the Tool

Run the script standalone (after installing `huggingface_hub` with `pip`):

```bash
python3 learning/agents/tern/torch_agent/_agents/skills/smart_hf_search/scripts/smart_hf_search.py [flags]
```

--------------------------------------------------------------------------------

## Query Examples and Flag Mappings

The examples below show standalone execution prioritizing **Scenario B
(open-source)**. If running inside Google3, prepend the flags to `bazel run
//learning/agents/tern/torch_agent/_agents/skills/smart_hf_search/scripts:smart_hf_search
-- [flags]`.

### 1. Top-K Hugging Face models (General popular models)

```bash
python3 learning/agents/tern/torch_agent/_agents/skills/smart_hf_search/scripts/smart_hf_search.py --k=5 --sort=downloads
```

### 2. Top-K popular or trending Hugging Face libraries

```bash
python3 learning/agents/tern/torch_agent/_agents/skills/smart_hf_search/scripts/smart_hf_search.py --k=5 --mode=libraries
```

### 3. Top-K trending Hugging Face models this week

```bash
python3 learning/agents/tern/torch_agent/_agents/skills/smart_hf_search/scripts/smart_hf_search.py --k=5 --sort=trending
```

### 4. Top-K trending Hugging Face models with transformer architecture

We filter by the `transformers` library as most transformer models use it:

```bash
python3 learning/agents/tern/torch_agent/_agents/skills/smart_hf_search/scripts/smart_hf_search.py --k=5 --sort=trending --library=transformers
```

### 5. Top-K trending Hugging Face models with less than 20B parameters

```bash
python3 learning/agents/tern/torch_agent/_agents/skills/smart_hf_search/scripts/smart_hf_search.py --k=5 --sort=trending --max_params=20B
```

--------------------------------------------------------------------------------

## Available Flags

To view the complete and up-to-date list of available command-line flags and
descriptions, run the script with the `--help` flag:

```bash
python3 learning/agents/tern/torch_agent/_agents/skills/smart_hf_search/scripts/smart_hf_search.py --help
```

--------------------------------------------------------------------------------

## Time-Based Trending Queries (API Limitations)

The Hugging Face API only supports a weekly window for the `trending` score. Map
time-based queries as follows:

*   **"this week" / "trending" (default)**: Use `--sort=trending`.
*   **"this month" / "this year" / "overall"**: Hugging Face does not support
    monthly trending API queries. Fallback to `--sort=downloads` or
    `--sort=likes` to approximate long-term popularity, and inform the user of
    the fallback.

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
