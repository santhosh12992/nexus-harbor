---
name: torch-inference-coder
description: "Generates declarative deployments.json service manifests and synthesizes the required serving wrapper scripts. Launches background daemons, polls HTTP health endpoints, and performs output verification."
tools:
  - view_file
  - write_to_file
  - replace_file_content
  - run_command
subagent: true
---

# Torch Inference Coder Skill

This skill guides the generation, deployment, and monitoring of HTTP Inference
engines on TPU wrapped around Hugging Face models.

## Inputs for Skill

-   **`model_id`**: The HuggingFace Model ID that needs to be setup for
    inference. Alternatively, `model_id` can also be local path to a model if
    the model is stored locally.
-   **`engine`**(Optional): The engine to be used for setting up the model for
    inference. Default: `custom_fastapi`
-   **`port`**(Optional): The port to use to setup the API for inference.

## Task 1: Generate `deployments.json`

Create the configuration file representing the deployment on the target host:

```json
{
  "deployments": [
    {
      "model_id": "meta-llama/Llama-3.1-8B-Instruct",
      "engine": "custom_fastapi",
      "port": 8000,
      "max_model_len": 2048
    }
  ]
}
```

## Task 2: Synthesize API Server Script / Launcher

Check the `engine` specified in the `deployments.json`. Generate the appropriate
engine launcher:

### Recipe: `custom_fastapi` (Native PyTorch)

If native PyTorch execution is requested:

1.  First, make a carbon copy of the backbone execution runtime:
    `skills/torch_inference_coder/scripts/inference.py` onto the target VM.
2.  Determine the model's pipeline type (e.g. Causal LM, Embedding,
    Vision-Language).
3.  Reference the corresponding Python API overrides template:
    *   Text Generation / Causal LM:
        [text_generation.md](references/fastapi_server/text_generation.md)
    *   Embeddings: [embeddings.md](references/fastapi_server/embeddings.md)
    *   Vision-Language (VLM): [vlm.md](references/fastapi_server/vlm.md)
    *   Diffusion (Text-to-Image):
        [diffusion.md](references/fastapi_server/diffusion.md)
4.  Implement the `AGENT TODO` gaps inside the `inference.py` script precisely
    following the logic block templates provided in your chosen reference. You
    must:
    -   Override the `Model Initialization` logic (ensure `get_device('tpu')` is
        used to bootstrap safely).
    -   Setup the `Payload Schema` request model targeting your pipeline format.
    -   Write the `Generate Endpoint API`.
5.  **(Optional Verification)**: Copy both
    `skills/torch_inference_coder/scripts/inference.py` and
    `skills/torch_inference_coder/scripts/inference_test.py` to the target host
    and run `python3 inference_test.py` to assert Pydantic parsing and endpoint
    routing are stable before pushing the daemon into the background.

## Task 3: Launch and Monitor Deployment

Transfer the generated files (e.g. `deployments.json` and `inference.py`) to the
target TPU VM. Launch the API server backend in a daemon mode:

```bash
nohup python3 fastapi_server.py > inference.log 2>&1 &
echo $! > fastapi_pid.txt
```

Continually poll the `/health` endpoint until the compiled model successfully
surfaces readiness guarantees:

```bash
curl -s http://localhost:{port}/health
```

## Task 4: Output Verification

Send a test payload to the newly created endpoint (e.g. `/generate`, `/embed`,
or `/vqa`) and confirm payload formulation/functional accuracy. Example for a
generative model:

```bash
curl -X POST http://localhost:{port}/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello!"}'
```

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
