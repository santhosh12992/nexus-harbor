---
name: torch-coder
description: "Generates and refactors benchmark execution scripts tailored for custom Hugging Face models and PyTorch/XLA on TPU VM runners."
tools:
  - view_file
  - write_to_file
  - replace_file_content
  - run_command
subagent: true
---

# Torch Coder Skill (Benchmark Generator)

This skill specifies how to generate or refactor benchmark execution scripts
tailored for running deep learning models on Google Cloud TPU architectures
using PyTorch/XLA.

## Preconditions & Inputs

You receive candidate model metadata from prior discovery steps (e.g.
`model_id`, `library`, and `pipeline_tag`).

The skill contains a template script `benchmark_tpu.py` which already implements
all standard patches, helpers, and utility functions (e.g., `get_cache_dir()`,
`import_dependency()`, `run_benchmark_pass()`).

## Task

1.  CRITICAL: Start by creating a copy of both the `benchmark_tpu.py` and
    `run_benchmarks.py` scripts.
2.  Synthesize the benchmark_tpu.py script by **replacing the `benchmark_model`
    placeholder function** in the template with the bespoke loading and
    execution logic matching the candidate model's type or pipeline_tag.
3.  DO NOT modify the other helpers unless required to fix custom bugs.

--------------------------------------------------------------------------------

## Bespoke Loader Recipes (To replace `benchmark_model` placeholder)

Choose the recipe that matches the candidate model's tag/library and overwrite
`benchmark_model`:

-   **Causal LM (Standard LLMs - Llama, Gemma, Qwen, GPT, Mistral)**: See
    [causal_lm.md](references/causal_lm.md).
-   **Vision-Language Models (VLM - LLaVA, SmolVLM, Qwen2.5-VL, InternVL)**: See
    [vlm.md](references/vlm.md).
-   **CLIP (Multimodal Image-Text Alignment & Zero-Shot Classification)**: See
    [clip.md](references/clip.md).
-   **Whisper (Automatic Speech Recognition & Speech Seq2Seq)**: See
    [whisper.md](references/whisper.md).
-   **Wav2Vec2 (Audio Feature Representation & CTC Speech Recognition)**: See
    [wav2vec2.md](references/wav2vec2.md).
-   **Text-to-Audio / TTS (SpeechT5, Bark, AudioLDM)**: See
    [text_to_audio.md](references/text_to_audio.md).
-   **Timm (Torch Image Models & ConvNet Backbones)**: See
    [timm.md](references/timm.md).
-   **Image Classification / ViT (Vision Transformers & Torchvision)**: See
    [image_classification.md](references/image_classification.md).
-   **Seq2Seq / Translation (T5, FLAN-T5, BART)**: See
    [seq2seq.md](references/seq2seq.md).
-   **Embeddings & Masked LMs (BERT, DistilBERT, ModernBERT, RoBERTa)**: See
    [embeddings.md](references/embeddings.md).
-   **Object Detection & Document Structuring (DETR, Table Transformer,
    Grounding DINO)**: See [grounding_dino.md](references/grounding_dino.md)
-   **Vision Segmentation (SAM, SegFormer, Mask2Former)**: See
    [vision_segmentation.md](references/vision_segmentation.md).
-   **Diffusers (Stable Diffusion, SDXL, Flux Image Generation)**: See
    [diffusers.md](references/diffusers.md).
-   **SpeechBrain & CLAP (Audio Separation & CLAP Embeddings)**: See
    [speechbrain.md](references/speechbrain.md) and
    [clap.md](references/clap.md).
-   **Generic Fallback Loader (Unsupported / Custom Architecture Heads)**: See
    [fallback.md](references/fallback.md).

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
