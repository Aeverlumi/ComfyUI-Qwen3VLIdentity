# ComfyUI-Qwen3VLIdentity

Two ComfyUI nodes that use Qwen3-VL-4B-Instruct to describe a person's visible identity from one or two images.

## Nodes

- `Qwen3VLIdentityCaption`: one image to one short identity description.
- `Qwen3VLIdentityCaption2P`: two images to two descriptions; the second image is optional.

The default prompt focuses on face, skin tone, hair, glasses, and head accessories. You can edit the prompt in either node. The nodes return text and do not generate or modify images.

## Install

1. Place this folder at `ComfyUI/custom_nodes/ComfyUI-Qwen3VLIdentity`.
2. In ComfyUI's Python environment, install `requirements.txt`. The environment also needs a Qwen3-VL-compatible `transformers` installation and its model-loading dependencies.
3. Restart ComfyUI.

The nodes first look for model files in `ComfyUI/models/LLM/Qwen3-VL-4B-Instruct`. If that directory is unavailable, they load `Qwen/Qwen3-VL-4B-Instruct` through Hugging Face, which may download the model on first use. Model weights are not included in this repository.

`keep_model_loaded` retains the model for subsequent calls. Leave it off when GPU memory is needed by the next video-generation node.
