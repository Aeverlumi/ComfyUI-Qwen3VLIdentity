from __future__ import annotations

from pathlib import Path

import folder_paths
import numpy as np
import torch
from PIL import Image


CAPTION_PROMPT = (
    "Describe this person's appearance in one short English phrase for video identity: "
    "face, skin tone, hair, and any visible glasses or head accessories. "
    "Do not describe expression, pose, clothing, or background. Output the phrase only."
)

_MODEL = None
_PROCESSOR = None
_MODEL_ID = None


def _tensor_to_pil(image) -> Image.Image:
    frame = image[0] if getattr(image, "ndim", 0) == 4 else image
    array = (255.0 * frame.detach().cpu().numpy()).clip(0, 255).astype(np.uint8)
    if array.shape[-1] == 4:
        array = array[:, :, :3]
    return Image.fromarray(array)


def _resolve_model(model_id: str) -> str:
    local = Path(folder_paths.models_dir) / "LLM" / "Qwen3-VL-4B-Instruct"
    if local.is_dir() and any(local.glob("*.safetensors")):
        return str(local)
    return model_id


def _unload() -> None:
    global _MODEL, _PROCESSOR, _MODEL_ID
    _MODEL = None
    _PROCESSOR = None
    _MODEL_ID = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load(model_id: str):
    global _MODEL, _PROCESSOR, _MODEL_ID
    resolved = _resolve_model(model_id)
    if _MODEL is not None and _MODEL_ID == resolved:
        return _MODEL, _PROCESSOR
    _unload()
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    _PROCESSOR = AutoProcessor.from_pretrained(resolved, trust_remote_code=True)
    _MODEL = Qwen3VLForConditionalGeneration.from_pretrained(
        resolved,
        torch_dtype=dtype,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    _MODEL_ID = resolved
    return _MODEL, _PROCESSOR


def _caption(image, prompt: str, max_new_tokens: int) -> str:
    model, processor = _load("Qwen/Qwen3-VL-4B-Instruct")
    pil = _tensor_to_pil(image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": prompt.strip() or CAPTION_PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil], padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = output_ids[:, inputs["input_ids"].shape[1] :]
    text_out = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return " ".join(text_out.strip().split())


class Qwen3VLIdentityCaption:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": CAPTION_PROMPT}),
                "max_new_tokens": ("INT", {"default": 64, "min": 16, "max": 128}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("identity",)
    FUNCTION = "run"
    CATEGORY = "identity"

    def run(self, image, prompt, max_new_tokens, keep_model_loaded):
        text = _caption(image, prompt, int(max_new_tokens))
        if not keep_model_loaded:
            _unload()
        return (text,)


class Qwen3VLIdentityCaption2P:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": CAPTION_PROMPT}),
                "max_new_tokens": ("INT", {"default": 64, "min": 16, "max": 128}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image_2": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("identity_1", "identity_2")
    FUNCTION = "run"
    CATEGORY = "identity"

    def run(self, image_1, prompt, max_new_tokens, keep_model_loaded=False, image_2=None):
        identity_1 = _caption(image_1, prompt, int(max_new_tokens))
        identity_2 = _caption(image_2, prompt, int(max_new_tokens)) if image_2 is not None else ""
        if not keep_model_loaded:
            _unload()
        return (identity_1, identity_2)


NODE_CLASS_MAPPINGS = {
    "Qwen3VLIdentityCaption": Qwen3VLIdentityCaption,
    "Qwen3VLIdentityCaption2P": Qwen3VLIdentityCaption2P,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3VLIdentityCaption": "Qwen3-VL-4B 身份反推",
    "Qwen3VLIdentityCaption2P": "Qwen3-VL-4B 双图身份反推",
}
