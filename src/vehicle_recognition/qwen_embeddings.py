"""Image embeddings from the official Qwen3-VL-Embedding-2B checkpoint.

Uses the authors' image-only instruction and last non-padding token pooling:
https://github.com/QwenLM/Qwen3-VL-Embedding
No class names, labels, generated descriptions, or reranker enter the encoder.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from torch.nn import functional as F

from .similarity_efficientnet import as_rgb_image, file_sha256, resolve_device
from .configuration import QWEN_DEFAULT_OPTIONS, qwen_options

MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
REVISION = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
INSTRUCTION = QWEN_DEFAULT_OPTIONS["instruction"]
MIN_PIXELS = QWEN_DEFAULT_OPTIONS["min_pixels"]
MAX_PIXELS = QWEN_DEFAULT_OPTIONS["max_pixels"]


def model_snapshot(allow_download: bool = False, force: bool = False) -> Path:
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(
            MODEL_ID, revision=REVISION, local_files_only=not allow_download,
            force_download=force,
            allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors"],
        ))
    except OSError as error:
        if not allow_download:
            raise FileNotFoundError("Qwen weights are not cached. Run prepare with the Qwen configuration first.") from error
        raise


def snapshot_metadata(path: Path) -> dict:
    files = {p.relative_to(path).as_posix(): file_sha256(p) for p in sorted(path.rglob("*")) if p.is_file()}
    if not any(name.endswith(".safetensors") for name in files):
        raise FileNotFoundError("Qwen checkpoint snapshot has no model weights")
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"backbone": "qwen3_vl_embedding_2b", "weights": MODEL_ID, "revision": REVISION,
            "checkpoint_sha256": fingerprint, "feature_dimension": 2048, "files_sha256": files}


def prepare_qwen_weights(force: bool = False) -> dict:
    path = model_snapshot(allow_download=True, force=force)
    return {**snapshot_metadata(path), "directory": str(path)}


def last_token_embedding(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pool the last non-padding token, including batches with different lengths."""
    positions = mask.shape[1]-1-mask.flip(dims=[1]).argmax(dim=1)
    pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), positions]
    return F.normalize(pooled.float(), p=2, dim=-1)


class QwenFeatureExtractor:
    """Frozen 4-bit CUDA encoder with the same encode() contract as the CNNs."""

    feature_dim = 2048

    def __init__(self, device: str = "auto", min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS, instruction: str = INSTRUCTION):
        options = qwen_options({"min_pixels": min_pixels, "max_pixels": max_pixels, "instruction": instruction})
        self.min_pixels, self.max_pixels = options["min_pixels"], options["max_pixels"]
        self.instruction = options["instruction"]
        self.device = resolve_device(device)
        if self.device != "cuda":
            raise ValueError("The initial Qwen 4-bit backend requires CUDA")
        try:
            import transformers
            import bitsandbytes
            from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
        except ImportError as error:
            raise ImportError("Install the Qwen backend with pip install -r requirements-qwen.txt") from error
        path = model_snapshot()
        self.metadata = snapshot_metadata(path)
        quantization = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        # Match the checkpoint's native wrapper when loading quantized tied weights.
        # Inference calls only .model below; the language-model head is never run.
        self.model, loading = Qwen3VLForConditionalGeneration.from_pretrained(
            path, local_files_only=True, dtype=torch.bfloat16,
            quantization_config=quantization, device_map={"": self.device},
            attn_implementation="sdpa", output_loading_info=True,
        )
        if loading["missing_keys"] or loading.get("mismatched_keys"):
            raise ValueError(f"Qwen checkpoint did not load completely: missing={loading['missing_keys'][:5]}, "
                             f"mismatched={loading.get('mismatched_keys', [])[:5]}")
        self.model.eval().requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(
            path, local_files_only=True, min_pixels=self.min_pixels, max_pixels=self.max_pixels,
            padding_side="right",
        )
        self.processor.image_processor.size = {"shortest_edge": self.min_pixels, "longest_edge": self.max_pixels}
        self.processor.tokenizer.padding_side = "right"
        self.metadata.update({
            "preprocessing_mode": "qwen_native", "preprocessing": repr(self.processor.image_processor),
            "instruction": self.instruction, "pooling": "last_nonpadding_token_after_generation_prompt",
            "normalization": "l2", "projection_head": False,
            "quantization": "bitsandbytes_nf4_double_quant", "compute_dtype": "bfloat16",
            "attention": "sdpa", "min_pixels": self.min_pixels, "max_pixels": self.max_pixels,
            "forward": "Qwen3VLForConditionalGeneration.model (no logits or generation)",
            "transformers": transformers.__version__, "bitsandbytes": bitsandbytes.__version__,
            "torch": torch.__version__, "implementation_sha256": file_sha256(Path(__file__)),
        })

    def encode(self, images: list, batch_size: int = 1) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        if not images:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        embeddings = []
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                batch = [as_rgb_image(image) for image in images[start:start+batch_size]]
                messages = [[
                    {"role": "system", "content": [{"type": "text", "text": self.instruction}]},
                    {"role": "user", "content": [{"type": "image", "image": image}]},
                ] for image in batch]
                text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                inputs = self.processor(
                    text=text, images=batch, padding=True, return_tensors="pt",
                    images_kwargs={"min_pixels": self.min_pixels, "max_pixels": self.max_pixels},
                ).to(self.device)
                output = self.model.model(**inputs, use_cache=False, return_dict=True)
                vectors = last_token_embedding(output.last_hidden_state, inputs["attention_mask"])
                if vectors.shape != (len(batch), self.feature_dim) or not torch.isfinite(vectors).all():
                    raise ValueError("Qwen returned invalid image embeddings")
                embeddings.append(vectors.cpu().numpy())
        return np.concatenate(embeddings).astype(np.float32, copy=False)
