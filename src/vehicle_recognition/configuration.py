"""Project configuration, paths, and persistent asset locations."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
FEATURE_DIMENSIONS = {"efficientnet_b3": 1536, "convnext_base": 1024, "resnet50": 2048,
                      "qwen3_vl_embedding_2b": 2048}
PREPROCESSING_MODES = {"checkpoint_preset", "square_pad_resize", "qwen_native"}
QWEN_DEFAULT_OPTIONS = {"min_pixels": 4096, "max_pixels": 262144,
                        "instruction": "Represent the user's input."}


def qwen_options(options: dict | None = None) -> dict:
    if options is None:
        options = {}
    if not isinstance(options, dict) or set(options) - QWEN_DEFAULT_OPTIONS.keys():
        raise ValueError("Qwen options must contain only min_pixels, max_pixels and instruction")
    result = {**QWEN_DEFAULT_OPTIONS, **options}
    if any(type(result[key]) is not int for key in ("min_pixels", "max_pixels")):
        raise ValueError("Qwen pixel limits must be integers")
    if not 4096 <= result["min_pixels"] <= result["max_pixels"]:
        raise ValueError("Qwen requires 4096 <= min_pixels <= max_pixels")
    if not isinstance(result["instruction"], str) or not result["instruction"].strip():
        raise ValueError("Qwen instruction must be a nonempty string")
    result["instruction"] = result["instruction"].strip()
    return result


def read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return data


@dataclass
class ProjectSettings:
    root: Path
    config: dict
    datasets: dict
    labels: list[str]

    def path(self, value: str | Path) -> Path:
        value = Path(value).expanduser()
        return value.resolve() if value.is_absolute() else (self.root / value).resolve()

    @property
    def gallery(self) -> Path:
        return self.path(self.config["references"]["directory"])

    @property
    def index(self) -> Path:
        name = self.config["recognition"]["backbone"] + ".npz"
        return self.path(self.config["references"]["index_directory"]) / name


def load_settings(root: Path, config_path: Path | None = None) -> ProjectSettings:
    root = root.expanduser().resolve()
    path = config_path or root / "configs/default.yaml"
    if not path.is_absolute():
        path = root / path
    config = read_yaml(path)
    catalog = read_yaml(root / config["classes_file"])
    datasets = read_yaml(root / config["datasets_file"])
    labels = catalog.get("labels")
    if not isinstance(labels, list) or not labels or not all(isinstance(x, str) and x for x in labels):
        raise ValueError("The recognition catalog must contain nonempty string labels")
    if len(set(labels)) != len(labels):
        raise ValueError("Recognition labels must be unique")
    if any(Path(label).name != label or label in {".", ".."} for label in labels):
        raise ValueError("Recognition labels must be single directory names")
    unknown = catalog["unknown_label"]
    if not isinstance(unknown, str) or not unknown or unknown in labels or config["recognition"]["unknown_label"] != unknown:
        raise ValueError("Use one consistent unknown outcome outside the reference catalog")
    if config["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device must be auto, cpu, or cuda")
    recognition = config["recognition"]
    if recognition["backbone"] not in FEATURE_DIMENSIONS:
        raise ValueError(f"Unsupported backbone: {recognition['backbone']}")
    recognition["feature_dimension"] = FEATURE_DIMENSIONS[recognition["backbone"]]
    if recognition.get("projection_head", False):
        raise ValueError("This inference baseline uses pretrained features without a projection head")
    if recognition.get("preprocessing") not in PREPROCESSING_MODES:
        raise ValueError(f"Unsupported preprocessing; choose one of {sorted(PREPROCESSING_MODES)}")
    if (recognition["backbone"] == "qwen3_vl_embedding_2b") != (recognition["preprocessing"] == "qwen_native"):
        raise ValueError("The Qwen embedding backend requires qwen_native preprocessing; CNNs use their own modes")
    if recognition["backbone"] == "qwen3_vl_embedding_2b" and recognition["weights"] != "Qwen/Qwen3-VL-Embedding-2B":
        raise ValueError("Use the dedicated Qwen/Qwen3-VL-Embedding-2B checkpoint")
    if recognition["backbone"] == "qwen3_vl_embedding_2b":
        qwen_options(recognition.get("qwen"))
    elif "qwen" in recognition:
        raise ValueError("recognition.qwen options require the Qwen backend")
    for key, expected in {"normalization": "l2", "similarity": "cosine", "class_score": "maximum_reference_similarity"}.items():
        if recognition.get(key) != expected:
            raise ValueError(f"Unsupported recognition {key}: use {expected}")
    if not -1 <= float(recognition["acceptance_threshold"]) <= 1:
        raise ValueError("Cosine acceptance threshold must be between -1 and 1")
    if int(recognition["top_k"]) < 1 or int(recognition.get("batch_size", 16)) < 1:
        raise ValueError("top_k and batch_size must be positive")
    if int(recognition.get("cpu_threads", 4)) < 0:
        raise ValueError("cpu_threads must be nonnegative; zero keeps the framework default")
    detector = config["detector"]
    if detector.get("provider") != "ultralytics":
        raise ValueError("Only the ultralytics detector provider is currently supported")
    if not 0 <= float(detector["confidence_threshold"]) <= 1:
        raise ValueError("Detection confidence must be between 0 and 1")
    if not 0 < float(detector["iou_threshold"]) <= 1:
        raise ValueError("Detection IoU threshold must be between 0 and 1")
    if int(detector["crop_margin_pixels"]) < 0 or int(detector["image_size"]) < 1:
        raise ValueError("Crop margin must be nonnegative and image size positive")
    return ProjectSettings(root, config, datasets, labels)


def configure_environment(root: Path) -> None:
    """Create application caches without modifying the host Python environment."""
    cache = root / "artifacts/cache"
    for name, directory in {
        "TORCH_HOME": "torch", "HF_HOME": "huggingface",
        "YOLO_CONFIG_DIR": "ultralytics", "MPLCONFIGDIR": "matplotlib",
    }.items():
        os.environ.setdefault(name, str(cache / directory))
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


def image_paths(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image format: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Image input does not exist: {source}")
    paths = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f"No supported images found in {source}")
    return paths
