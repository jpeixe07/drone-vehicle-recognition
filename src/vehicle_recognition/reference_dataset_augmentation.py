"""Deterministic reference augmentation with an explicit parent-image manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil

import albumentations as A
import cv2
import numpy as np
from PIL import Image

from .reporting import write_json
from .similarity_efficientnet import as_rgb_image, file_sha256, gallery_manifest


def augment_dataset(input_dir: Path, output_dir: Path, num_variants: int = 5,
                    seed: int = 42, labels: list[str] | None = None) -> dict:
    source, destination = Path(input_dir).resolve(), Path(output_dir).resolve()
    if destination == source or destination.is_relative_to(source):
        raise ValueError("Augmented references must be written outside the source gallery")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Choose an empty output directory for reference augmentation")
    if num_variants < 0:
        raise ValueError("The number of variants cannot be negative")
    original = gallery_manifest(source, labels)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "seed": seed, "variants_per_original": num_variants,
                "albumentations": A.__version__,
                "source_gallery_sha256": original["sha256"], "images": []}
    for entry in original["entries"]:
        parent = source / entry["path"]
        copied = destination / entry["path"]
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent, copied)
        manifest["images"].append({"path": entry["path"], "parent": entry["path"], "generated": False,
                                   "sha256": entry["sha256"]})
        # Existing augmented examples remain available but are not augmented recursively.
        if "_aug_" in parent.stem:
            continue
        image_seed = int.from_bytes(hashlib.sha256(f"{seed}:{entry['path']}".encode()).digest()[:4], "big")
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=(-180, 180), p=1.0, border_mode=cv2.BORDER_CONSTANT, fill=0),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=(-30, 30), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        ], seed=image_seed)
        manifest.setdefault("transform", A.to_dict(transform))
        pixels = np.asarray(as_rgb_image(parent))
        for variant in range(num_variants):
            identifier = hashlib.sha256(entry["path"].encode()).hexdigest()[:8]
            name = f"{parent.stem}_{identifier}_aug_{variant:03}.jpg"
            target = destination / entry["label"] / name
            Image.fromarray(transform(image=pixels)["image"]).save(target, quality=95)
            manifest["images"].append({"path": target.relative_to(destination).as_posix(),
                                       "parent": entry["path"], "generated": True,
                                       "seed": image_seed, "variant": variant, "sha256": file_sha256(target)})
    write_json(destination / "augmentation.json", manifest)
    return {"directory": str(destination), "images": len(manifest["images"]),
            "generated": sum(item["generated"] for item in manifest["images"]), "seed": seed}
