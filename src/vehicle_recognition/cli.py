"""Inspect scaffold configuration without downloading data or model weights."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _read_mapping(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def inspect_scaffold(root: Path, check_runtime: bool = False) -> dict:
    """Check the configured catalog/gallery and report implementation status."""
    config = _read_mapping(root / "configs/default.yaml")
    if config["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device must be auto, cpu, or cuda")
    catalog = _read_mapping(root / config["classes_file"])
    datasets = _read_mapping(root / config["datasets_file"])
    labels = catalog["labels"]
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("The recognition catalog must contain unique labels")
    if catalog["unknown_label"] in labels:
        raise ValueError("The unknown outcome must not be a gallery class")
    gallery = root / config["references"]["directory"]
    if not gallery.is_dir():
        raise ValueError(f"Reference gallery is missing: {gallery}")
    unexpected = {item.name for item in gallery.iterdir() if item.is_dir()} - set(labels)
    if unexpected:
        raise ValueError(f"Unconfigured gallery classes: {', '.join(sorted(unexpected))}")
    counts = {
        label: sum(
            path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            for path in (gallery / label).glob("*")
        )
        for label in labels
    }
    empty = [label for label, count in counts.items() if not count]
    if empty:
        raise ValueError(f"Classes without reference images: {', '.join(empty)}")
    forest = datasets["forest_vehicles"]
    workspace = forest["workspace"] or os.environ.get(forest["workspace_env"])
    version = forest["version"] or os.environ.get(forest["version_env"])
    result = {
        "stage": "architecture_scaffold",
        "implemented_commands": ["status"],
        "planned_commands": ["prepare", "augment-references", "index", "predict", "evaluate"],
        "device": config["device"],
        "extractor": config["recognition"]["backbone"],
        "checkpoint": config["recognition"]["weights"],
        "reference_images": counts,
        "reference_total": sum(counts.values()),
        "roboflow_project": os.environ.get("ROBOFLOW_PROJECT") or forest["project"],
        "roboflow_workspace_required": not bool(workspace),
        "roboflow_version_required": not bool(version),
        "visdrone_download_by_default": datasets["visdrone"]["download_by_default"],
    }
    if check_runtime:
        import torch
        import torchvision

        cuda_available = torch.cuda.is_available()
        requested_device = config["device"]
        if requested_device == "cuda" and not cuda_available:
            raise ValueError("CUDA was requested but is unavailable in this environment")
        selected_device = (
            "cuda" if cuda_available else "cpu"
        ) if requested_device == "auto" else requested_device
        result["runtime"] = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": cuda_available,
            "selected_device": selected_device,
        }
        if cuda_available:
            result["runtime"]["gpu"] = torch.cuda.get_device_name(0)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aerial vehicle recognition scaffold. Experiment execution is not implemented yet."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="Validate configuration and count reference images")
    status.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root (default: current directory)")
    status.add_argument("--check-runtime", action="store_true", help="Also inspect installed PyTorch and CUDA availability")
    args = parser.parse_args(argv)
    try:
        result = inspect_scaffold(args.root, args.check_runtime)
    except (OSError, ValueError, KeyError, TypeError, ImportError, yaml.YAMLError) as error:
        print(f"Configuration check failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0
