"""Offline image recognition and explicit experiment preparation commands."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys

from .configuration import FEATURE_DIMENSIONS, IMAGE_SUFFIXES, configure_environment, load_settings

IMPLEMENTED_COMMANDS = ["status", "prepare", "download-dataset", "index", "predict", "collect-crops", "augment-references"]


def inspect_project(settings, check_runtime: bool = False) -> dict:
    gallery = settings.gallery
    if not gallery.is_dir():
        raise FileNotFoundError(f"Reference gallery is missing: {gallery}")
    unexpected = {path.name for path in gallery.iterdir() if path.is_dir()} - set(settings.labels)
    if unexpected:
        raise ValueError(f"Unconfigured gallery classes: {', '.join(sorted(unexpected))}")
    counts = {label: sum(path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                         for path in (gallery / label).rglob("*")) for label in settings.labels}
    empty = [label for label, count in counts.items() if not count]
    if empty:
        raise ValueError(f"Classes without reference images: {', '.join(empty)}")
    forest = settings.datasets["forest_vehicles"]
    result = {
        "stage": "offline_pipeline", "implemented_commands": IMPLEMENTED_COMMANDS,
        "planned_commands": ["evaluate"], "device": settings.config["device"],
        "extractor": settings.config["recognition"]["backbone"],
        "checkpoint": settings.config["recognition"]["weights"],
        "reference_images": counts, "reference_total": sum(counts.values()),
        "index_present": settings.index.is_file(),
        "detector_weights_present": settings.path(settings.config["detector"]["weights"]).is_file(),
        "roboflow_project": os.environ.get("ROBOFLOW_PROJECT") or forest["project"],
        "roboflow_workspace": os.environ.get(forest["workspace_env"]) or forest["workspace"],
        "roboflow_version": os.environ.get(forest["version_env"]) or forest["version"],
        "forest_dataset_present": (settings.path(forest["directory"]) / "data.yaml").is_file(),
        "roboflow_workspace_required": not bool(forest["workspace"] or os.environ.get(forest["workspace_env"])),
        "roboflow_version_required": not bool(forest["version"] or os.environ.get(forest["version_env"])),
        "visdrone_download_by_default": settings.datasets["visdrone"]["download_by_default"],
    }
    if check_runtime:
        from .similarity_efficientnet import resolve_device
        import torch
        import torchvision

        selected = resolve_device(settings.config["device"])
        result["runtime"] = {"python": sys.version.split()[0], "torch": torch.__version__,
                             "torchvision": torchvision.__version__, "cuda_build": torch.version.cuda,
                             "cuda_available": torch.cuda.is_available(), "selected_device": selected}
        if selected == "cuda":
            result["runtime"]["gpu"] = torch.cuda.get_device_name(0)
    return result


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline vehicle detection and reference-based recognition.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=Path, default=Path.cwd(), help="Project root (default: current directory)")
    common.add_argument("--config", type=Path, help="Configuration YAML, relative to the project root")
    common.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Override the configured compute backend")
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", parents=[common], help="Inspect configuration, reference counts, and asset readiness")
    status.add_argument("--check-runtime", action="store_true")

    prepare = commands.add_parser("prepare", parents=[common], help="Download model weights and optionally datasets")
    prepare.add_argument("--backbone", choices=sorted(FEATURE_DIMENSIONS))
    prepare.add_argument("--include-visdrone", action="store_true", help="Also download and convert VisDrone")
    prepare.add_argument("--include-forest", action="store_true", help="Also fetch the configured Roboflow export")
    prepare.add_argument("--visdrone-splits", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"])
    prepare.add_argument("--force", action="store_true", help="Refresh detector weights or reprepare selected datasets")

    download = commands.add_parser("download-dataset", parents=[common], help="Download a dataset without model weights")
    download.add_argument("dataset", choices=["forest"], help="Configured Roboflow forest dataset")
    download.add_argument("--force", action="store_true", help="Replace the local export with a fresh download")

    index = commands.add_parser("index", parents=[common], help="Build a fingerprinted reference-embedding index")
    index.add_argument("--backbone", choices=sorted(FEATURE_DIMENSIONS))
    index.add_argument("--references", type=Path, help="Override the reference gallery directory")
    index.add_argument("--index", type=Path, help="Override the generated index path")
    index.add_argument("--batch-size", type=int)

    predict = commands.add_parser("predict", parents=[common], help="Recognize vehicles in offline images or crops")
    source = predict.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="An image file or recursively scanned image directory")
    source.add_argument("--groups", type=Path, help="JSON mapping vehicle IDs to lists of already-cropped images")
    predict.add_argument("--crops", action="store_true", help="Inputs are already vehicle crops; skip YOLO")
    predict.add_argument("--backbone", choices=sorted(FEATURE_DIMENSIONS))
    predict.add_argument("--references", type=Path)
    predict.add_argument("--index", type=Path)
    predict.add_argument("--detector-weights", type=Path)
    predict.add_argument("--threshold", type=float, help="Override cosine acceptance threshold")
    predict.add_argument("--batch-size", type=int)
    predict.add_argument("--rotations", nargs="+", type=float, help="Detection views in degrees, e.g. 0 90 180 270")
    predict.add_argument("--output", type=Path, help="An empty/new directory for this run")

    collect = commands.add_parser("collect-crops", parents=[common], help="Export unlabeled detections for manual reference selection")
    collect.add_argument("--input", type=Path, required=True)
    collect.add_argument("--detector-weights", type=Path)
    collect.add_argument("--output", type=Path)

    augment = commands.add_parser("augment-references", parents=[common], help="Create a separate, seeded augmented gallery")
    augment.add_argument("--input", type=Path, help="Default: configured reference gallery")
    augment.add_argument("--output", type=Path, help="Default: data/reference_dataset_augmented")
    augment.add_argument("--variants", type=int, default=5)
    augment.add_argument("--seed", type=int, default=42)
    return parser


def dispatch(args) -> dict:
    from dotenv import load_dotenv

    root = args.root.expanduser().resolve()
    load_dotenv(root / ".env", override=False)
    settings = load_settings(root, args.config)
    if args.device:
        settings.config["device"] = args.device
    if getattr(args, "backbone", None):
        settings.config["recognition"]["backbone"] = args.backbone
        settings.config["recognition"]["feature_dimension"] = FEATURE_DIMENSIONS[args.backbone]
    if getattr(args, "references", None):
        settings.config["references"]["directory"] = str(args.references)
    if getattr(args, "detector_weights", None):
        settings.config["detector"]["weights"] = str(args.detector_weights)
    if getattr(args, "threshold", None) is not None:
        if not -1 <= args.threshold <= 1:
            raise ValueError("Cosine acceptance threshold must be between -1 and 1")
        settings.config["recognition"]["acceptance_threshold"] = args.threshold
    if getattr(args, "batch_size", None) is not None:
        if args.batch_size < 1:
            raise ValueError("Batch size must be positive")
        settings.config["recognition"]["batch_size"] = args.batch_size
    if getattr(args, "rotations", None):
        if args.crops or args.groups:
            raise ValueError("Detection rotations are not used when classifying existing crops")
        settings.config["inference"]["rotations_degrees"] = args.rotations
    if args.command == "status":
        return inspect_project(settings, args.check_runtime)
    if args.command == "download-dataset":
        from .datasets import prepare_forest

        return prepare_forest(settings, args.force)
    configure_environment(settings.root)
    import torch

    threads = int(settings.config["recognition"].get("cpu_threads", 4))
    if threads > 0:
        torch.set_num_threads(threads)
    if args.command == "prepare":
        from .datasets import prepare_assets

        return prepare_assets(settings, args.include_visdrone, args.include_forest, args.visdrone_splits, args.force)
    if args.command == "index":
        from .pipeline import build_index

        return build_index(settings, settings.path(args.index) if args.index else None)
    if args.command == "predict":
        from .pipeline import run_predictions

        return run_predictions(settings, settings.path(args.input) if args.input else None, args.crops,
                               settings.path(args.groups) if args.groups else None,
                               settings.path(args.index) if args.index else None,
                               settings.path(args.output) if args.output else None)
    if args.command == "collect-crops":
        from .pipeline import collect_crops

        return collect_crops(settings, settings.path(args.input), settings.path(args.output) if args.output else None)
    if args.command == "augment-references":
        from .reference_dataset_augmentation import augment_dataset

        return augment_dataset(settings.path(args.input) if args.input else settings.gallery,
                               settings.path(args.output) if args.output else settings.root / "data/reference_dataset_augmented",
                               args.variants, args.seed, settings.labels)
    raise ValueError(f"Unknown command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        with redirect_stdout(sys.stderr):
            result = dispatch(args)
    except Exception as error:
        message = str(error)
        for key in ["ROBOFLOW_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]:
            secret = os.environ.get(key)
            if secret:
                message = message.replace(secret, "<redacted>")
        print(f"Error: {message}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 2 if result.get("failed_images", 0) else 0
