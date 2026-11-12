"""Measured MVRSD exploration, matching the VisDrone notebook's seven sections.

Prefer the original named VOC XML labels when both XML and YOLO are present.
Never infer numeric YOLO class order from the order in the publication.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from PIL import Image
import yaml

CLASSES = ("SMV", "LMV", "AFV", "MCV", "CV")
CLASS_DESCRIPTIONS = {
    "SMV": "Small Military Vehicles", "LMV": "Large Military Vehicles",
    "AFV": "Armored Fighting Vehicles", "MCV": "Military Construction Vehicles",
    "CV": "Civilian Vehicles",
}
PALETTE = dict(zip(CLASSES, ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c969e")))
SCALES = ("Small (<32² px)", "Medium (32²–96² px)", "Large (≥96² px)")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLIT_ALIASES = {"train": "train", "training": "train", "val": "val", "valid": "val",
                 "validation": "val", "test": "test", "testing": "test"}
SOURCES = {
    "doi": "https://doi.org/10.57760/sciencedb.j00240.00036",
    "cstr": "https://cstr.cn/31253.11.sciencedb.j00240.00036.011B082C",
    "repository": "https://github.com/baidongls/MVRSD",
    "paper": "https://www.cjig.cn/en/article/doi/10.11834/jig.230884/",
}
PUBLISHED = {"images": 3000, "objects": 32626, "image_width": 640, "image_height": 640,
             "ground_sample_distance_m": 0.3, "scenarios_more_than": 40}
STYLE = {"figure.dpi": 120, "savefig.dpi": 180, "font.size": 10, "axes.titlesize": 11,
         "axes.titleweight": "bold", "axes.labelsize": 10, "axes.grid": True,
         "grid.alpha": 0.25, "grid.linestyle": "--"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_class(value: str) -> str:
    token = "".join(c for c in value.upper() if c.isalpha())
    aliases = {"".join(c for c in name.upper() if c.isalpha()): label
               for label, name in CLASS_DESCRIPTIONS.items()}
    # Accept singular forms without changing the meaning of the classes.
    aliases.update({key[:-1]: label for key, label in list(aliases.items()) if key.endswith("S")})
    result = token if token in CLASSES else aliases.get(token)
    if result not in CLASSES:
        raise ValueError(f"Unknown original MVRSD class {value!r}; expected {CLASSES}. "
                         "Use the original five-class labels, not a merged 'vehicle' export.")
    return result


def split_of(path: Path) -> str:
    for part in reversed(path.parts[:-1]):
        if part.lower() in SPLIT_ALIASES:
            return SPLIT_ALIASES[part.lower()]
    return "unspecified"


def read_annotation(path: Path, width: int, height: int, class_names: dict[int, str]) -> list[dict]:
    records = []
    if path.suffix.lower() == ".xml":
        document = ET.parse(path).getroot()
        for obj in document.findall(".//object"):
            box = obj.find("bndbox")
            if box is None:
                raise ValueError(f"Expected horizontal VOC bndbox in {path}")
            coordinates = [float(box.findtext(key, "nan")) for key in ("xmin", "ymin", "xmax", "ymax")]
            flags = {key: int(obj.findtext(key)) if obj.findtext(key) is not None else None
                     for key in ("truncated", "difficult", "occluded")}
            records.append({"class_name": canonical_class(obj.findtext("name", "")),
                            "coordinates": coordinates, **flags})
    else:
        if not class_names:
            raise ValueError(f"Numeric YOLO annotations require an explicit class mapping or data.yaml: {path}")
        for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"Expected YOLO HBB with five fields: {path}:{number}; OBB exports are not equivalent")
            class_id = int(fields[0])
            if class_id not in class_names:
                raise ValueError(f"Unknown YOLO class ID {class_id}: {path}:{number}")
            x, y, w, h = map(float, fields[1:])
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y, w, h)):
                raise ValueError(f"Expected finite normalized YOLO coordinates: {path}:{number}")
            records.append({"class_name": class_names[class_id],
                            "coordinates": [(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height],
                            "truncated": None, "difficult": None, "occluded": None})
    return records


def load_mvrsd_dataframe(root: Path, dataset_root: Path | None = None,
                         splits: tuple[str, ...] | None = None,
                         class_names: dict[int, str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Inventory all images; missing labels are unannotated, never negative images.

    The official archive uses data/images/{train,val} and
    data/labels/{train,val}/*.txt plus xml/*.xml. XML is preferred so IDs are not guessed.
    VOC coordinates use the recorded edge coordinates (xmax - xmin).
    """
    root = root.resolve()
    dataset_root = (dataset_root or root / "data/mvrsd").resolve()
    images = sorted(path for path in dataset_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No MVRSD images in {dataset_root}. Extract the official archive there; "
                                "see notebooks/README.md and the notebook's acquisition cell.")
    mapping = {int(key): canonical_class(value) for key, value in (class_names or {}).items()}
    yaml_sources = []
    if not mapping:
        for path in sorted(dataset_root.rglob("*.yaml")) + sorted(dataset_root.rglob("*.yml")):
            content = yaml.safe_load(path.read_text())
            if isinstance(content, dict) and content.get("names"):
                names = content["names"]
                candidate = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
                candidate = {key: canonical_class(value) for key, value in candidate.items()}
                if mapping and mapping != candidate:
                    raise ValueError("Conflicting YOLO class mappings; pass class_names explicitly")
                mapping = candidate
                yaml_sources.append(str(path.relative_to(dataset_root)))
    annotations = defaultdict(list)
    annotation_files = []
    for path in sorted(dataset_root.rglob("*")):
        if path.suffix.lower() not in {".xml", ".txt"}:
            continue
        if path.suffix.lower() == ".xml" or any("label" in p.lower() or "annotation" in p.lower()
                                                 for p in path.relative_to(dataset_root).parts[:-1]):
            annotations[(split_of(path), path.stem)].append(path)
            annotation_files.append(path)
    selected = [path for path in images if splits is None or split_of(path) in splits]
    if not selected:
        raise ValueError(f"No images for requested splits {splits}; available: {sorted({split_of(p) for p in images})}")
    image_rows, box_rows, inventory = [], [], []
    used = set()
    hashes = Counter()
    format_counts = Counter()
    paired_formats, paired_count_mismatches = 0, []
    for path in selected:
        split = split_of(path)
        identifier = path.relative_to(dataset_root).as_posix()
        with Image.open(path) as image:
            width, height = image.size
        candidates = annotations.get((split, path.stem), [])
        xml = [p for p in candidates if p.suffix.lower() == ".xml"]
        candidates = xml or [p for p in candidates if p.suffix.lower() == ".txt"]
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous annotations for {identifier}: {candidates}")
        annotation = candidates[0] if candidates else None
        records = read_annotation(annotation, width, height, mapping) if annotation else []
        if annotation and annotation.suffix.lower() == ".xml":
            txt = [p for p in annotations[(split, path.stem)] if p.suffix.lower() == ".txt"]
            if len(txt) == 1:
                paired_formats += 1
                txt_rows = sum(bool(line.strip()) for line in txt[0].read_text().splitlines())
                if txt_rows != len(records):
                    paired_count_mismatches.append({"image_id": identifier, "xml_objects": len(records), "txt_rows": txt_rows})
        count = 0
        for record in records:
            x1, y1, x2, y2 = record.pop("coordinates")
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                raise ValueError(f"Non-finite bounding box in {annotation}")
            out_of_bounds = x1 < 0 or y1 < 0 or x2 > width or y2 > height
            cx1, cy1, cx2, cy2 = max(0., x1), max(0., y1), min(float(width), x2), min(float(height), y2)
            w, h = cx2-cx1, cy2-cy1
            valid = w > 0 and h > 0
            area = w*h if valid else 0.
            scale = SCALES[0] if area < 32**2 else SCALES[1] if area < 96**2 else SCALES[2]
            box_rows.append({"image_id": identifier, "split": split, **record,
                             "raw_xmin": x1, "raw_ymin": y1, "raw_xmax": x2, "raw_ymax": y2,
                             "image_width": width, "image_height": height,
                             "x_left": cx1, "y_top": cy1, "w_box": w, "h_box": h,
                             "box_area": area, "rel_area_pct": area/(width*height)*100,
                             "aspect_ratio": w/h if valid else np.nan,
                             "x_center_norm": (cx1+cx2)/(2*width), "y_center_norm": (cy1+cy2)/(2*height),
                             "coco_scale": scale, "is_valid_yolo": valid, "out_of_bounds": out_of_bounds,
                             "boundary_touching": valid and (cx1 <= 1 or cy1 <= 1 or cx2 >= width-1 or cy2 >= height-1)})
            count += int(valid)
        digest = sha256(path)
        hashes[digest] += 1
        label_digest = sha256(annotation) if annotation else None
        if annotation:
            # Both formats for this image are accounted for, but objects are read once.
            used.update(annotations[(split, path.stem)])
            format_counts[annotation.suffix.lower()] += 1
        inventory.append({"image": identifier, "image_sha256": digest,
                          "annotation": str(annotation.relative_to(dataset_root)) if annotation else None,
                          "annotation_sha256": label_digest})
        image_rows.append({"image_id": identifier, "image_path": str(path), "split": split,
                           "width": width, "height": height, "resolution": f"{width}×{height}",
                           "has_annotation": annotation is not None, "valid_objects": count,
                           "raw_objects": len(records), "is_negative": annotation is not None and not records})
    images_df = pd.DataFrame(image_rows)
    boxes_df = pd.DataFrame(box_rows)
    if boxes_df.empty or not boxes_df["is_valid_yolo"].any():
        raise ValueError("No valid annotated MVRSD bounding boxes in the selected data")
    selected_splits = set(images_df["split"])
    orphans = [str(p.relative_to(dataset_root)) for p in annotation_files
               if p not in used and split_of(p) in selected_splits]
    audit = {"dataset_root": str(dataset_root), "sources": SOURCES, "published_reference": PUBLISHED,
             "acquisition": json.loads((dataset_root / "source.json").read_text()) if (dataset_root / "source.json").is_file() else None,
             "all_discovered_images": len(images),
             "images_outside_selected_splits": [str(p.relative_to(dataset_root)) for p in images if p not in selected],
             "selected_splits": sorted(selected_splits), "annotation_formats_used": dict(format_counts),
             "yolo_class_mapping": mapping, "mapping_yaml_sources": yaml_sources,
             "annotation_policy": "Prefer named VOC XML; never add paired XML and TXT object counts together",
             "paired_xml_txt_images": paired_formats, "paired_count_mismatches": paired_count_mismatches,
             "unannotated_images": int((~images_df.has_annotation).sum()),
             "explicit_negative_images": int(images_df.is_negative.sum()),
             "invalid_geometry_boxes": int((~boxes_df.is_valid_yolo).sum()),
             "out_of_bounds_boxes": int(boxes_df.out_of_bounds.sum()),
             "exact_duplicate_images": sum(n-1 for n in hashes.values()),
             "orphan_annotation_files": orphans,
             "orphan_annotation_images": sorted({f"{split_of(Path(p))}/{Path(p).stem}" for p in orphans}),
             "dataset_sha256": hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest(),
             "inventory": inventory}
    return images_df, boxes_df, audit


def save_figure(fig, directory: Path, name: str, title: str) -> str:
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(directory / name, bbox_inches="tight")
    plt.close(fig)
    return name


def plot_statistics(images: pd.DataFrame, boxes: pd.DataFrame, figures: Path) -> dict:
    annotated = images[images.has_annotation]
    valid = boxes[boxes.is_valid_yolo].copy()
    counts = valid.class_name.value_counts().reindex(CLASSES, fill_value=0)
    presence = valid.groupby("class_name").image_id.nunique().reindex(CLASSES, fill_value=0)
    frequency = pd.DataFrame({"class_name": CLASSES, "total_count": counts.values,
                              "pct_of_total": counts.values/len(valid)*100,
                              "image_presence_pct": presence.values/len(annotated)*100})
    split_counts = pd.crosstab(valid.class_name, valid.split).reindex(CLASSES, fill_value=0)
    frequency = frequency.merge(split_counts.add_suffix("_count"), left_on="class_name", right_index=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), layout="constrained")
    bars = axes[0, 0].barh(CLASSES, counts, color=[PALETTE[c] for c in CLASSES])
    axes[0, 0].bar_label(bars, labels=[f"{n:,} ({n/len(valid):.1%})" for n in counts], padding=3)
    axes[0, 0].set(xlabel="Bounding boxes válidas", title=f"A) Frequência das cinco classes — {len(valid):,} boxes")
    split_counts.div(split_counts.sum(axis=0), axis=1).mul(100).plot.bar(ax=axes[0, 1], rot=0)
    axes[0, 1].set(title="B) Distribuição por split", ylabel="% das boxes do split", xlabel="Classe")
    axes[1, 0].bar(["Militares (SMV, LMV, AFV, MCV)", "Civis (CV)"],
                   [counts.drop("CV").sum(), counts["CV"]], color=["#1f77b4", "#8c969e"])
    axes[1, 0].set(title="C) Veículos militares e civis", ylabel="Bounding boxes")
    axes[1, 1].barh(CLASSES, presence/len(annotated)*100, color=[PALETTE[c] for c in CLASSES])
    axes[1, 1].set(title="D) Presença nas imagens anotadas", xlabel="% de imagens", xlim=(0, 105))
    fig1 = save_figure(fig, figures, "01_class_frequencies.png", "MVRSD — Frequência e presença de classes")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), layout="constrained")
    supported = [c for c in CLASSES if (valid.class_name == c).any()]
    data = [valid.loc[valid.class_name == c, "rel_area_pct"].to_numpy() for c in supported]
    axes[0, 0].boxplot(data, showfliers=False)
    axes[0, 0].set_xticks(range(1, len(supported)+1), labels=supported)
    axes[0, 0].set(yscale="log", ylabel="% da área da imagem (log)", title="A) Área relativa por classe")
    scale_counts = pd.crosstab(valid.class_name, valid.coco_scale).reindex(index=CLASSES, columns=SCALES, fill_value=0)
    scale_counts.div(scale_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).mul(100).plot.bar(
        stacked=True, ax=axes[0, 1], rot=0, color=["#e45756", "#f58518", "#54a24b"])
    axes[0, 1].set(title="B) Escala absoluta COCO por área", ylabel="% da classe", xlabel="Classe", ylim=(0, 100))
    for label in supported:
        values = valid.loc[valid.class_name == label, "aspect_ratio"]
        axes[1, 0].hist(values, bins=np.linspace(0, 5, 40), density=True, histtype="step",
                        color=PALETTE[label], label=f"{label}: mediana {values.median():.2f}")
    axes[1, 0].set(title="C) Razão de aspecto — intervalo exibido [0, 5]", xlabel="Largura / altura", ylabel="Densidade")
    axes[1, 0].legend()
    hb = axes[1, 1].hexbin(valid.w_box, valid.h_box, gridsize=35, mincnt=1, bins="log", cmap="viridis")
    fig.colorbar(hb, ax=axes[1, 1], label="Contagem (cor log)")
    axes[1, 1].set(title="D) Dimensões das caixas", xlabel="Largura (px)", ylabel="Altura (px)")
    fig2 = save_figure(fig, figures, "02_bbox_sizes_and_scales.png", "MVRSD — Escala e geometria das bounding boxes")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), layout="constrained")
    images.resolution.value_counts().head(10).plot.barh(ax=axes[0, 0], color="#3b6fb6")
    axes[0, 0].set(title="A) Resoluções medidas — inclui imagens sem rótulo", xlabel="Imagens")
    axes[0, 1].hist(annotated.valid_objects, bins=35, color="#1f77b4", edgecolor="white")
    axes[0, 1].set(title="B) Densidade nas imagens anotadas", xlabel="Veículos / imagem", ylabel="Imagens")
    spatial = axes[1, 0].hist2d(valid.x_center_norm, valid.y_center_norm, bins=40, range=[[0, 1], [0, 1]], cmap="inferno")
    fig.colorbar(spatial[3], ax=axes[1, 0], label="Veículos")
    axes[1, 0].invert_yaxis()
    axes[1, 0].set(title="C) Distribuição espacial dos centros", xlabel="x normalizado", ylabel="y normalizado")
    groups = list(annotated.groupby("split", sort=True))
    axes[1, 1].boxplot([frame.valid_objects for _, frame in groups], showfliers=False)
    axes[1, 1].set_xticks(range(1, len(groups)+1), labels=[name for name, _ in groups])
    axes[1, 1].set(title="D) Densidade por split", ylabel="Veículos / imagem")
    fig3 = save_figure(fig, figures, "03_resolutions_and_scene_density.png", "MVRSD — Resolução, densidade e distribuição espacial")

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), layout="constrained")
    axes[0, 0].bar(["Anotadas", "Sem anotação"], [int(images.has_annotation.sum()), int((~images.has_annotation).sum())], color=["#1f77b4", "#8c969e"])
    axes[0, 0].set(title="A) Cobertura de anotações", ylabel="Imagens")
    for axis, field, title in [(axes[0, 1], "truncated", "B) Truncamento nativo"),
                                (axes[1, 0], "occluded", "C) Oclusão nativa")]:
        values = valid[field].dropna()
        axis.set_title(title)
        if values.empty:
            axis.text(.5, .5, "Campo não fornecido\nNão inferido a partir da imagem", ha="center", va="center", transform=axis.transAxes)
            axis.set_axis_off()
        else:
            values.astype(int).value_counts().sort_index().plot.bar(ax=axis, color="#3b6fb6", rot=0)
            axis.set(xlabel="Valor registrado na anotação", ylabel="Bounding boxes")
    boundary = valid.groupby("class_name").boundary_touching.mean().reindex(CLASSES).fillna(0)*100
    axes[1, 1].bar(CLASSES, boundary, color=[PALETTE[c] for c in CLASSES])
    axes[1, 1].set(title="D) Caixas junto à borda (≤1 px) — proxy geométrico", ylabel="% da classe")
    fig4 = save_figure(fig, figures, "04_annotation_quality_and_metadata.png", "MVRSD — Qualidade de anotações e disponibilidade de metadados")
    size = valid.groupby("class_name").agg(objects=("class_name", "size"), median_width_px=("w_box", "median"),
            median_height_px=("h_box", "median"), median_rel_area_pct=("rel_area_pct", "median"),
            median_aspect_ratio=("aspect_ratio", "median")).reindex(CLASSES)
    scale_pct = scale_counts.div(scale_counts.sum(axis=1).replace(0, np.nan), axis=0).mul(100)
    size[["small_pct", "medium_pct", "large_pct"]] = scale_pct.reindex(CLASSES).to_numpy()
    return {"class_frequencies": frequency, "box_stats": size,
            "coco_scale_pct": (valid.coco_scale.value_counts(normalize=True).reindex(SCALES, fill_value=0)*100).to_dict(),
            "figures": [fig1, fig2, fig3, fig4]}


def representative_scenes(images: pd.DataFrame, boxes: pd.DataFrame, count: int = 4) -> list[str]:
    annotated = images[images.has_annotation].copy()
    candidates = annotated[annotated.split == "val"]
    if candidates.empty:
        candidates = annotated
    valid = boxes[boxes.is_valid_yolo]
    candidates = candidates.set_index("image_id")
    candidates["diversity"] = valid.groupby("image_id").class_name.nunique().reindex(candidates.index, fill_value=0)
    candidates["median_area"] = valid.groupby("image_id").rel_area_pct.median().reindex(candidates.index)
    rankings = [candidates.sort_values(["diversity", "valid_objects"], ascending=False),
                candidates.sort_values("valid_objects", ascending=False),
                candidates.sort_values("median_area"),
                candidates.assign(distance=(candidates.valid_objects-candidates.valid_objects.median()).abs()).sort_values("distance")]
    selected = []
    for ranking in rankings:
        for identifier in ranking.index:
            if identifier not in selected:
                selected.append(identifier)
                break
        if len(selected) >= count:
            break
    return selected


def draw_boxes(axis, image, frame: pd.DataFrame) -> None:
    axis.imshow(image)
    for row in frame.itertuples():
        axis.add_patch(Rectangle((row.x_left, row.y_top), row.w_box, row.h_box,
                                 fill=False, edgecolor=PALETTE[row.class_name], linewidth=1.2))
    axis.set_axis_off()


def plot_scenes(images: pd.DataFrame, boxes: pd.DataFrame, identifiers: list[str], figures: Path) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(14, 13), layout="constrained")
    for axis in axes.flat:
        axis.set_axis_off()
    for axis, identifier in zip(axes.flat, identifiers):
        row = images.set_index("image_id").loc[identifier]
        frame = boxes[(boxes.image_id == identifier) & boxes.is_valid_yolo]
        with Image.open(row.image_path) as image:
            draw_boxes(axis, image.convert("RGB"), frame)
        axis.set_title(f"{Path(identifier).stem} — {row.split}\n{len(frame)} veículos; {frame.class_name.nunique()} classes")
    fig.legend(handles=[Patch(facecolor="none", edgecolor=PALETTE[c], label=c) for c in CLASSES],
               loc="outside lower center", ncol=len(CLASSES))
    return save_figure(fig, figures, "05_representative_scenes.png", "MVRSD — Cenas representativas com rótulos originais")


def plot_detectors(root: Path, images: pd.DataFrame, boxes: pd.DataFrame, identifiers: list[str],
                   figures: Path, device: str) -> dict:
    from .configuration import configure_environment
    configure_environment(root)
    from .configuration import load_settings
    from .pipeline import make_classifier

    configurations = ("configs/default.yaml", "configs/yolo26s_mvrsd_vis.yaml")
    classifiers = [(name, make_classifier(load_settings(root, Path(name)))) for name in configurations]
    for _, classifier in classifiers:
        from .similarity_efficientnet import resolve_device
        classifier.device = resolve_device(device)
    selected = identifiers[:3]
    fig, axes = plt.subplots(len(selected), 3, figsize=(17, 5*len(selected)), squeeze=False, layout="constrained")
    records, identities = [], {}
    for row_index, identifier in enumerate(selected):
        path = Path(images.set_index("image_id").loc[identifier, "image_path"])
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        actual = boxes[(boxes.image_id == identifier) & boxes.is_valid_yolo]
        draw_boxes(axes[row_index, 0], image, actual)
        axes[row_index, 0].set_title(f"Ground truth — {len(actual)} veículos\n{path.stem}")
        for column, (config, classifier) in enumerate(classifiers, 1):
            detections = classifier.detect_vehicle_crops(image)
            axis = axes[row_index, column]
            axis.imshow(image)
            for detection in detections:
                x1, y1, x2, y2 = detection["bbox"]
                axis.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="#1f77b4", linewidth=1.3))
            axis.set_axis_off()
            name = Path(classifier.yolo_model_path).stem
            axis.set_title(f"{name}\n{len(detections)} detecções — conf ≥ {classifier.detection_settings['confidence_threshold']}", fontsize=9)
            records.append({"image_id": identifier, "configuration": config, "gt_objects": len(actual),
                            "detections": len(detections), "mean_confidence": float(np.mean([d["detection_confidence"] for d in detections])) if detections else None})
            identities[config] = {**classifier.detector_metadata(), "device": classifier.device,
                                  "parameters": classifier.detection_settings, "classes": classifier.detector.names}
    name = save_figure(fig, figures, "06_qualitative_detectors.png", "MVRSD — Inspeção qualitativa; possíveis dados de treino, sem estimativa de generalização")
    return {"status": "completed", "figure": name, "checkpoints": identities, "scenes": records}


def plot_gallery(root: Path, figures: Path) -> dict:
    labels = yaml.safe_load((root / "configs/classes.yaml").read_text())["labels"]
    rows, examples = [], {}
    for label in labels:
        for path in sorted((root / "data/reference_dataset" / label).rglob("*")):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            with Image.open(path) as image:
                width, height = image.size
            augmented = "_aug_" in path.stem
            rows.append({"label": label, "width": width, "height": height, "augmented": augmented})
            if label not in examples or ("_aug_" in examples[label].stem and not augmented):
                examples[label] = path
    frame = pd.DataFrame(rows)
    fig = plt.figure(figsize=(16, 8), layout="constrained")
    grid = fig.add_gridspec(2, len(labels))
    axis = fig.add_subplot(grid[0, :3])
    pd.crosstab(frame.label, frame.augmented).reindex(labels).rename(columns={False: "Base", True: "Aumentada"}).plot.bar(
        stacked=True, ax=axis, color=["#2b5c8f", "#6baed6"], rot=15)
    axis.set(title=f"A) Galeria simulada — {len(frame)} referências", ylabel="Crops", xlabel="Modelo operacional")
    axis = fig.add_subplot(grid[0, 3:])
    for label in labels:
        subset = frame[frame.label == label]
        axis.scatter(subset.width, subset.height, label=label, alpha=.7)
    axis.set(title="B) Dimensões dos recortes", xlabel="Largura (px)", ylabel="Altura (px)")
    axis.legend(fontsize=8)
    for i, label in enumerate(labels):
        axis = fig.add_subplot(grid[1, i])
        with Image.open(examples[label]) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(label, fontsize=9)
        axis.set_axis_off()
    name = save_figure(fig, figures, "07_reference_gallery_overview.png", "Cinco categorias MVRSD ≠ cinco modelos operacionais da galeria")
    return {"figure": name, "total_reference_crops": len(frame),
            "classes": {label: {"total": int((frame.label == label).sum()),
                                "augmented": int(((frame.label == label) & frame.augmented).sum())} for label in labels}}


def run_mvrsd_eda(root: Path, out_dir: Path | None = None, dataset_root: Path | None = None,
                  splits: tuple[str, ...] | None = ("train", "val"), class_names: dict[int, str] | None = None,
                  include_detector: bool = True, device: str = "auto") -> dict:
    root = root.resolve()
    out_dir = out_dir or root / "outputs/eda_mvrsd"
    figures = out_dir / "figures"
    images, boxes, audit = load_mvrsd_dataframe(root, dataset_root, splits, class_names)
    figures.mkdir(parents=True, exist_ok=True)
    images.to_csv(out_dir / "mvrsd_images_summary.csv", index=False)
    boxes.to_csv(out_dir / "mvrsd_boxes.csv", index=False)
    with plt.rc_context(STYLE):
        statistics = plot_statistics(images, boxes, figures)
        selected = representative_scenes(images, boxes)
        scenes = plot_scenes(images, boxes, selected, figures)
        qualitative = plot_detectors(root, images, boxes, selected, figures, device) if include_detector else {"status": "disabled"}
        gallery = plot_gallery(root, figures)
    statistics["class_frequencies"].to_csv(out_dir / "mvrsd_class_frequencies.csv", index=False)
    statistics["box_stats"].to_csv(out_dir / "mvrsd_box_sizes.csv", index_label="class_name")
    valid = boxes[boxes.is_valid_yolo]
    annotated = images[images.has_annotation]
    report = {"dataset": "MVRSD", "sources": SOURCES, "published_reference": PUBLISHED,
              "total_images": len(images), "annotated_images": len(annotated), "valid_boxes": len(valid),
              "splits_image_count": {str(k): int(v) for k, v in images.split.value_counts().items()},
              "annotation_audit": {key: value for key, value in audit.items() if key != "inventory"},
              "class_frequencies": statistics["class_frequencies"].to_dict(orient="records"),
              "bbox_sizes_and_scales": {"coco_scale_pct": statistics["coco_scale_pct"],
                                        "per_class": json.loads(statistics["box_stats"].to_json(orient="index"))},
              "resolutions_and_density": {"resolutions": {str(k): int(v) for k,v in images.resolution.value_counts().items()},
                                          "objects_per_image": {"mean": float(annotated.valid_objects.mean()),
                                                                "median": float(annotated.valid_objects.median()),
                                                                "p90": float(annotated.valid_objects.quantile(.9)),
                                                                "max": int(annotated.valid_objects.max())}},
              "metadata_availability": {key: {"boxes_with_field": int(valid[key].notna().sum()),
                                                "values": {str(k): int(v) for k,v in valid[key].dropna().value_counts().items()}}
                                        for key in ("truncated", "difficult", "occluded")},
              "representative_scenes": {"figure": scenes, "image_ids": selected},
              "qualitative_detector": qualitative, "reference_gallery_overview": gallery,
              "figures": statistics["figures"] + [scenes, gallery["figure"]] +
                         ([qualitative["figure"]] if qualitative.get("figure") else [])}
    (out_dir / "dataset_inventory.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    (out_dir / "eda_mvrsd_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--data", type=Path, help="Extracted MVRSD directory; default: data/mvrsd")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test", "unspecified"], default=["train", "val"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--skip-detector", action="store_true")
    args = parser.parse_args()
    report = run_mvrsd_eda(args.root, args.output, args.data, tuple(args.splits) if args.splits else None,
                           include_detector=not args.skip_detector, device=args.device)
    print(json.dumps({key: report[key] for key in ("dataset", "total_images", "annotated_images", "valid_boxes")}, indent=2))


if __name__ == "__main__":
    main()
