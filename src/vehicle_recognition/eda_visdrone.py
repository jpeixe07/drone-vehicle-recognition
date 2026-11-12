"""Reproducible Exploratory Data Analysis (EDA) for VisDrone2019-DET and Reference Gallery.

Reads prepared VisDrone splits (train and val) from data/visdrone/, computes
comprehensive statistics on class frequencies (highlighting vehicle classes:
car, van, truck, bus), bounding-box sizes/aspect ratios/COCO scales, image
resolutions, scene object density, truncation/occlusion metadata, representative
aerial scenes, and qualitative YOLOv8s detector results. Writes figures and
summary tables to outputs/eda_visdrone/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from PIL import Image

VISDRONE_SPLIT_DIRS: dict[str, str] = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test": "VisDrone2019-DET-test-dev",
    "test-dev": "VisDrone2019-DET-test-dev",
}

VISDRONE_RAW_CATEGORIES: dict[int, str] = {
    0: "ignored regions",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}

VISDRONE_YOLO_CLASSES: dict[int, str] = {
    i - 1: name for i, name in VISDRONE_RAW_CATEGORIES.items() if 1 <= i <= 10
}

VEHICLE_CLASS_NAMES: tuple[str, ...] = ("car", "van", "truck", "bus")

# Consistent visual styling
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

VEHICLE_COLOR = "#1f77b4"
NON_VEHICLE_COLOR = "#8c969e"
CLASS_PALETTE: dict[str, str] = {
    "car": "#1f77b4",
    "van": "#2ca02c",
    "truck": "#ff7f0e",
    "bus": "#d62728",
    "pedestrian": "#9467bd",
    "people": "#8c564b",
    "motor": "#e377c2",
    "bicycle": "#7f7f7f",
    "tricycle": "#bcbd22",
    "awning-tricycle": "#17becf",
}

TRUNCATION_LABELS = {0: "Sem truncamento (0%)", 1: "Truncado parcial (1–50%)"}
OCCLUSION_LABELS = {
    0: "Sem oclusão (0%)",
    1: "Oclusão parcial (1–50%)",
    2: "Oclusão severa (>50%)",
}


def load_visdrone_dataframe(
    root: Path, splits: Iterable[str] = ("train", "val")
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse raw VisDrone annotations and image headers into image-level and box-level DataFrames."""
    visdrone_root = root / "data/visdrone"
    image_rows: list[dict] = []
    box_rows: list[dict] = []

    for split in splits:
        split_dir = visdrone_root / VISDRONE_SPLIT_DIRS[split]
        if (split_dir / "images").is_dir() and (split_dir / "annotations").is_dir():
            images_dir = split_dir / "images"
            annotations_dir = split_dir / "annotations"
        elif (visdrone_root / "images" / split).is_dir() and (visdrone_root / "annotations" / split).is_dir():
            images_dir = visdrone_root / "images" / split
            annotations_dir = visdrone_root / "annotations" / split
        else:
            raise FileNotFoundError(f"Missing VisDrone split directory for '{split}' in {visdrone_root}")

        ann_files = sorted(annotations_dir.glob("*.txt"))
        for ann_file in ann_files:
            stem = ann_file.stem
            img_path = images_dir / f"{stem}.jpg"
            if not img_path.is_file():
                continue
            with Image.open(img_path) as img:
                img_w, img_h = img.size
            img_area = float(img_w * img_h)

            raw_lines = ann_file.read_text(encoding="utf-8").splitlines()
            valid_obj_count = 0
            vehicle_obj_count = 0
            ignored_count = 0

            for line in raw_lines:
                line = line.strip().rstrip(",")
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 8:
                    continue
                x_left = float(parts[0])
                y_top = float(parts[1])
                w_box = float(parts[2])
                h_box = float(parts[3])
                score = int(float(parts[4]))
                category = int(float(parts[5]))
                truncation = int(float(parts[6]))
                occlusion = int(float(parts[7]))

                raw_cat_name = VISDRONE_RAW_CATEGORIES.get(category, "unknown")
                is_valid_yolo = score != 0 and 1 <= category <= 10 and w_box > 0 and h_box > 0
                if is_valid_yolo:
                    valid_obj_count += 1
                    if raw_cat_name in VEHICLE_CLASS_NAMES:
                        vehicle_obj_count += 1
                else:
                    ignored_count += 1

                box_area = max(0.0, w_box * h_box)
                rel_area = box_area / img_area if img_area > 0 else 0.0
                aspect_ratio = (w_box / h_box) if h_box > 0 else np.nan
                x_center_norm = (x_left + w_box / 2.0) / img_w if img_w > 0 else np.nan
                y_center_norm = (y_top + h_box / 2.0) / img_h if img_h > 0 else np.nan

                if box_area < 32.0 * 32.0:
                    coco_scale = "Small (<32² px)"
                elif box_area < 96.0 * 96.0:
                    coco_scale = "Medium (32²–96² px)"
                else:
                    coco_scale = "Large (≥96² px)"

                box_rows.append({
                    "split": split,
                    "image_id": stem,
                    "img_w": img_w,
                    "img_h": img_h,
                    "x_left": x_left,
                    "y_top": y_top,
                    "w_box": w_box,
                    "h_box": h_box,
                    "box_area_px": box_area,
                    "rel_area": rel_area,
                    "rel_area_pct": rel_area * 100.0,
                    "aspect_ratio": aspect_ratio,
                    "x_center_norm": x_center_norm,
                    "y_center_norm": y_center_norm,
                    "score_flag": score,
                    "raw_category_id": category,
                    "class_name": raw_cat_name,
                    "yolo_class_id": (category - 1) if is_valid_yolo else -1,
                    "is_valid_yolo": is_valid_yolo,
                    "is_vehicle": raw_cat_name in VEHICLE_CLASS_NAMES and is_valid_yolo,
                    "group": (
                        "Veículo-Alvo (car/van/truck/bus)"
                        if (raw_cat_name in VEHICLE_CLASS_NAMES and is_valid_yolo)
                        else ("Outras Classes YOLO" if is_valid_yolo else "Ignorado / Outros")
                    ),
                    "truncation": truncation,
                    "occlusion": occlusion,
                    "coco_scale": coco_scale,
                })

            image_rows.append({
                "split": split,
                "image_id": stem,
                "image_path": str(img_path.relative_to(root).as_posix()),
                "img_w": img_w,
                "img_h": img_h,
                "resolution": f"{img_w}x{img_h}",
                "aspect_ratio": round(img_w / img_h, 3),
                "megapixels": round((img_w * img_h) / 1e6, 3),
                "valid_objects": valid_obj_count,
                "vehicle_objects": vehicle_obj_count,
                "non_vehicle_objects": valid_obj_count - vehicle_obj_count,
                "ignored_objects": ignored_count,
                "has_vehicles": vehicle_obj_count > 0,
            })

    images_df = pd.DataFrame(image_rows)
    boxes_df = pd.DataFrame(box_rows)
    return images_df, boxes_df


def plot_class_frequencies(
    images_df: pd.DataFrame, boxes_df: pd.DataFrame, out_dir: Path
) -> dict:
    """Figure 1: Comprehensive class frequency distribution highlighting vehicle classes."""
    valid_df = boxes_df[boxes_df["is_valid_yolo"]].copy()
    total_valid = len(valid_df)

    # Order classes by total frequency descending
    class_counts = (
        valid_df.groupby("class_name")
        .size()
        .rename("total")
        .sort_values(ascending=False)
        .reset_index()
    )
    class_counts["pct_total"] = class_counts["total"] / total_valid * 100.0
    class_counts["is_vehicle"] = class_counts["class_name"].isin(VEHICLE_CLASS_NAMES)
    ordered_classes = class_counts["class_name"].tolist()

    # Split breakdown
    split_counts = (
        valid_df.groupby(["class_name", "split"])
        .size()
        .unstack(fill_value=0)
        .reindex(ordered_classes)
    )
    split_pct = split_counts.div(split_counts.sum(axis=0), axis=1) * 100.0

    # Image presence rate (% of images where class appears at least once)
    total_images = len(images_df)
    img_presence = (
        valid_df.groupby("class_name")["image_id"]
        .nunique()
        .reindex(ordered_classes)
        .fillna(0)
        / total_images
        * 100.0
    )

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    # Panel A: Total count bar chart highlighting vehicle classes
    ax1 = fig.add_subplot(gs[0, 0])
    colors = [
        VEHICLE_COLOR if c in VEHICLE_CLASS_NAMES else NON_VEHICLE_COLOR
        for c in ordered_classes
    ]
    bars = ax1.barh(
        ordered_classes[::-1],
        class_counts["total"].values[::-1],
        color=colors[::-1],
        edgecolor="black",
        linewidth=0.6,
    )
    ax1.set_title("A) Frequência Total por Classe (Train + Val, n = 381.963 boxes)")
    ax1.set_xlabel("Número de Anotações (Bounding Boxes)")
    ax1.set_xlim(0, class_counts["total"].max() * 1.24)
    for bar, pct in zip(bars, class_counts["pct_total"].values[::-1]):
        w = bar.get_width()
        ax1.text(
            w + class_counts["total"].max() * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{int(w):,} ({pct:.1f}%)",
            va="center",
            fontsize=8.5,
            fontweight="bold" if bar.get_facecolor()[:3] == matplotlib.colors.to_rgb(VEHICLE_COLOR) else "normal",
        )
    from matplotlib.lines import Line2D
    legend_elements = [
        patches.Patch(facecolor=VEHICLE_COLOR, edgecolor="black", label="Veículos-Alvo (car, van, truck, bus)"),
        patches.Patch(facecolor=NON_VEHICLE_COLOR, edgecolor="black", label="Outras Classes VisDrone"),
    ]
    ax1.legend(handles=legend_elements, loc="lower right", frameon=True)

    # Panel B: Train vs Val relative distribution (%)
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(ordered_classes))
    width = 0.38
    train_vals = split_pct["train"].values if "train" in split_pct else np.zeros(len(ordered_classes))
    val_vals = split_pct["val"].values if "val" in split_pct else np.zeros(len(ordered_classes))
    ax2.bar(x - width / 2, train_vals, width, label=f"Train ({split_counts['train'].sum():,} boxes)", color="#2b5c8f")
    ax2.bar(x + width / 2, val_vals, width, label=f"Val ({split_counts['val'].sum():,} boxes)", color="#e68a00")
    ax2.set_xticks(x)
    ax2.set_xticklabels(ordered_classes, rotation=30, ha="right")
    ax2.set_ylabel("% das Anotações do Split")
    ax2.set_title("B) Consistência de Distribuição: Split Train vs. Val (%)")
    ax2.legend(frameon=True)

    # Panel C: Vehicle classes breakdown vs Non-vehicles + Internal vehicle distribution
    ax3 = fig.add_subplot(gs[1, 0])
    veh_df = valid_df[valid_df["is_vehicle"]]
    veh_counts = (
        veh_df["class_name"]
        .value_counts()
        .reindex(["car", "van", "truck", "bus"])
        .fillna(0)
    )
    veh_pct_internal = veh_counts / veh_counts.sum() * 100.0
    veh_pct_global = veh_counts / total_valid * 100.0
    v_bars = ax3.bar(
        veh_counts.index,
        veh_counts.values,
        color=[CLASS_PALETTE[c] for c in veh_counts.index],
        edgecolor="black",
        linewidth=0.8,
        width=0.55,
    )
    ax3.set_title(
        f"C) Detalhamento dos Veículos-Alvo ({veh_counts.sum():,} boxes = {veh_counts.sum()/total_valid*100:.1f}% do VisDrone)"
    )
    ax3.set_ylabel("Número de Bounding Boxes")
    ax3.set_ylim(0, veh_counts.max() * 1.22)
    for bar, p_int, p_glob in zip(v_bars, veh_pct_internal.values, veh_pct_global.values):
        h = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            h + veh_counts.max() * 0.02,
            f"{int(h):,}\n({p_int:.1f}% dos veículos |\n{p_glob:.1f}% do total)",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    # Panel D: Image Presence Rate (% of images containing at least 1 instance of the class)
    ax4 = fig.add_subplot(gs[1, 1])
    pres_bars = ax4.barh(
        ordered_classes[::-1],
        img_presence.values[::-1],
        color=colors[::-1],
        edgecolor="black",
        linewidth=0.6,
    )
    ax4.set_xlim(0, 108)
    ax4.set_xlabel("% de Imagens com Pelo Menos 1 Instância da Classe")
    ax4.set_title(f"D) Taxa de Presença por Imagem (Total = {total_images:,} imagens)")
    for bar in pres_bars:
        w = bar.get_width()
        ax4.text(
            w + 1.2,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}%",
            va="center",
            fontsize=8.5,
        )

    fig.suptitle(
        "VisDrone2019-DET — Análise de Frequência de Classes e Dominância de Veículos",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    out_path = out_dir / "01_class_frequencies.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    # Build summary dict
    class_table = []
    for cls_name in ordered_classes:
        tr_c = int(split_counts.loc[cls_name, "train"]) if "train" in split_counts else 0
        va_c = int(split_counts.loc[cls_name, "val"]) if "val" in split_counts else 0
        tot_c = tr_c + va_c
        class_table.append({
            "class_name": cls_name,
            "is_vehicle": cls_name in VEHICLE_CLASS_NAMES,
            "train_count": tr_c,
            "val_count": va_c,
            "total_count": tot_c,
            "pct_of_total": round(tot_c / total_valid * 100.0, 2),
            "pct_of_vehicles": round(tot_c / int(veh_counts.sum()) * 100.0, 2) if cls_name in VEHICLE_CLASS_NAMES else None,
            "image_presence_pct": round(float(img_presence.loc[cls_name]), 2),
        })
    return {
        "figure": str(out_path.name),
        "total_valid_boxes": int(total_valid),
        "total_vehicle_boxes": int(veh_counts.sum()),
        "vehicle_share_pct": round(float(veh_counts.sum() / total_valid * 100.0), 2),
        "classes": class_table,
    }


def plot_bbox_sizes_and_scales(boxes_df: pd.DataFrame, out_dir: Path) -> dict:
    """Figure 2: Bounding-box relative area, COCO scale breakdown, and aspect ratios."""
    valid_df = boxes_df[boxes_df["is_valid_yolo"] & (boxes_df["rel_area"] > 0)].copy()
    valid_df["log10_rel_area_pct"] = np.log10(valid_df["rel_area_pct"].clip(lower=1e-4))

    ordered_classes = [
        "car", "van", "truck", "bus",
        "pedestrian", "people", "motor", "bicycle", "tricycle", "awning-tricycle",
    ]

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.25)

    # Panel A: Boxplot of relative bounding box area (% of image area, log scale)
    ax1 = fig.add_subplot(gs[0, 0])
    palette = [VEHICLE_COLOR if c in VEHICLE_CLASS_NAMES else NON_VEHICLE_COLOR for c in ordered_classes]
    sns.boxplot(
        data=valid_df,
        x="class_name",
        y="rel_area_pct",
        order=ordered_classes,
        palette=palette,
        hue="class_name",
        legend=False,
        showfliers=False,
        ax=ax1,
    )
    ax1.set_yscale("log")
    ax1.set_xticks(range(len(ordered_classes)))
    ax1.set_xticklabels(ordered_classes, rotation=30, ha="right")
    ax1.set_ylabel("Área Relativa da Box (% da Imagem, escala log)")
    ax1.set_xlabel("")
    ax1.set_title("A) Área da Bounding Box Relativa à Imagem (%) por Classe")

    # Panel B: COCO Scale distribution (Small <32², Medium 32²-96², Large >=96²) per class
    ax2 = fig.add_subplot(gs[0, 1])
    scale_order = ["Small (<32² px)", "Medium (32²–96² px)", "Large (≥96² px)"]
    scale_ct = (
        valid_df.groupby(["class_name", "coco_scale"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=ordered_classes, columns=scale_order, fill_value=0)
    )
    scale_pct = scale_ct.div(scale_ct.sum(axis=1), axis=0) * 100.0
    bottom = np.zeros(len(ordered_classes))
    scale_colors = ["#e45756", "#f58518", "#54a24b"]
    for scale_label, scolor in zip(scale_order, scale_colors):
        vals = scale_pct[scale_label].values
        ax2.bar(ordered_classes, vals, bottom=bottom, label=scale_label, color=scolor, edgecolor="white", linewidth=0.5)
        bottom += vals
    ax2.set_xticks(range(len(ordered_classes)))
    ax2.set_xticklabels(ordered_classes, rotation=30, ha="right")
    ax2.set_ylabel("% das Boxes da Classe")
    ax2.set_ylim(0, 100)
    ax2.set_title("B) Proporção por Escala Absoluta (Critério COCO: Small / Medium / Large)")
    ax2.legend(loc="upper right", frameon=True)

    # Panel C: Aspect Ratio (width / height) distribution for the 4 target vehicle classes
    ax3 = fig.add_subplot(gs[1, 0])
    veh_df = valid_df[valid_df["is_vehicle"]].copy()
    veh_df["aspect_ratio_clipped"] = veh_df["aspect_ratio"].clip(0.2, 4.0)
    for v_cls in ["car", "van", "truck", "bus"]:
        sub = veh_df[veh_df["class_name"] == v_cls]["aspect_ratio_clipped"].dropna()
        sns.kdeplot(
            sub,
            ax=ax3,
            label=f"{v_cls} (mediana={sub.median():.2f})",
            color=CLASS_PALETTE[v_cls],
            linewidth=2.0,
            fill=True,
            alpha=0.12,
        )
    ax3.axvline(1.0, color="black", linestyle="--", linewidth=1.0, label="Quadrado (w/h = 1.0)")
    ax3.set_xlim(0.2, 3.5)
    ax3.set_xlabel("Razão de Aspecto da Bounding Box (Largura / Altura)")
    ax3.set_ylabel("Densidade")
    ax3.set_title("C) Distribuição da Razão de Aspecto (w/h) nos 4 Veículos-Alvo")
    ax3.legend(frameon=True)

    # Panel D: Width vs Height scatter/2D histogram (in pixels) for vehicle bounding boxes
    ax4 = fig.add_subplot(gs[1, 1])
    w_px = veh_df["w_box"].clip(upper=300)
    h_px = veh_df["h_box"].clip(upper=300)
    hb = ax4.hexbin(w_px, h_px, gridsize=45, cmap="viridis", bins="log", mincnt=1)
    plt.colorbar(hb, ax=ax4, label="log10(Contagem de Veículos)")
    ax4.axvline(32, color="#ff4d4d", linestyle="--", linewidth=1.1, label="Limite Small (32 px)")
    ax4.axhline(32, color="#ff4d4d", linestyle="--", linewidth=1.1)
    ax4.axvline(96, color="#ffa600", linestyle=":", linewidth=1.1, label="Limite Medium (96 px)")
    ax4.axhline(96, color="#ffa600", linestyle=":", linewidth=1.1)
    ax4.set_xlabel("Largura da Box em Pixels (truncado em 300 px)")
    ax4.set_ylabel("Altura da Box em Pixels (truncado em 300 px)")
    ax4.set_title("D) Dimensões em Pixels (w × h) das Boxes de Veículos")
    ax4.legend(loc="upper right", frameon=True)

    fig.suptitle(
        "VisDrone2019-DET — Escala dos Objetos, Área Relativa e Geometria das Bounding Boxes",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    out_path = out_dir / "02_bbox_sizes_and_scales.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    # Stats summary per class
    size_stats = {}
    for cls_name in ordered_classes:
        sub = valid_df[valid_df["class_name"] == cls_name]
        size_stats[cls_name] = {
            "median_rel_area_pct": round(float(sub["rel_area_pct"].median()), 4),
            "mean_rel_area_pct": round(float(sub["rel_area_pct"].mean()), 4),
            "p90_rel_area_pct": round(float(sub["rel_area_pct"].quantile(0.90)), 4),
            "median_width_px": round(float(sub["w_box"].median()), 1),
            "median_height_px": round(float(sub["h_box"].median()), 1),
            "median_aspect_ratio": round(float(sub["aspect_ratio"].median()), 2),
            "small_pct": round(float(scale_pct.loc[cls_name, "Small (<32² px)"]), 2),
            "medium_pct": round(float(scale_pct.loc[cls_name, "Medium (32²–96² px)"]), 2),
            "large_pct": round(float(scale_pct.loc[cls_name, "Large (≥96² px)"]), 2),
        }

    overall_scales = valid_df["coco_scale"].value_counts(normalize=True) * 100.0
    veh_scales = veh_df["coco_scale"].value_counts(normalize=True) * 100.0
    return {
        "figure": str(out_path.name),
        "overall_coco_scale_pct": {k: round(float(v), 2) for k, v in overall_scales.items()},
        "vehicle_coco_scale_pct": {k: round(float(v), 2) for k, v in veh_scales.items()},
        "per_class_box_stats": size_stats,
    }


def plot_resolutions_and_scene_density(
    images_df: pd.DataFrame, boxes_df: pd.DataFrame, out_dir: Path
) -> dict:
    """Figure 3: Image resolutions, object density per image, and spatial center heatmap."""
    valid_df = boxes_df[boxes_df["is_valid_yolo"]].copy()
    veh_df = valid_df[valid_df["is_vehicle"]].copy()

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.25)

    # Panel A: Top image resolutions
    ax1 = fig.add_subplot(gs[0, 0])
    res_counts = images_df["resolution"].value_counts().head(10)
    res_pct = res_counts / len(images_df) * 100.0
    bars = ax1.barh(
        res_counts.index[::-1],
        res_counts.values[::-1],
        color="#3b6fb6",
        edgecolor="black",
        linewidth=0.6,
    )
    ax1.set_xlim(0, res_counts.max() * 1.22)
    ax1.set_xlabel("Número de Imagens")
    ax1.set_title(f"A) Top 10 Resoluções de Imagem ({images_df['resolution'].nunique()} resoluções únicas)")
    for bar, pct in zip(bars, res_pct.values[::-1]):
        w = bar.get_width()
        ax1.text(
            w + res_counts.max() * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{int(w):,} ({pct:.1f}%)",
            va="center",
            fontsize=8.5,
        )

    # Panel B: Objects per image vs Vehicles per image distribution
    ax2 = fig.add_subplot(gs[0, 1])
    bins = np.linspace(0, 250, 50)
    ax2.hist(
        images_df["valid_objects"].clip(upper=250),
        bins=bins,
        alpha=0.55,
        color="#7f7f7f",
        edgecolor="black",
        label=f"Total Objetos / Img (média={images_df['valid_objects'].mean():.1f}, med={images_df['valid_objects'].median():.0f})",
    )
    ax2.hist(
        images_df["vehicle_objects"].clip(upper=250),
        bins=bins,
        alpha=0.75,
        color=VEHICLE_COLOR,
        edgecolor="black",
        label=f"Veículos / Img (média={images_df['vehicle_objects'].mean():.1f}, med={images_df['vehicle_objects'].median():.0f})",
    )
    ax2.set_xlabel("Quantidade de Objetos por Cena (truncado em 250)")
    ax2.set_ylabel("Número de Imagens")
    ax2.set_title("B) Densidade de Objetos por Cena Aérea (Crowding)")
    ax2.legend(frameon=True)

    # Panel C: Spatial heatmap of vehicle centers (x_center_norm, y_center_norm)
    ax3 = fig.add_subplot(gs[1, 0])
    h2d = ax3.hist2d(
        veh_df["x_center_norm"].clip(0, 1),
        veh_df["y_center_norm"].clip(0, 1),
        bins=[50, 40],
        cmap="inferno",
    )
    plt.colorbar(h2d[3], ax=ax3, label="Contagem de Veículos")
    ax3.invert_yaxis()
    ax3.set_xlabel(" Posição Horizontal Normalizada (x_center)")
    ax3.set_ylabel("Posição Vertical Normalizada (y_center)")
    ax3.set_title("C) Mapa de Calor Espacial dos Centros de Veículos (Perspectiva Oblíqua/Nadir)")

    # Panel D: Vehicle count per image across splits (Train vs Val boxplot/violin)
    ax4 = fig.add_subplot(gs[1, 1])
    sns.violinplot(
        data=images_df,
        x="split",
        y="vehicle_objects",
        hue="split",
        palette={"train": "#2b5c8f", "val": "#e68a00"},
        legend=False,
        cut=0,
        inner="quartile",
        ax=ax4,
    )
    ax4.set_xlabel("Split do Dataset")
    ax4.set_ylabel("Veículos por Imagem (car + van + truck + bus)")
    ax4.set_title("D) Distribuição de Veículos por Imagem: Train vs. Val")

    fig.suptitle(
        "VisDrone2019-DET — Resoluções, Densidade de Cenas e Distribuição Espacial",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    out_path = out_dir / "03_resolutions_and_scene_density.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "figure": str(out_path.name),
        "unique_resolutions": int(images_df["resolution"].nunique()),
        "top_resolutions": {k: int(v) for k, v in res_counts.head(6).items()},
        "objects_per_image": {
            "mean": round(float(images_df["valid_objects"].mean()), 2),
            "median": round(float(images_df["valid_objects"].median()), 1),
            "p90": round(float(images_df["valid_objects"].quantile(0.90)), 1),
            "max": int(images_df["valid_objects"].max()),
        },
        "vehicles_per_image": {
            "mean": round(float(images_df["vehicle_objects"].mean()), 2),
            "median": round(float(images_df["vehicle_objects"].median()), 1),
            "p90": round(float(images_df["vehicle_objects"].quantile(0.90)), 1),
            "max": int(images_df["vehicle_objects"].max()),
            "images_with_vehicles_pct": round(float(images_df["has_vehicles"].mean() * 100.0), 2),
        },
    }


def plot_occlusion_and_truncation(boxes_df: pd.DataFrame, out_dir: Path) -> dict:
    """Figure 4: Native VisDrone occlusion, truncation, and ignored-region metadata."""
    valid_df = boxes_df[boxes_df["is_valid_yolo"]].copy()
    ordered_classes = [
        "car", "van", "truck", "bus",
        "pedestrian", "people", "motor", "bicycle", "tricycle", "awning-tricycle",
    ]

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.25)

    # Panel A: Occlusion level per class
    ax1 = fig.add_subplot(gs[0, 0])
    occ_ct = (
        valid_df.groupby(["class_name", "occlusion"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=ordered_classes, columns=[0, 1, 2], fill_value=0)
    )
    occ_pct = occ_ct.div(occ_ct.sum(axis=1), axis=0) * 100.0
    occ_colors = ["#4daf4a", "#ffae19", "#e41a1c"]
    bottom = np.zeros(len(ordered_classes))
    for occ_code, ocolor in zip([0, 1, 2], occ_colors):
        vals = occ_pct[occ_code].values
        ax1.bar(
            ordered_classes,
            vals,
            bottom=bottom,
            label=OCCLUSION_LABELS[occ_code],
            color=ocolor,
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += vals
    ax1.set_xticks(range(len(ordered_classes)))
    ax1.set_xticklabels(ordered_classes, rotation=30, ha="right")
    ax1.set_ylabel("% das Boxes da Classe")
    ax1.set_ylim(0, 100)
    ax1.set_title("A) Grau de Oclusão por Classe (Metadado Nativo VisDrone)")
    ax1.legend(loc="upper right", frameon=True)

    # Panel B: Truncation rate (% truncated > 0) per class
    ax2 = fig.add_subplot(gs[0, 1])
    trunc_pct = (
        valid_df.groupby("class_name")["truncation"]
        .apply(lambda s: (s > 0).mean() * 100.0)
        .reindex(ordered_classes)
    )
    colors = [VEHICLE_COLOR if c in VEHICLE_CLASS_NAMES else NON_VEHICLE_COLOR for c in ordered_classes]
    t_bars = ax2.bar(ordered_classes, trunc_pct.values, color=colors, edgecolor="black", linewidth=0.6)
    ax2.set_xticks(range(len(ordered_classes)))
    ax2.set_xticklabels(ordered_classes, rotation=30, ha="right")
    ax2.set_ylabel("% de Objetos Truncados (Bordas/Corte)")
    ax2.set_ylim(0, max(trunc_pct.max() * 1.25, 15))
    ax2.set_title("B) Taxa de Truncamento (% Parcialmente Fora do Quadro) por Classe")
    for bar in t_bars:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.3,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    # Panel C: Impact of occlusion on bounding box size for vehicles
    ax3 = fig.add_subplot(gs[1, 0])
    veh_df = valid_df[valid_df["is_vehicle"] & valid_df["occlusion"].isin([0, 1, 2])].copy()
    veh_df["occ_label"] = veh_df["occlusion"].map(OCCLUSION_LABELS)
    sns.boxplot(
        data=veh_df,
        x="class_name",
        y="rel_area_pct",
        hue="occ_label",
        order=["car", "van", "truck", "bus"],
        hue_order=[OCCLUSION_LABELS[0], OCCLUSION_LABELS[1], OCCLUSION_LABELS[2]],
        palette=occ_colors,
        showfliers=False,
        ax=ax3,
    )
    ax3.set_yscale("log")
    ax3.set_xlabel("Classe de Veículo")
    ax3.set_ylabel("Área Relativa (% da Imagem, log)")
    ax3.set_title("C) Relação entre Tamanho do Veículo e Nível de Oclusão")
    ax3.legend(title="Nível de Oclusão", frameon=True)

    # Panel D: Raw annotations composition (Valid Vehicles vs Other YOLO vs Ignored Regions/Others)
    ax4 = fig.add_subplot(gs[1, 1])
    group_counts = boxes_df["group"].value_counts()
    g_labels = group_counts.index.tolist()
    g_vals = group_counts.values
    g_colors = [VEHICLE_COLOR, NON_VEHICLE_COLOR, "#d95f02"][: len(g_labels)]
    wedges, texts, autotexts = ax4.pie(
        g_vals,
        labels=g_labels,
        autopct=lambda p: f"{p:.1f}%\n({int(round(p * sum(g_vals) / 100.0)):,})",
        startangle=140,
        colors=g_colors,
        wedgeprops={"edgecolor": "black", "linewidth": 0.8},
        textprops={"fontsize": 9.5},
    )
    for autotext in autotexts:
        autotext.set_fontweight("bold")
    ax4.set_title(f"D) Composição Bruta das Anotações ({len(boxes_df):,} linhas totais)")

    fig.suptitle(
        "VisDrone2019-DET — Análise de Oclusão, Truncamento e Regiões Ignoradas",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    out_path = out_dir / "04_occlusion_and_truncation.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    veh_occ = veh_df["occlusion"].value_counts(normalize=True) * 100.0
    return {
        "figure": str(out_path.name),
        "raw_total_rows": int(len(boxes_df)),
        "ignored_or_others_rows": int((~boxes_df["is_valid_yolo"]).sum()),
        "vehicle_occlusion_pct": {
            OCCLUSION_LABELS.get(int(k), str(k)): round(float(v), 2)
            for k, v in veh_occ.items()
        },
        "vehicle_truncation_pct": round(float((veh_df["truncation"] > 0).mean() * 100.0), 2),
    }


def plot_representative_scenes(
    root: Path, images_df: pd.DataFrame, boxes_df: pd.DataFrame, out_dir: Path
) -> dict:
    """Figure 5: Grid of representative VisDrone validation scenes with Ground Truth annotations."""
    val_imgs = images_df[images_df["split"] == "val"].copy()
    val_boxes = boxes_df[(boxes_df["split"] == "val") & boxes_df["is_valid_yolo"]].copy()

    # Select 4 diverse, informative scenes from val:
    # 1. Multi-class vehicle scene (car + van + truck + bus all present)
    veh_variety = (
        val_boxes[val_boxes["is_vehicle"]]
        .groupby("image_id")["class_name"]
        .nunique()
    )
    multi_veh_ids = veh_variety[veh_variety == 4].index.tolist()
    scene1_id = multi_veh_ids[0] if multi_veh_ids else val_imgs.iloc[0]["image_id"]

    # 2. High vehicle density urban traffic scene (~60-100 vehicles)
    dense_candidates = val_imgs[
        (val_imgs["vehicle_objects"] >= 60)
        & (val_imgs["vehicle_objects"] <= 110)
        & (~val_imgs["image_id"].isin([scene1_id]))
    ].sort_values("vehicle_objects", ascending=False)
    scene2_id = dense_candidates.iloc[0]["image_id"] if len(dense_candidates) else val_imgs.iloc[1]["image_id"]

    # 3. Heavy vehicle occlusion scene
    occ_veh = (
        val_boxes[val_boxes["is_vehicle"] & (val_boxes["occlusion"] >= 1)]
        .groupby("image_id")
        .size()
        .sort_values(ascending=False)
    )
    occ_candidates = [i for i in occ_veh.index if i not in {scene1_id, scene2_id}]
    scene3_id = occ_candidates[0] if occ_candidates else val_imgs.iloc[2]["image_id"]

    # 4. Moderate density / larger scale vehicles (lower altitude oblique view)
    med_box_area = (
        val_boxes[val_boxes["is_vehicle"]]
        .groupby("image_id")["rel_area_pct"]
        .median()
        .sort_values(ascending=False)
    )
    large_candidates = [
        i
        for i in med_box_area.index
        if i not in {scene1_id, scene2_id, scene3_id}
        and int(val_imgs.loc[val_imgs["image_id"] == i, "vehicle_objects"].iloc[0]) >= 8
    ]
    scene4_id = large_candidates[0] if large_candidates else val_imgs.iloc[3]["image_id"]

    selected_scenes = [
        (scene1_id, "Cena 1: Diversidade Completa (car, van, truck, bus)"),
        (scene2_id, "Cena 2: Alta Densidade de Tráfego Urbano"),
        (scene3_id, "Cena 3: Forte Oclusão Parcial/Severa em Veículos"),
        (scene4_id, "Cena 4: Baixa Altitude / Veículos de Maior Escala"),
    ]

    fig, axes = plt.subplots(
        2, 2, figsize=(16, 11.5), gridspec_kw={"hspace": 0.18, "wspace": 0.06}
    )
    axes = axes.flatten()

    scene_records = []
    for ax, (img_id, title_prefix) in zip(axes, selected_scenes):
        row = val_imgs[val_imgs["image_id"] == img_id].iloc[0]
        img_path = root / row["image_path"]
        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        sub_boxes = val_boxes[val_boxes["image_id"] == img_id]

        for _, b in sub_boxes.iterrows():
            cls_name = b["class_name"]
            is_veh = b["is_vehicle"]
            color = CLASS_PALETTE.get(cls_name, "#cccccc") if is_veh else "#aaaaaa"
            lw = 1.6 if is_veh else 0.7
            alpha = 0.95 if is_veh else 0.45
            rect = patches.Rectangle(
                (b["x_left"], b["y_top"]),
                b["w_box"],
                b["h_box"],
                linewidth=lw,
                edgecolor=color,
                facecolor="none",
                alpha=alpha,
            )
            ax.add_patch(rect)

        veh_breakdown = (
            sub_boxes[sub_boxes["is_vehicle"]]["class_name"]
            .value_counts()
            .to_dict()
        )
        breakdown_str = ", ".join(f"{k}:{v}" for k, v in veh_breakdown.items())
        ax.set_title(
            f"{title_prefix}\n{img_id} ({row['resolution']}) — Veículos: {row['vehicle_objects']} ({breakdown_str})",
            fontsize=9.2,
        )
        ax.axis("off")
        scene_records.append({
            "image_id": img_id,
            "title": title_prefix,
            "resolution": row["resolution"],
            "vehicle_objects": int(row["vehicle_objects"]),
            "total_objects": int(row["valid_objects"]),
            "vehicle_breakdown": {k: int(v) for k, v in veh_breakdown.items()},
        })

    legend_patches = [
        patches.Patch(edgecolor=CLASS_PALETTE["car"], facecolor="none", linewidth=2, label="car"),
        patches.Patch(edgecolor=CLASS_PALETTE["van"], facecolor="none", linewidth=2, label="van"),
        patches.Patch(edgecolor=CLASS_PALETTE["truck"], facecolor="none", linewidth=2, label="truck"),
        patches.Patch(edgecolor=CLASS_PALETTE["bus"], facecolor="none", linewidth=2, label="bus"),
        patches.Patch(edgecolor="#aaaaaa", facecolor="none", linewidth=1, label="Outras classes (pedestrian, motor, etc.)"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=5,
        frameon=True,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        "VisDrone2019-DET (Val) — Cenas Aéreas Representativas e Anotações Ground Truth",
        fontsize=14,
        fontweight="bold",
        y=0.96,
    )
    out_path = out_dir / "05_representative_scenes.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return {"figure": str(out_path.name), "scenes": scene_records}


def plot_qualitative_detector_results(
    root: Path, images_df: pd.DataFrame, boxes_df: pd.DataFrame, out_dir: Path
) -> dict:
    """Figure 6: Qualitative YOLOv8s detector results and vehicle crops on VisDrone val."""
    with (root / "configs/default.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    det_cfg = config["detector"]
    weights_path = root / det_cfg["weights"]
    if not weights_path.is_file():
        return {"status": "skipped", "reason": f"Missing detector weights at {weights_path}"}

    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    # Resolve vehicle class names dynamically against the checkpoint's names dict
    ckpt_names: dict[int, str] = model.names
    target_vehicle_names = set(det_cfg["vehicle_classes"])
    allowed_cls_ids = [
        cid for cid, cname in ckpt_names.items() if cname in target_vehicle_names
    ]

    val_imgs = images_df[images_df["split"] == "val"].copy()
    val_boxes = boxes_df[(boxes_df["split"] == "val") & boxes_df["is_valid_yolo"]].copy()

    # Pick 3 distinct validation scenes with good mix of vehicles
    candidates = val_imgs[
        (val_imgs["vehicle_objects"] >= 18) & (val_imgs["vehicle_objects"] <= 45)
    ].sort_values("vehicle_objects")
    sample_rows = [
        candidates.iloc[0],
        candidates.iloc[len(candidates) // 2],
        candidates.iloc[-1],
    ]

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.28, wspace=0.12)

    qualitative_records = []
    sample_crops: list[tuple[Image.Image, str, float]] = []

    for row_idx, img_row in enumerate(sample_rows):
        img_id = img_row["image_id"]
        img_path = root / img_row["image_path"]
        pil_img = Image.open(img_path).convert("RGB")
        img_w, img_h = pil_img.size

        # Ground truth vehicles
        gt_veh = val_boxes[(val_boxes["image_id"] == img_id) & val_boxes["is_vehicle"]]

        # Run YOLOv8s inference with project config parameters
        preds = model.predict(
            source=str(img_path),
            conf=float(det_cfg["confidence_threshold"]),
            iou=float(det_cfg["iou_threshold"]),
            imgsz=int(det_cfg["image_size"]),
            classes=allowed_cls_ids,
            verbose=False,
        )[0]

        # Left column: Ground Truth Vehicles
        ax_gt = fig.add_subplot(gs[row_idx, 0])
        ax_gt.imshow(pil_img)
        for _, b in gt_veh.iterrows():
            c_name = b["class_name"]
            rect = patches.Rectangle(
                (b["x_left"], b["y_top"]),
                b["w_box"],
                b["h_box"],
                linewidth=1.5,
                edgecolor=CLASS_PALETTE.get(c_name, "#1f77b4"),
                facecolor="none",
            )
            ax_gt.add_patch(rect)
        ax_gt.set_title(
            f"Ground Truth ({img_id}) — {len(gt_veh)} Veículos Anotados",
            fontsize=10.5,
        )
        ax_gt.axis("off")

        # Right column: YOLOv8s Detections
        ax_det = fig.add_subplot(gs[row_idx, 1])
        ax_det.imshow(pil_img)
        boxes_xyxy = preds.boxes.xyxy.cpu().numpy() if preds.boxes is not None else np.empty((0, 4))
        confs = preds.boxes.conf.cpu().numpy() if preds.boxes is not None else np.empty((0,))
        cls_ids = preds.boxes.cls.cpu().numpy().astype(int) if preds.boxes is not None else np.empty((0,), dtype=int)

        det_counts: dict[str, int] = {}
        margin = int(det_cfg["crop_margin_pixels"])
        for (x1, y1, x2, y2), conf, cid in zip(boxes_xyxy, confs, cls_ids):
            c_name = ckpt_names[int(cid)]
            det_counts[c_name] = det_counts.get(c_name, 0) + 1
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=1.5,
                edgecolor=CLASS_PALETTE.get(c_name, "#1f77b4"),
                facecolor="none",
            )
            ax_det.add_patch(rect)

            if len(sample_crops) < 8 and (x2 - x1) >= 40 and (y2 - y1) >= 40:
                cx1 = max(0, int(round(x1)) - margin)
                cy1 = max(0, int(round(y1)) - margin)
                cx2 = min(img_w, int(round(x2)) + margin)
                cy2 = min(img_h, int(round(y2)) + margin)
                sample_crops.append((pil_img.crop((cx1, cy1, cx2, cy2)), c_name, float(conf)))

        det_summary = ", ".join(f"{k}:{v}" for k, v in sorted(det_counts.items()))
        ax_det.set_title(
            f"YOLOv8s (conf≥{det_cfg['confidence_threshold']}, imgsz={det_cfg['image_size']}) — {len(boxes_xyxy)} Detecções ({det_summary})",
            fontsize=10.5,
        )
        ax_det.axis("off")

        qualitative_records.append({
            "image_id": img_id,
            "gt_vehicles": int(len(gt_veh)),
            "detected_vehicles": int(len(boxes_xyxy)),
            "detected_by_class": det_counts,
            "mean_confidence": round(float(confs.mean()), 3) if len(confs) else 0.0,
        })

    legend_patches = [
        patches.Patch(edgecolor=CLASS_PALETTE[c], facecolor="none", linewidth=2, label=c)
        for c in ["car", "van", "truck", "bus"]
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=4,
        frameon=True,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        "Resultado Qualitativo do Detector Pré-Treinado (dronefreak/visdrone-yolov8s) vs. Ground Truth",
        fontsize=14,
        fontweight="bold",
        y=0.96,
    )
    out_path = out_dir / "06_qualitative_detector_yolov8s.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "figure": str(out_path.name),
        "checkpoint_classes": {int(k): v for k, v in ckpt_names.items()},
        "filtered_vehicle_class_ids": allowed_cls_ids,
        "scenes_evaluated": qualitative_records,
    }


def plot_reference_gallery_bridge(root: Path, out_dir: Path) -> dict:
    """Figure 7: Reference gallery analysis (83 simulated crops) contrasting detector vs recognition taxonomy."""
    ref_root = root / "data/reference_dataset"
    with (root / "configs/classes.yaml").open(encoding="utf-8") as handle:
        classes_cfg = yaml.safe_load(handle)

    labels = classes_cfg["labels"]
    rows = []
    sample_images: dict[str, Path] = {}
    for label in labels:
        cls_dir = ref_root / label
        for p in sorted(cls_dir.glob("*")):
            if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                continue
            with Image.open(p) as im:
                w, h = im.size
            is_aug = "_aug_" in p.stem
            if label not in sample_images and not is_aug:
                sample_images[label] = p
            rows.append({
                "label": label,
                "filename": p.name,
                "is_augmented": is_aug,
                "origin_type": "Variante Aumentada (_aug_)" if is_aug else "Crop Simulado Base (3D)",
                "width": w,
                "height": h,
                "area_px": w * h,
                "aspect_ratio": w / h if h > 0 else 1.0,
            })

    ref_df = pd.DataFrame(rows)

    fig = plt.figure(figsize=(16, 8.8))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.25, 1.0], hspace=0.42, wspace=0.35)

    # Top Left (spans 3 cols): Base vs Augmented counts per class
    ax1 = fig.add_subplot(gs[0, :3])
    ct = (
        ref_df.groupby(["label", "origin_type"])
        .size()
        .unstack(fill_value=0)
        .reindex(labels)
    )
    base_col = "Crop Simulado Base (3D)"
    aug_col = "Variante Aumentada (_aug_)"
    base_vals = ct[base_col].values if base_col in ct else np.zeros(len(labels))
    aug_vals = ct[aug_col].values if aug_col in ct else np.zeros(len(labels))
    ax1.bar(labels, base_vals, label=base_col, color="#2b5c8f", edgecolor="black", width=0.55)
    ax1.bar(labels, aug_vals, bottom=base_vals, label=aug_col, color="#6baed6", edgecolor="black", width=0.55)
    ax1.set_ylabel("Número de Imagens de Referência")
    ax1.set_ylim(0, (base_vals + aug_vals).max() * 1.22)
    ax1.set_title("A) Galeria Simulada 3D: Crops Base vs. Aumentados (n = 83)", fontsize=11)
    for idx, (b_v, a_v) in enumerate(zip(base_vals, aug_vals)):
        tot = b_v + a_v
        ax1.text(idx, tot + 0.6, f"{int(tot)} ({int(b_v)}+{int(a_v)})", ha="center", fontweight="bold", fontsize=9)
    ax1.legend(frameon=True)

    # Top Right (spans 2 cols): Crop resolution scatter
    ax2 = fig.add_subplot(gs[0, 3:])
    sns.scatterplot(
        data=ref_df,
        x="width",
        y="height",
        hue="label",
        style="origin_type",
        s=70,
        ax=ax2,
    )
    ax2.set_xlabel("Largura do Crop (px)")
    ax2.set_ylabel("Altura do Crop (px)")
    ax2.set_title("B) Dimensões dos Crops na Galeria (px)", fontsize=11)
    ax2.legend(fontsize=8, loc="best", frameon=True)

    # Bottom row: 1 sample simulated crop per operational vehicle class
    for idx, label in enumerate(labels):
        ax_img = fig.add_subplot(gs[1, idx])
        img_p = sample_images.get(label)
        if img_p and img_p.is_file():
            im = Image.open(img_p).convert("RGB")
            ax_img.imshow(im)
            ax_img.set_title(f"{label}\n(Simulado: {im.size[0]}×{im.size[1]}px)", fontsize=9.5)
        ax_img.axis("off")

    fig.suptitle(
        "Ponte Metodológica: Galeria de Referência Simulada (5 Modelos Operacionais) vs. Taxonomia Ampla do Detector",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )
    out_path = out_dir / "07_reference_gallery_overview.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    summary_by_label = {}
    for label in labels:
        sub = ref_df[ref_df["label"] == label]
        summary_by_label[label] = {
            "total": int(len(sub)),
            "base_crops": int((~sub["is_augmented"]).sum()),
            "augmented_crops": int(sub["is_augmented"].sum()),
            "median_width_px": round(float(sub["width"].median()), 1),
            "median_height_px": round(float(sub["height"].median()), 1),
        }
    return {
        "figure": str(out_path.name),
        "total_reference_crops": int(len(ref_df)),
        "base_crops": int((~ref_df["is_augmented"]).sum()),
        "augmented_crops": int(ref_df["is_augmented"].sum()),
        "per_class": summary_by_label,
    }


def run_visdrone_eda(root: Path, out_dir: Path | None = None) -> dict:
    """Execute complete VisDrone + Reference Gallery EDA and save figures and JSON/CSV summaries."""
    if out_dir is None:
        out_dir = root / "outputs/eda_visdrone"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    images_df, boxes_df = load_visdrone_dataframe(root, splits=("train", "val"))
    images_df.to_csv(out_dir / "visdrone_images_summary.csv", index=False)

    # Save aggregated class summary CSV
    freq_summary = plot_class_frequencies(images_df, boxes_df, fig_dir)
    pd.DataFrame(freq_summary["classes"]).to_csv(out_dir / "visdrone_class_frequencies.csv", index=False)

    size_summary = plot_bbox_sizes_and_scales(boxes_df, fig_dir)
    res_summary = plot_resolutions_and_scene_density(images_df, boxes_df, fig_dir)
    occ_summary = plot_occlusion_and_truncation(boxes_df, fig_dir)
    scenes_summary = plot_representative_scenes(root, images_df, boxes_df, fig_dir)
    det_summary = plot_qualitative_detector_results(root, images_df, boxes_df, fig_dir)
    ref_summary = plot_reference_gallery_bridge(root, fig_dir)

    report = {
        "dataset": "VisDrone2019-DET (train + val)",
        "total_images": int(len(images_df)),
        "splits_image_count": {
            k: int(v) for k, v in images_df["split"].value_counts().items()
        },
        "class_frequencies": freq_summary,
        "bbox_sizes_and_scales": size_summary,
        "resolutions_and_density": res_summary,
        "occlusion_and_truncation": occ_summary,
        "representative_scenes": scenes_summary,
        "qualitative_detector": det_summary,
        "reference_gallery_overview": ref_summary,
    }
    report_path = out_dir / "eda_visdrone_summary.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    summary = run_visdrone_eda(repo_root)
    print(json.dumps({"status": "completed", "total_images": summary["total_images"]}, indent=2))
