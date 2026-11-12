"""Compare measured EDA summaries without equating incompatible class labels.

Prefer the native VisDrone JSON/CSV exports. If they are unavailable, recover
only tables and JSON actually saved in the executed VisDrone notebook outputs;
never parse numerical findings from prose or invent missing distributions.
"""

from __future__ import annotations

from html.parser import HTMLParser
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd

from .eda_mvrsd import CLASSES, STYLE, save_figure, sha256

VIS_VEHICLES = ("car", "van", "truck", "bus")


class OutputTable(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows, self.row, self.cell = [], [], None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in {"td", "th"}:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cell is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        elif tag == "tr" and self.row:
            self.rows.append(self.row)

    def frame(self) -> pd.DataFrame:
        if len(self.rows) < 2 or any(len(row) != len(self.rows[0]) for row in self.rows):
            raise ValueError("Unsupported saved notebook table; regenerate the native VisDrone summary")
        frame = pd.DataFrame(self.rows[1:], columns=self.rows[0])
        for name in frame.columns:
            if name not in {"class_name", "model_label", "is_vehicle"}:
                frame[name] = pd.to_numeric(frame[name].replace({"NaN": pd.NA, "nan": pd.NA, "": pd.NA}), errors="raise")
        return frame


def load_visdrone_summary(root: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    path = root / "outputs/eda_visdrone/eda_visdrone_summary.json"
    if path.is_file():
        summary = json.loads(path.read_text())
        frequency_path = path.parent / "visdrone_class_frequencies.csv"
        frequency = pd.read_csv(frequency_path) if frequency_path.is_file() else pd.DataFrame(summary["class_frequencies"]["classes"])
        size = pd.DataFrame.from_dict(summary["bbox_sizes_and_scales"]["per_class_box_stats"], orient="index")
        size.index.name = "class_name"
        return summary, frequency, size.reset_index(), {"kind": "native_json_csv", "file": str(path), "sha256": sha256(path)}
    path = root / "notebooks/01_eda_visdrone.ipynb"
    if not path.is_file():
        raise FileNotFoundError("Run the VisDrone EDA or provide the executed 01 notebook")
    notebook = json.loads(path.read_text())
    frequency = size = density = None
    setup_output = ""
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        outputs = cell.get("outputs", [])
        streams = "".join("".join(output.get("text", [])) for output in outputs if output.get("output_type") == "stream")
        if "Total de imagens" in streams:
            setup_output = streams
        if "Resumo de Resoluções e Densidade" in source and streams.strip():
            density = json.loads(streams[streams.index("{"):])
        for output in outputs:
            html = output.get("data", {}).get("text/html")
            if not html:
                continue
            parser = OutputTable()
            parser.feed("".join(html))
            table = parser.frame()
            if {"class_name", "total_count", "image_presence_pct"}.issubset(table.columns):
                frequency = table
            if {"class_name", "median_rel_area_pct", "small_pct"}.issubset(table.columns):
                size = table
    match = re.search(r"Total de imagens.*?:\s*([\d,.]+)", setup_output)
    if frequency is None or size is None or density is None or not match:
        raise FileNotFoundError("The VisDrone notebook has no complete saved EDA outputs. Run its EDA and export JSON/CSV first.")
    total_images = int(match.group(1).replace(",", "").replace(".", ""))
    vehicles = frequency[frequency.class_name.isin(VIS_VEHICLES)]
    summary = {"dataset": "VisDrone2019-DET (train + val; saved notebook outputs)", "total_images": total_images,
               "class_frequencies": {"total_valid_boxes": int(frequency.total_count.sum()),
                                      "total_vehicle_boxes": int(vehicles.total_count.sum())},
               "resolutions_and_density": density}
    provenance = {"kind": "executed_notebook_outputs", "file": str(path.relative_to(root)), "sha256": sha256(path),
                  "note": "Historical saved HTML tables and JSON stdout; raw VisDrone images were not reprocessed. "
                          "Percentages and geometry statistics retain the rounding of those saved outputs."}
    return summary, frequency, size, provenance


def run_eda_comparison(root: Path, out_dir: Path | None = None) -> dict:
    root = root.resolve()
    out_dir = out_dir or root / "outputs/eda_comparison"
    mv_path = root / "outputs/eda_mvrsd/eda_mvrsd_summary.json"
    if not mv_path.is_file():
        raise FileNotFoundError("Execute 02_eda_mvrsd.ipynb before the comparison notebook")
    mv = json.loads(mv_path.read_text())
    vis, vis_frequency, vis_size, provenance = load_visdrone_summary(root)
    mv_frequency = pd.DataFrame(mv["class_frequencies"])
    mv_size = pd.DataFrame.from_dict(mv["bbox_sizes_and_scales"]["per_class"], orient="index")
    mv_size.index.name = "class_name"
    mv_size = mv_size.reset_index()
    scope = pd.DataFrame([
        {"dataset": "VisDrone", "images": vis["total_images"], "annotated_images": vis["total_images"],
         "all_valid_boxes": vis["class_frequencies"]["total_valid_boxes"],
         "vehicle_boxes": vis["class_frequencies"]["total_vehicle_boxes"],
         "scope": vis["dataset"], "vehicle_taxonomy": ", ".join(VIS_VEHICLES)},
        {"dataset": "MVRSD", "images": mv["total_images"], "annotated_images": mv["annotated_images"],
         "all_valid_boxes": mv["valid_boxes"], "vehicle_boxes": mv["valid_boxes"],
         "scope": ", ".join(mv["splits_image_count"]), "vehicle_taxonomy": ", ".join(CLASSES)},
    ])
    geometries = pd.concat([vis_size[vis_size.class_name.isin(VIS_VEHICLES)].assign(dataset="VisDrone"),
                            mv_size.assign(dataset="MVRSD")], ignore_index=True)
    geometries["display_label"] = geometries.dataset+": "+geometries.class_name
    densities = pd.DataFrame({"VisDrone": vis["resolutions_and_density"]["vehicles_per_image"],
                              "MVRSD": mv["resolutions_and_density"]["objects_per_image"]}).loc[["mean", "median", "p90", "max"]]
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    scope.to_csv(out_dir / "dataset_scope.csv", index=False)
    geometries.to_csv(out_dir / "vehicle_box_geometry.csv", index=False)
    densities.to_csv(out_dir / "vehicle_density.csv", index_label="statistic")
    names = []
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout="constrained")
        for axis, frequency, dataset in [(axes[0], vis_frequency, "VisDrone"), (axes[1], mv_frequency, "MVRSD")]:
            frequency = frequency.copy()
            frequency["share"] = frequency.total_count/frequency.total_count.sum()*100
            axis.barh(frequency.class_name, frequency.share, color=["#1f77b4" if c in VIS_VEHICLES or dataset == "MVRSD" else "#8c969e" for c in frequency.class_name])
            axis.set(title=dataset+" — taxonomia original", xlabel="% das anotações do próprio dataset")
        names.append(save_figure(fig, figures, "01_class_taxonomies.png", "Frequências relativas — categorias não são equivalentes entre datasets"))
        fig, axes = plt.subplots(1, 2, figsize=(17, 6), layout="constrained")
        geometries.set_index("display_label")[["small_pct", "medium_pct", "large_pct"]].rename(
            columns={"small_pct": "Small (<32² px²)", "medium_pct": "Medium (32²–96² px²)", "large_pct": "Large (≥96² px²)"}).plot.bar(
            stacked=True, ax=axes[0], rot=65, color=["#e45756", "#f58518", "#54a24b"])
        axes[0].set(title="A) Escala COCO por classe", ylabel="% da classe", xlabel="", ylim=(0, 100))
        geometries.plot.bar(x="display_label", y="median_rel_area_pct", ax=axes[1], rot=65, legend=False,
                             color=["#1f77b4" if d == "VisDrone" else "#ff7f0e" for d in geometries.dataset])
        axes[1].set(title="B) Mediana da área relativa por classe", ylabel="% da área da imagem", xlabel="")
        names.append(save_figure(fig, figures, "02_vehicle_scale_and_relative_area.png", "VisDrone × MVRSD — Escala COCO e área relativa por classe"))
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")
        densities.loc[["mean", "median", "p90"]].rename(index={"mean": "Média", "median": "Mediana", "p90": "P90"}).plot.bar(ax=axes[0], rot=0, color=["#1f77b4", "#ff7f0e"])
        axes[0].set(title="A) Densidade nas imagens anotadas", ylabel="Veículos / imagem", xlabel="Estatística")
        geometries.plot.bar(x="display_label", y="median_aspect_ratio", ax=axes[1], rot=65, legend=False,
                             color=["#1f77b4" if d == "VisDrone" else "#ff7f0e" for d in geometries.dataset])
        axes[1].axhline(1, color="black", linestyle="--")
        axes[1].set(title="B) Razão de aspecto mediana", ylabel="Largura / altura", xlabel="")
        names.append(save_figure(fig, figures, "03_density_and_aspect_ratios.png", "Comparação descritiva — altitude, resolução e taxonomia diferem"))
        vis_resolutions = dict(vis["resolutions_and_density"]["top_resolutions"])
        remainder = vis["total_images"] - sum(vis_resolutions.values())
        if remainder > 0:
            vis_resolutions["Outras resoluções"] = remainder
        mv_resolutions = mv["resolutions_and_density"]["resolutions"]
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")
        for axis, values, dataset in [(axes[0], vis_resolutions, "VisDrone"), (axes[1], mv_resolutions, "MVRSD")]:
            pd.Series(values).plot.barh(ax=axis, color="#1f77b4" if dataset == "VisDrone" else "#ff7f0e")
            axis.set(title=dataset, xlabel="Imagens", ylabel="Resolução nativa")
        names.append(save_figure(fig, figures, "04_native_resolutions.png", "Resoluções medidas — contagens por dataset, sem redimensionamento"))
    report = {"sources": {"visdrone": provenance, "mvrsd": {"kind": "native_json_csv", "file": str(mv_path.relative_to(root)), "sha256": sha256(mv_path)}},
              "scope": scope.to_dict(orient="records"), "figures": names,
              "resolutions": {"visdrone": vis_resolutions, "mvrsd": mv_resolutions},
              "mvrsd_metadata_availability": mv["metadata_availability"],
              "limitations": ["Taxonomies are not mapped one-to-one", "No claim of detector generalization or dataset causality",
                              "VisDrone occlusion annotations cannot be compared with absent MVRSD occlusion labels",
                              "The published 0.3 m/pixel MVRSD resolution is not an inferred VisDrone ground sample distance",
                              "The final forest evaluation dataset is not used"]}
    (out_dir / "eda_comparison_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    return report
