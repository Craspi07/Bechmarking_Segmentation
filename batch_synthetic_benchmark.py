#!/usr/bin/env python3
"""
Batch Synthetic Dot Segmentation Benchmark
===========================================

Standalone command-line tool (no Streamlit required) that sweeps synthetic
dot-image generation across Poisson noise, Gaussian noise, and dot-size
("size distribution") parameters, runs both a custom-trained Cellpose model
and Cellpose's native pretrained model on every generated image, evaluates
both against the exact synthetic ground truth (IoU / AP / precision /
recall at COCO-style IoU thresholds 0.5-0.95), and writes:

  * results_raw.csv      -- one row per (image, model, IoU threshold)
  * results_summary.csv  -- the same, aggregated across replicate images
  * report.html           -- a self-contained, interactive Plotly report
                              comparing the two models' AP across the grid

Usage
-----
    python batch_synthetic_benchmark.py --custom-model path/to/model.pth

    python batch_synthetic_benchmark.py --config batch_config_example.json

    python batch_synthetic_benchmark.py --print-grid   # preview grid size only

Run `python batch_synthetic_benchmark.py --help` for all options. See
batch_config_example.json for the equivalent, more expressive JSON config
format (useful for large/irregular sweeps or reproducible runs).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tifffile

from modules import synthetic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _parse_float_list(text: str) -> List[float]:
    return [float(x) for x in text.split(",") if x.strip() != ""]


@dataclass
class SweepConfig:
    poisson_scales: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 5.0])
    gaussian_sigmas: List[float] = field(default_factory=lambda: [2.0, 5.0, 10.0, 20.0])
    radius_means: List[float] = field(default_factory=lambda: [3.0, 5.0, 8.0])
    radius_stds: List[float] = field(default_factory=lambda: [1.0, 1.5])
    n_replicates: int = 1
    image_height: int = 512
    image_width: int = 512
    n_dots: int = 150
    blur_sigma: float = 1.2
    iou_thresholds: List[float] = field(default_factory=lambda: [round(float(t), 2) for t in np.arange(0.5, 1.0, 0.05)])
    seed: int = 0
    custom_model_path: Optional[str] = None
    native_model_name: str = "cpsam_v2"
    run_custom: bool = True
    run_native: bool = True
    diameter: Optional[float] = None
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    gpu: bool = False
    output_dir: str = "batch_benchmark_results"
    save_images: bool = False

    @classmethod
    def from_json(cls, path: str) -> "SweepConfig":
        """Load a sweep config from JSON. See batch_config_example.json for
        the expected structure -- unrecognized/omitted keys fall back to
        this dataclass's defaults."""
        data = json.loads(Path(path).read_text())
        grid = data.pop("grid", {})
        models = data.pop("models", {})
        cellpose_eval = data.pop("cellpose_eval", {})
        image = data.pop("image", {})
        defaults = cls()

        kwargs = {
            "poisson_scales": grid.get("poisson_scales", defaults.poisson_scales),
            "gaussian_sigmas": grid.get("gaussian_sigmas", defaults.gaussian_sigmas),
            "radius_means": grid.get("radius_means", defaults.radius_means),
            "radius_stds": grid.get("radius_stds", defaults.radius_stds),
            "image_height": image.get("height", defaults.image_height),
            "image_width": image.get("width", defaults.image_width),
            "n_dots": image.get("n_dots", defaults.n_dots),
            "blur_sigma": image.get("blur_sigma", defaults.blur_sigma),
            "diameter": cellpose_eval.get("diameter", defaults.diameter),
            "flow_threshold": cellpose_eval.get("flow_threshold", defaults.flow_threshold),
            "cellprob_threshold": cellpose_eval.get("cellprob_threshold", defaults.cellprob_threshold),
            "gpu": cellpose_eval.get("gpu", defaults.gpu),
        }
        if "custom" in models:
            kwargs["custom_model_path"] = models["custom"].get("pretrained_model")
        if "native" in models:
            kwargs["native_model_name"] = models["native"].get("pretrained_model", defaults.native_model_name)

        for key in ("n_replicates", "seed", "output_dir", "save_images", "run_custom", "run_native", "iou_thresholds"):
            if key in data:
                kwargs[key] = data[key]

        return cls(**kwargs)

    @property
    def grid_combinations(self) -> List[tuple]:
        """Every (radius_mean, radius_std, poisson_scale, gaussian_sigma)
        combination -- the full cross product of the size-distribution and
        noise-level sweeps."""
        return list(itertools.product(self.radius_means, self.radius_stds, self.poisson_scales, self.gaussian_sigmas))

    @property
    def n_models(self) -> int:
        return int(self.run_custom) + int(self.run_native)

    @property
    def total_runs(self) -> int:
        return len(self.grid_combinations) * self.n_replicates * self.n_models


# ---------------------------------------------------------------------------
# Model loading / inference
# ---------------------------------------------------------------------------


def load_cellpose_model(pretrained_model: str, gpu: bool):
    """Load a Cellpose model (custom file path or built-in model name) and
    report the device it actually resolved to -- Cellpose silently falls
    back to CPU if `gpu=True` was requested but no usable CUDA/MPS device
    is visible to PyTorch, so this is surfaced explicitly rather than left
    to guesswork.
    """
    from cellpose import models as cellpose_models

    model = cellpose_models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model)
    print(f"  -> device: {model.device} (gpu={model.gpu})")
    if gpu and not model.gpu:
        print("  ! GPU was requested but PyTorch did not report a usable CUDA/MPS device -- running on CPU.")
    resolved_name = Path(str(model.pretrained_model)).name
    if not Path(pretrained_model).exists() and resolved_name != pretrained_model:
        print(f"  ! Requested model '{pretrained_model}' was not recognized; Cellpose fell back to '{resolved_name}'.")
    return model


def run_model(model, image: np.ndarray, diameter, flow_threshold: float, cellprob_threshold: float) -> np.ndarray:
    masks, _flows, _styles = model.eval(
        image, diameter=diameter, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold
    )
    return np.asarray(masks).astype(np.int32)


# ---------------------------------------------------------------------------
# Sweep execution
# ---------------------------------------------------------------------------


def _empty_eval_row(thresholds: np.ndarray, error: str) -> pd.DataFrame:
    """A placeholder evaluation row (all-NaN metrics) used when a model
    run fails on a given image, so the sweep continues instead of aborting
    and the failure is still visible in the output CSV."""
    df = pd.DataFrame({"iou_threshold": thresholds})
    for col in ("TP", "FP", "FN", "precision", "recall", "f1", "AP", "mean_matched_iou"):
        df[col] = np.nan
    df["error"] = error
    return df


def run_sweep(config: SweepConfig) -> pd.DataFrame:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.save_images:
        (output_dir / "images").mkdir(exist_ok=True)

    models: Dict[str, object] = {}
    if config.run_custom:
        if not config.custom_model_path:
            raise ValueError("run_custom is True but no custom_model_path was given.")
        print(f"Loading custom model from {config.custom_model_path} ...")
        models["custom"] = load_cellpose_model(config.custom_model_path, config.gpu)
    if config.run_native:
        print(f"Loading native pretrained model '{config.native_model_name}' ...")
        models["native"] = load_cellpose_model(config.native_model_name, config.gpu)
    if not models:
        raise ValueError("Both run_custom and run_native are disabled -- nothing to do.")

    thresholds = np.array(config.iou_thresholds, dtype=float)
    combos = config.grid_combinations
    total_images = len(combos) * config.n_replicates
    print(
        f"Running {total_images} synthetic image(s) x {len(models)} model(s) "
        f"= {total_images * len(models)} evaluations."
    )

    records = []
    run_index = 0
    t_start = time.time()
    for radius_mean, radius_std, poisson_scale, gaussian_sigma in combos:
        for replicate in range(config.n_replicates):
            run_index += 1
            seed = config.seed + run_index  # deterministic and unique per combo+replicate

            raw, gt = synthetic.generate_synthetic_dots(
                shape=(config.image_height, config.image_width),
                n_dots=config.n_dots,
                radius_mean=radius_mean,
                radius_std=radius_std,
                blur_sigma=config.blur_sigma,
                poisson_scale=poisson_scale,
                gaussian_noise_sigma=gaussian_sigma,
                seed=seed,
            )
            n_gt = int(gt.max())
            elapsed = time.time() - t_start
            print(
                f"[{run_index}/{total_images}] radius={radius_mean}+/-{radius_std}px "
                f"poisson_scale={poisson_scale} gaussian_sigma={gaussian_sigma} replicate={replicate} "
                f"(n_gt={n_gt} objects, {elapsed:.0f}s elapsed)"
            )

            if config.save_images:
                tag = f"r{radius_mean}-{radius_std}_p{poisson_scale}_g{gaussian_sigma}_rep{replicate}"
                tifffile.imwrite(output_dir / "images" / f"{tag}_raw.tif", raw.astype(np.float32))
                np.save(output_dir / "images" / f"{tag}_gt.npy", gt)

            for model_name, model in models.items():
                try:
                    pred_mask = run_model(model, raw, config.diameter, config.flow_threshold, config.cellprob_threshold)
                    eval_df = synthetic.evaluate_against_ground_truth(gt, pred_mask, thresholds=thresholds)
                    eval_df["error"] = None
                    n_pred = int(pred_mask.max())
                except Exception as exc:  # noqa: BLE001 -- keep the sweep going on a per-image model failure
                    print(f"    ! {model_name} model failed on this image: {exc}")
                    eval_df = _empty_eval_row(thresholds, str(exc))
                    n_pred = np.nan

                eval_df["model"] = model_name
                eval_df["radius_mean"] = radius_mean
                eval_df["radius_std"] = radius_std
                eval_df["poisson_scale"] = poisson_scale
                eval_df["gaussian_sigma"] = gaussian_sigma
                eval_df["replicate"] = replicate
                eval_df["seed"] = seed
                eval_df["n_gt_objects"] = n_gt
                eval_df["n_pred_objects"] = n_pred
                records.append(eval_df)

    return pd.concat(records, ignore_index=True)


def save_results(results: pd.DataFrame, config: SweepConfig) -> Dict[str, pd.DataFrame]:
    output_dir = Path(config.output_dir)
    raw_path = output_dir / "results_raw.csv"
    results.to_csv(raw_path, index=False)
    print(f"Saved raw per-image results to {raw_path}")

    group_cols = ["model", "radius_mean", "radius_std", "poisson_scale", "gaussian_sigma", "iou_threshold"]
    summary = results.groupby(group_cols, as_index=False).agg(
        mean_AP=("AP", "mean"),
        std_AP=("AP", "std"),
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
        mean_f1=("f1", "mean"),
        n_runs=("AP", "count"),
    )
    summary_path = output_dir / "results_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved aggregated summary to {summary_path}")
    return {"raw": results, "summary": summary}


# ---------------------------------------------------------------------------
# Visualization report
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Batch Synthetic Segmentation Benchmark Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; padding: 2rem;
         background: #fafafa; color: #1a1a1a; }}
  h1 {{ margin-bottom: 0.2rem; }}
  .subtitle {{ color: #555; margin-bottom: 2rem; }}
  .figure-block {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem;
                   margin-bottom: 1.5rem; }}
  .metric-row {{ display: flex; gap: 1.5rem; margin-bottom: 1.5rem; flex-wrap: wrap; }}
  .metric-card {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem 1.5rem;
                  min-width: 180px; }}
  .metric-card .label {{ font-size: 0.85rem; color: #666; }}
  .metric-card .value {{ font-size: 1.6rem; font-weight: 600; }}
  table.summary-table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
  table.summary-table th, table.summary-table td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: right; }}
  table.summary-table th {{ background: #f0f0f0; position: sticky; top: 0; }}
  .table-wrap {{ max-height: 500px; overflow: auto; }}
</style>
</head>
<body>
<h1>Batch Synthetic Segmentation Benchmark</h1>
<p class="subtitle">{subtitle}</p>
{metric_cards}
{figures}
<div class="figure-block">
  <h2>Full aggregated summary</h2>
  <div class="table-wrap">{summary_table}</div>
</div>
</body>
</html>
"""


def _aggregate_per_run(results: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, parameter combo, replicate) with AP@0.5 and
    mAP averaged over the full 0.5-0.95 threshold range -- the base table
    every comparison figure in the report is built from."""
    group_keys = ["model", "radius_mean", "radius_std", "poisson_scale", "gaussian_sigma", "replicate", "seed"]

    ap50 = results.loc[results["iou_threshold"] == 0.5, group_keys + ["AP", "precision", "recall"]].rename(
        columns={"AP": "AP50", "precision": "precision50", "recall": "recall50"}
    )
    map_full = results.groupby(group_keys, as_index=False)["AP"].mean().rename(columns={"AP": "mAP_50_95"})
    return ap50.merge(map_full, on=group_keys)


def _line_by(per_run: pd.DataFrame, x_col: str, x_title: str) -> go.Figure:
    agg = per_run.groupby(["model", x_col], as_index=False).agg(
        mean_mAP=("mAP_50_95", "mean"), std_mAP=("mAP_50_95", "std")
    )
    fig = go.Figure()
    for model_name in sorted(agg["model"].unique()):
        sub = agg[agg["model"] == model_name].sort_values(x_col)
        fig.add_trace(
            go.Scatter(
                x=sub[x_col],
                y=sub["mean_mAP"],
                error_y=dict(array=sub["std_mAP"].fillna(0)),
                mode="lines+markers",
                name=model_name,
            )
        )
    fig.update_layout(
        title=f"Mean AP (IoU 0.5-0.95) vs {x_title}",
        xaxis_title=x_title,
        yaxis_title="Mean AP",
        yaxis_range=[0, 1],
    )
    return fig


def build_report(results: pd.DataFrame, summary: pd.DataFrame, config: SweepConfig) -> str:
    per_run = _aggregate_per_run(results)
    figures: List[go.Figure] = []

    # Headline: overall AP@0.5 and mAP(0.5:0.95), custom vs native.
    headline = per_run.groupby("model", as_index=False).agg(
        mean_AP50=("AP50", "mean"), std_AP50=("AP50", "std"),
        mean_mAP=("mAP_50_95", "mean"), std_mAP=("mAP_50_95", "std"),
    )
    fig_headline = go.Figure()
    fig_headline.add_trace(go.Bar(x=headline["model"], y=headline["mean_AP50"], error_y=dict(array=headline["std_AP50"]), name="AP@0.5"))
    fig_headline.add_trace(go.Bar(x=headline["model"], y=headline["mean_mAP"], error_y=dict(array=headline["std_mAP"]), name="mAP (0.5:0.95)"))
    fig_headline.update_layout(barmode="group", title="Overall performance across the entire grid", yaxis_title="Mean AP", yaxis_range=[0, 1])
    figures.append(fig_headline)

    # COCO-style AP-vs-IoU-threshold curve, averaged across the whole grid.
    thresh_agg = results.groupby(["model", "iou_threshold"], as_index=False)["AP"].mean()
    fig_thresh = go.Figure()
    for model_name in sorted(thresh_agg["model"].unique()):
        sub = thresh_agg[thresh_agg["model"] == model_name].sort_values("iou_threshold")
        fig_thresh.add_trace(go.Scatter(x=sub["iou_threshold"], y=sub["AP"], mode="lines+markers", name=model_name))
    fig_thresh.update_layout(title="Mean AP vs IoU threshold (COCO-style curve)", xaxis_title="IoU threshold", yaxis_title="Mean AP", yaxis_range=[0, 1])
    figures.append(fig_thresh)

    # AP vs each sweep dimension.
    figures.append(_line_by(per_run, "poisson_scale", "Poisson noise scale (higher = less shot noise)"))
    figures.append(_line_by(per_run, "gaussian_sigma", "Gaussian read-noise sigma"))
    figures.append(_line_by(per_run, "radius_mean", "Dot radius mean (px, size distribution)"))

    # 2D noise-sensitivity heatmaps, one per model, sharing a color scale.
    for model_name in sorted(per_run["model"].unique()):
        sub = per_run[per_run["model"] == model_name]
        pivot = sub.groupby(["gaussian_sigma", "poisson_scale"])["AP50"].mean().unstack("poisson_scale")
        fig_heat = px.imshow(
            pivot, labels=dict(x="Poisson noise scale", y="Gaussian noise sigma", color="AP@0.5"),
            title=f"AP@0.5 heatmap -- {model_name} model", color_continuous_scale="Viridis", zmin=0, zmax=1, aspect="auto",
        )
        figures.append(fig_heat)

    # Head-to-head per-image comparison, if both models were run.
    win_rate = None
    if {"custom", "native"}.issubset(set(per_run["model"].unique())):
        pivot_keys = ["radius_mean", "radius_std", "poisson_scale", "gaussian_sigma", "replicate", "seed"]
        pivot = per_run.pivot_table(index=pivot_keys, columns="model", values="mAP_50_95").reset_index()
        pivot_ap50 = per_run.pivot_table(index=pivot_keys, columns="model", values="AP50").reset_index()
        if "custom" in pivot.columns and "native" in pivot.columns:
            win_rate = float((pivot["custom"] > pivot["native"]).mean() * 100)
            fig_h2h = px.scatter(
                pivot_ap50, x="native", y="custom", hover_data=pivot_keys,
                title="Per-image AP@0.5: custom vs native model (above the diagonal = custom wins)",
            )
            fig_h2h.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
            fig_h2h.update_xaxes(range=[0, 1], title="Native model AP@0.5")
            fig_h2h.update_yaxes(range=[0, 1], title="Custom model AP@0.5")
            figures.append(fig_h2h)

    metric_cards = ""
    for _, row in headline.iterrows():
        metric_cards += (
            f'<div class="metric-card"><div class="label">{row["model"]} mAP (0.5:0.95)</div>'
            f'<div class="value">{row["mean_mAP"]:.3f}</div></div>'
        )
    if win_rate is not None:
        metric_cards += (
            f'<div class="metric-card"><div class="label">Custom beats native (per-image mAP)</div>'
            f'<div class="value">{win_rate:.1f}%</div></div>'
        )

    figures_html = ""
    for i, fig in enumerate(figures):
        include_js = "cdn" if i == 0 else False
        figures_html += f'<div class="figure-block">{fig.to_html(full_html=False, include_plotlyjs=include_js)}</div>\n'

    subtitle = (
        f"{len(config.grid_combinations)} parameter combination(s) x {config.n_replicates} replicate(s) x "
        f"{config.n_models} model(s) = {config.total_runs} image/model evaluations."
    )

    return _HTML_TEMPLATE.format(
        subtitle=subtitle,
        metric_cards=f'<div class="metric-row">{metric_cards}</div>',
        figures=figures_html,
        summary_table=summary.to_html(index=False, classes="summary-table", float_format=lambda x: f"{x:.4f}"),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default=None, help="Path to a JSON sweep config; overrides all flags below.")
    p.add_argument("--custom-model", type=str, default=None, help="Path to your trained Cellpose model file.")
    p.add_argument("--native-model", type=str, default="cpsam_v2", help="Built-in Cellpose pretrained model name.")
    p.add_argument("--skip-custom", action="store_true", help="Don't run the custom model (native only).")
    p.add_argument("--skip-native", action="store_true", help="Don't run the native pretrained model (custom only).")
    p.add_argument("--poisson-scales", type=str, default="0.5,1.0,2.0,5.0", help="Comma-separated Poisson noise scales (higher = less shot noise).")
    p.add_argument("--gaussian-sigmas", type=str, default="2,5,10,20", help="Comma-separated Gaussian read-noise sigmas.")
    p.add_argument("--radius-means", type=str, default="3,5,8", help="Comma-separated dot radius means (px) -- size distribution.")
    p.add_argument("--radius-stds", type=str, default="1.0,1.5", help="Comma-separated dot radius stds (px) -- size distribution spread.")
    p.add_argument("--n-replicates", type=int, default=1, help="Random replicate images per parameter combination.")
    p.add_argument("--n-dots", type=int, default=150, help="Dots per synthetic image.")
    p.add_argument("--image-height", type=int, default=512)
    p.add_argument("--image-width", type=int, default=512)
    p.add_argument("--blur-sigma", type=float, default=1.2, help="PSF Gaussian blur sigma (px).")
    p.add_argument("--diameter", type=float, default=None, help="Cellpose diameter override (px); omit for auto-estimate.")
    p.add_argument("--flow-threshold", type=float, default=0.4)
    p.add_argument("--cellprob-threshold", type=float, default=0.0)
    p.add_argument("--gpu", action="store_true", help="Request GPU inference (falls back to CPU if unavailable).")
    p.add_argument("--seed", type=int, default=0, help="Base random seed (each combo/replicate gets a unique derived seed).")
    p.add_argument("--output-dir", type=str, default="batch_benchmark_results")
    p.add_argument("--save-images", action="store_true", help="Also save each synthetic raw image (.tif) and ground truth (.npy).")
    p.add_argument("--print-grid", action="store_true", help="Print the grid size and exit without running anything.")
    return p


def config_from_args(args: argparse.Namespace) -> SweepConfig:
    if args.config:
        return SweepConfig.from_json(args.config)
    return SweepConfig(
        poisson_scales=_parse_float_list(args.poisson_scales),
        gaussian_sigmas=_parse_float_list(args.gaussian_sigmas),
        radius_means=_parse_float_list(args.radius_means),
        radius_stds=_parse_float_list(args.radius_stds),
        n_replicates=args.n_replicates,
        image_height=args.image_height,
        image_width=args.image_width,
        n_dots=args.n_dots,
        blur_sigma=args.blur_sigma,
        custom_model_path=args.custom_model,
        native_model_name=args.native_model,
        run_custom=not args.skip_custom,
        run_native=not args.skip_native,
        diameter=args.diameter,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        gpu=args.gpu,
        seed=args.seed,
        output_dir=args.output_dir,
        save_images=args.save_images,
    )


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = config_from_args(args)

    print(
        f"Grid: {len(config.grid_combinations)} parameter combination(s) x {config.n_replicates} replicate(s) x "
        f"{config.n_models} model(s) = {config.total_runs} total evaluations."
    )
    if args.print_grid:
        return 0

    if config.run_custom and not config.custom_model_path:
        print("ERROR: --custom-model is required unless --skip-custom is set (or models.custom in --config).", file=sys.stderr)
        return 2
    if not config.run_custom and not config.run_native:
        print("ERROR: both custom and native models are disabled -- nothing to do.", file=sys.stderr)
        return 2

    try:
        from cellpose import models as _cellpose_models  # noqa: F401
    except ImportError:
        print("ERROR: cellpose is not installed. Run `pip install cellpose` (see requirements.txt).", file=sys.stderr)
        return 2

    t0 = time.time()
    try:
        results = run_sweep(config)
    except Exception:
        traceback.print_exc()
        return 1

    saved = save_results(results, config)
    report_html = build_report(saved["raw"], saved["summary"], config)
    report_path = Path(config.output_dir) / "report.html"
    report_path.write_text(report_html, encoding="utf-8")
    print(f"Saved comprehensive visual report to {report_path}")
    print(f"Total run time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
