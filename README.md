# Segmentation QC Benchmark

Two companion Streamlit apps plus a standalone batch CLI tool for benchmarking
cell/particle instance-segmentation quality **without a full manual
ground-truth dataset**:

- **`app.py`** — basic benchmark: crop/count spot-checks, morphology QC,
  SNR/contrast validation, synthetic dot simulation, replicate consistency.
- **`app_advanced.py`** — advanced benchmark: consensus/STAPLE ensemble voting,
  perturbation/metamorphic stability, unsupervised boundary & homogeneity
  quality scoring, and a Reverse Classification Accuracy (RCA) proxy.
- **`batch_synthetic_benchmark.py`** — standalone (no Streamlit) CLI tool that
  sweeps synthetic dot generation across noise/size parameters and compares a
  custom-trained Cellpose model against the native pretrained model; see
  [Batch benchmarking](#batch-benchmarking-no-streamlit) below.

## Setup

### macOS / Linux

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py            # or: streamlit run app_advanced.py
```

### Windows

Double-click **`run_app.bat`** (or run it from a command prompt). It creates/
reuses a `.venv` virtual environment, installs `requirements.txt`, then prompts
you to choose which app to launch.

### Upload size limit

`.streamlit/config.toml` raises Streamlit's default 200 MB per-file upload cap
to 4096 MB (4 GB), since a trained Cellpose model file can easily be 1-2+ GB.
This applies to every uploader in both apps. If you need an even higher
ceiling, edit `maxUploadSize` in that file (value is in megabytes).

## Input data

- **Raw fluorescence image** — TIFF (single-channel, multi-channel, or z-stack).
- **Predicted segmentation mask** — TIFF or `.npy`, where `0` = background and
  each unique integer `> 0` is one instance (e.g. a Cellpose prediction).
- **Advanced app only** — optional secondary masks (StarDist, Otsu/Watershed,
  etc.) for the consensus tab.

## `app.py` tabs

1. **Crop & Count Micro-Validation** — pick a small patch (manual bbox or random
   N×N) and verify precision/recall/F1 against a manual count or point-click
   false-positive/false-negative annotation.
2. **Morphological & Intensity Consistency** — per-object `regionprops`
   features, distribution plots, and automated flagging of likely merged
   objects / split fragments based on population statistics. An in-app
   explainer covers what "Potential merge" / "Potential fragment" actually
   mean, and an object inspector lets you pick any flagged label and see
   exactly where it sits in the full image plus a zoomed, mask-overlaid crop.
3. **SNR & Contrast Ratio Validation** — per-object signal-to-background and
   signal-to-noise ratios from a local background ring, how detection
   confidence / area stability change with SNR, and the same object
   inspector to trace any object (defaulting to the lowest-SNR one) back to
   its location on the raw image.
4. **Synthetic Dot Simulation Benchmark** — generate a synthetic image with an
   exact ground truth (Gaussian PSF blur + Poisson + Gaussian noise),
   downloadable as `.npy` or `.tif`. Evaluate it either by uploading an
   externally-predicted mask, or by running Cellpose directly in-app (a
   built-in pretrained model or your own uploaded trained model) for instant
   IoU / AP / detection-rate QC against the exact ground truth.
5. **Downstream Statistical & Replicate Consistency** — compare a second
   uploaded image pair against the primary one using spatial pseudo-replicate
   tiles, coefficient of variation, and Mann-Whitney U / Welch's t-test.

## `app_advanced.py` tabs

1. **Consensus & Pseudo-Ground Truth** — fuse the primary mask with any
   secondary masks (or auto-generated Otsu + Watershed baselines) via
   pixel-wise majority vote or a simplified STAPLE EM label fusion, then
   score the primary mask against the consensus (IoU, Dice, instance AP).
2. **Perturbation & Metamorphic Stability** — sweep Gaussian noise and
   contrast, and apply 90°/180°/270° rotations and flips; measure Jaccard
   stability against the primary mask using a deterministic Otsu "proxy
   detector" (or a real predicted mask, if you upload one from an external
   model run on the perturbed image).
3. **Unsupervised Boundary & Homogeneity Quality** — per-object composite
   confidence score combining boundary edge sharpness (Sobel gradient),
   intra-object intensity homogeneity (CV), and signal-to-surround contrast;
   flags and overlays low-confidence "suspicious" objects.
4. **Reverse Classification Accuracy (RCA) Proxy** — trains a Random
   Forest / Logistic Regression classifier to separate the mask's
   foreground objects from sampled background patches, reporting
   cross-validated ROC-AUC / PR-AUC as a proxy for how cleanly separable
   (and thus how consistent) the segmentation is.

## Batch benchmarking (no Streamlit)

`batch_synthetic_benchmark.py` is a plain command-line tool for larger,
unattended comparisons -- it doesn't start a server or a browser. It sweeps
Tab 4's synthetic dot generator (PSF blur + Poisson + Gaussian noise) across a
grid of noise levels and dot-size distributions, runs **both** a custom
Cellpose model and Cellpose's native pretrained model on every generated
image, evaluates each against the exact synthetic ground truth (IoU / AP /
precision / recall at COCO-style IoU thresholds 0.5-0.95), and writes:

- `results_raw.csv` — one row per (image, model, IoU threshold)
- `results_summary.csv` — the same, aggregated across replicate images
- `report.html` — a self-contained, interactive Plotly report: overall
  custom-vs-native comparison, the AP-vs-IoU-threshold curve, AP vs. each
  swept parameter (Poisson scale, Gaussian sigma, dot radius), a per-model
  AP@0.5 noise-sensitivity heatmap, and a per-image head-to-head scatter

```bash
# Quick sweep via CLI flags
python batch_synthetic_benchmark.py --custom-model path/to/your_model.pth

# Preview the grid size (how many evaluations) without running anything
python batch_synthetic_benchmark.py --custom-model path/to/your_model.pth --print-grid

# Full control via a JSON config (see batch_config_example.json)
python batch_synthetic_benchmark.py --config batch_config_example.json
```

Run `python batch_synthetic_benchmark.py --help` for the full flag list
(noise/size grids, replicate count, image size, Cellpose eval parameters,
`--gpu`, `--save-images` to also dump each synthetic raw/ground-truth pair,
`--skip-custom`/`--skip-native` to run just one model). Model loading reuses
the same GPU-diagnostic logic as the Streamlit apps: it prints the actual
resolved device and warns if `--gpu` was requested but PyTorch couldn't find
a usable GPU.

On Windows, run it through the same venv `run_app.bat` sets up, calling the
venv's `python.exe` directly (as `run_app.bat` itself does) rather than a
bare `python`/`pip` command, to sidestep any broken launcher-stub issues:
`.venv\Scripts\python.exe batch_synthetic_benchmark.py --custom-model
path\to\your_model.pth`.

## Project layout

```
app.py                    Basic benchmark: Streamlit UI and tab wiring
app_advanced.py            Advanced benchmark: Streamlit UI and tab wiring
batch_synthetic_benchmark.py Standalone CLI: noise/size sweep, custom vs native Cellpose, HTML report
batch_config_example.json     Example JSON sweep config for the batch CLI tool
run_app.bat                 Windows setup + launcher (choose which app to run)
modules/
  io_utils.py                Loading TIFF/.npy, display normalization
  morphology.py               regionprops features + outlier flagging
  snr_utils.py                 Local background ring, SBR/SNR, detection confidence
  synthetic.py                  Synthetic image generator + IoU/AP evaluation
  stats_utils.py                 Replicate summary stats + significance testing
  consensus.py                    Majority vote / STAPLE fusion + Otsu-Watershed baseline
  perturbation.py                  Noise/contrast/geometric perturbations + Jaccard stability
  quality_metrics.py                Boundary sharpness, homogeneity, composite confidence score
  rca.py                              Reverse Classification Accuracy proxy (sklearn)
```
