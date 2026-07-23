# Segmentation QC Benchmark

A Streamlit app for benchmarking cell/particle instance-segmentation quality
**without a full manual ground-truth dataset**.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Input data

- **Raw fluorescence image** — TIFF (single-channel, multi-channel, or z-stack).
- **Predicted segmentation mask** — TIFF or `.npy`, where `0` = background and
  each unique integer `> 0` is one instance.

## Tabs

1. **Crop & Count Micro-Validation** — pick a small patch (manual bbox or random
   N×N) and verify precision/recall/F1 against a manual count or point-click
   false-positive/false-negative annotation.
2. **Morphological & Intensity Consistency** — per-object `regionprops`
   features, distribution plots, and automated flagging of likely merged
   objects / split fragments based on population statistics.
3. **SNR & Contrast Ratio Validation** — per-object signal-to-background and
   signal-to-noise ratios from a local background ring, and how detection
   confidence / area stability change with SNR.
4. **Synthetic Dot Simulation Benchmark** — generate a synthetic image with an
   exact ground truth (Gaussian PSF blur + Poisson + Gaussian noise), then
   evaluate an uploaded prediction against it (IoU, AP, detection rate).
5. **Downstream Statistical & Replicate Consistency** — compare a second
   uploaded image pair against the primary one using spatial pseudo-replicate
   tiles, coefficient of variation, and Mann-Whitney U / Welch's t-test.

## Project layout

```
app.py                  Streamlit UI and tab wiring
modules/
  io_utils.py            Loading TIFF/.npy, display normalization
  morphology.py           regionprops features + outlier flagging
  snr_utils.py            Local background ring, SBR/SNR, detection confidence
  synthetic.py             Synthetic image generator + IoU/AP evaluation
  stats_utils.py           Replicate summary stats + significance testing
```
