"""
Advanced Segmentation QC Benchmark
==================================

A companion Streamlit app that benchmarks segmentation quality using four
advanced, ground-truth-free strategies:

1. Consensus & Pseudo-Ground Truth  -- ensemble voting / STAPLE across masks.
2. Perturbation & Metamorphic Stability -- Jaccard stability under noise/
   contrast/geometric transforms.
3. Unsupervised Boundary & Homogeneity Quality -- per-object composite
   confidence score from edge sharpness, homogeneity, and contrast.
4. Reverse Classification Accuracy (RCA) proxy -- cross-validated
   foreground/background separability of the mask's own predictions.

Input: a raw fluorescence image (TIFF) and a primary predicted instance
mask (TIFF or .npy, typically from Cellpose), 0 = background, unique
int > 0 = one instance each. Optional secondary masks (e.g. StarDist,
Otsu/Watershed) enable the consensus tab's ensemble voting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from skimage.color import label2rgb
from skimage.measure import regionprops

from modules import consensus, io_utils, perturbation, quality_metrics, rca, synthetic

st.set_page_config(page_title="Advanced Segmentation QC Benchmark", page_icon="🧪", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar -- data input & global parameters
# ---------------------------------------------------------------------------
st.sidebar.title("🧪 Advanced Segmentation Benchmark")
st.sidebar.markdown(
    "Benchmark a Cellpose (or other) segmentation using consensus voting, "
    "perturbation stability, intrinsic quality scoring, and an RCA proxy -- "
    "all without a manual ground truth."
)

st.sidebar.header("1. Primary data")
raw_file = st.sidebar.file_uploader("Raw fluorescence image (TIFF)", type=["tif", "tiff"], key="adv_raw_upload")
primary_mask_file = st.sidebar.file_uploader(
    "Primary predicted mask (Cellpose, TIFF or .npy)", type=["tif", "tiff", "npy"], key="adv_primary_upload"
)

st.sidebar.header("2. Optional secondary masks")
secondary_files = st.sidebar.file_uploader(
    "Other models' masks (StarDist, Otsu/Watershed, ...)",
    type=["tif", "tiff", "npy"],
    accept_multiple_files=True,
    key="adv_secondary_upload",
)


def _load_and_reduce(uploaded_file, label_prefix: str, is_label_mask: bool = False) -> np.ndarray:
    """Load an uploaded file and, if it has more than 2 dimensions, let the
    user pick how to collapse it to a single 2D plane."""
    array = io_utils.load_array_from_upload(uploaded_file)
    if array.ndim == 2:
        return array

    role = io_utils.guess_axis_role(array.shape)
    st.sidebar.caption(f"{label_prefix}: shape {array.shape} detected -- select how to flatten to 2D.")

    if is_label_mask:
        method = "channel_last_index" if role == "channel_last" else "first_axis_index"
        axis_len = array.shape[-1] if method == "channel_last_index" else array.shape[0]
        index = st.sidebar.number_input(
            f"{label_prefix} plane/channel index", 0, max(axis_len - 1, 0), 0, key=f"{label_prefix}_idx"
        )
        return io_utils.reduce_to_2d(array, method, int(index))

    options = ["max_projection", "mean_projection", "channel_last_index", "first_axis_index"]
    default_idx = 2 if role == "channel_last" else 0
    method = st.sidebar.selectbox(f"{label_prefix} reduction", options, index=default_idx, key=f"{label_prefix}_method")
    index = 0
    if method in ("channel_last_index", "first_axis_index"):
        axis_len = array.shape[-1] if method == "channel_last_index" else array.shape[0]
        index = st.sidebar.number_input(f"{label_prefix} index", 0, max(axis_len - 1, 0), 0, key=f"{label_prefix}_idx2")
    return io_utils.reduce_to_2d(array, method, int(index))


raw_img, primary_mask = None, None
if raw_file is not None:
    raw_img = _load_and_reduce(raw_file, "Raw image").astype(np.float64)
if primary_mask_file is not None:
    primary_mask = _load_and_reduce(primary_mask_file, "Primary mask", is_label_mask=True).astype(np.int32)

if primary_mask is not None and raw_img is not None and primary_mask.shape != raw_img.shape:
    st.sidebar.error(f"Shape mismatch: raw {raw_img.shape} vs mask {primary_mask.shape}. Downstream tabs may fail.")

st.sidebar.header("3. Perturbation parameters")
noise_sigma_max = st.sidebar.slider("Max Gaussian noise sigma", 1, 100, 30)
n_noise_steps = st.sidebar.slider("Noise sweep steps", 3, 10, 5)
contrast_variation_pct = st.sidebar.slider("Contrast variation range (+/- %)", 5, 50, 20)
n_contrast_steps = st.sidebar.slider("Contrast sweep steps", 3, 10, 5)
rotation_angles = st.sidebar.multiselect("Rotation angles to test (deg)", [90, 180, 270], default=[90, 180, 270])
include_flips = st.sidebar.checkbox("Include horizontal/vertical flips", value=True)

st.sidebar.header("4. Thresholds & tolerances")
iou_threshold = st.sidebar.slider("IoU match threshold (AP / instance matching)", 0.1, 0.9, 0.5, 0.05)
majority_fraction = st.sidebar.slider("Majority-vote agreement fraction", 0.1, 1.0, 0.5, 0.05)
ring_width = st.sidebar.slider("Background ring width (px)", 1, 40, 6, help="Used by Tab 3 and Tab 4's local background estimate.")
ring_gap = st.sidebar.slider("Ring gap from object edge (px)", 0, 15, 2)

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "1 · Consensus & Pseudo-GT",
        "2 · Perturbation Stability",
        "3 · Boundary & Homogeneity QC",
        "4 · RCA Proxy",
    ]
)

# ===========================================================================
# TAB 1 -- Consensus & Pseudo-Ground Truth (Ensemble Voting / STAPLE)
# ===========================================================================
with tab1:
    st.header("Consensus & Pseudo-Ground Truth (Ensemble Voting / STAPLE)")
    st.markdown(
        "Fuse the primary mask with any secondary masks (or classical Otsu/Watershed baselines) into a "
        "consensus 'pseudo-ground-truth', then benchmark the primary mask against it."
    )

    if raw_img is None or primary_mask is None:
        st.info("Upload a raw image and the primary mask in the sidebar to use this tab.")
    else:
        secondary_masks = {}
        for f in secondary_files or []:
            arr = io_utils.load_array_from_upload(f)
            if arr.ndim != 2:
                arr = io_utils.reduce_to_2d(arr, "first_axis_index", 0)
            secondary_masks[f.name] = arr.astype(np.int32)

        voters = {"Primary (Cellpose)": primary_mask, **secondary_masks}
        if len(voters) < 2:
            st.info(
                "Fewer than 2 masks supplied -- auto-generating Otsu + Watershed baselines from the raw image "
                "as additional voters."
            )
            voters.update(consensus.otsu_watershed_baseline(raw_img))

        st.write(f"**Voters used for consensus:** {', '.join(voters.keys())}")

        method = st.radio("Consensus method", ["Majority vote", "STAPLE (EM)"], horizontal=True)
        mask_list = list(voters.values())
        shapes_ok = all(m.shape == mask_list[0].shape for m in mask_list)
        if not shapes_ok:
            st.error("All voter masks must share the same shape as the primary mask.")
        else:
            if method == "Majority vote":
                consensus_mask = consensus.majority_vote_consensus(mask_list, majority_fraction)
            else:
                consensus_mask, _prob_map = consensus.staple_consensus(mask_list)

            disp_raw = io_utils.normalize_for_display(raw_img)
            c1, c2 = st.columns(2)
            with c1:
                overlay_primary = label2rgb(primary_mask, image=disp_raw, bg_label=0, alpha=0.4)
                st.plotly_chart(
                    px.imshow(overlay_primary, title=f"Primary mask ({int(np.count_nonzero(np.unique(primary_mask)))} objects)"),
                    width="stretch",
                )
            with c2:
                overlay_consensus = label2rgb(consensus_mask, image=disp_raw, bg_label=0, alpha=0.4)
                st.plotly_chart(
                    px.imshow(
                        overlay_consensus,
                        title=f"Consensus ({method}, {int(np.count_nonzero(np.unique(consensus_mask)))} objects)",
                    ),
                    width="stretch",
                )

            global_metrics = consensus.binary_overlap_metrics(primary_mask, consensus_mask)
            eval_df = synthetic.evaluate_against_ground_truth(
                consensus_mask, primary_mask, thresholds=np.array([iou_threshold])
            )
            row = eval_df.iloc[0]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Global (binary) IoU", f"{global_metrics['IoU']:.3f}")
            m2.metric("Global Dice", f"{global_metrics['Dice']:.3f}")
            m3.metric(f"Instance AP @ IoU {iou_threshold}", f"{row['AP']:.3f}")
            m4.metric("Detection rate (recall)", f"{row['recall']:.3f}")
            st.caption(
                "'Instance AP' treats the consensus as pseudo-ground-truth and the primary mask as the "
                "prediction being evaluated, using the same TP/(TP+FP+FN) convention as Tab 4 of the basic app."
            )

            if len(voters) >= 2:
                st.subheader("Pairwise voter agreement (binary IoU)")
                names = list(voters.keys())
                agreement = np.zeros((len(names), len(names)))
                for i, name_i in enumerate(names):
                    for j, name_j in enumerate(names):
                        agreement[i, j] = consensus.binary_overlap_metrics(voters[name_i], voters[name_j])["IoU"]
                fig = px.imshow(
                    agreement, x=names, y=names, color_continuous_scale="Viridis", zmin=0, zmax=1,
                    text_auto=".2f", title="Pairwise IoU between all voters",
                )
                st.plotly_chart(fig, width="stretch")

# ===========================================================================
# TAB 2 -- Perturbation & Metamorphic Stability Analysis
# ===========================================================================
with tab2:
    st.header("Perturbation & Metamorphic Stability Analysis")
    st.markdown(
        "Probe how stable the primary mask's foreground is under synthetic perturbations of the raw image."
    )

    if raw_img is None or primary_mask is None:
        st.info("Upload a raw image and the primary mask in the sidebar to use this tab.")
    else:
        st.caption(
            "This app does not re-run the segmentation model (e.g. Cellpose) on perturbed images. By default it "
            "uses a deterministic Otsu-threshold 'proxy detector' as a consistent stand-in to test whether the "
            "foreground signal survives each perturbation. If you've re-run your model externally on a perturbed "
            "image, upload its predicted mask below the charts for a real (non-proxy) comparison."
        )

        primary_binary = primary_mask > 0
        all_jaccards = []

        st.subheader("Noise sweep")
        noise_levels = np.linspace(0, noise_sigma_max, n_noise_steps)
        noise_rows = []
        for sigma in noise_levels:
            perturbed = perturbation.add_gaussian_noise(raw_img, sigma, seed=0)
            proxy = perturbation.otsu_proxy_binary_mask(perturbed)
            j = perturbation.jaccard(proxy, primary_binary)
            noise_rows.append({"sigma": float(sigma), "jaccard": j})
        noise_df = pd.DataFrame(noise_rows)
        all_jaccards.extend(noise_df["jaccard"].dropna().tolist())
        st.plotly_chart(
            px.line(noise_df, x="sigma", y="jaccard", markers=True, title="Stability vs. Gaussian noise sigma", range_y=[0, 1]),
            width="stretch",
        )

        st.subheader("Contrast sweep")
        lo, hi = 1 - contrast_variation_pct / 100, 1 + contrast_variation_pct / 100
        contrast_factors = np.linspace(lo, hi, n_contrast_steps)
        contrast_rows = []
        for factor in contrast_factors:
            perturbed = perturbation.adjust_contrast(raw_img, factor)
            proxy = perturbation.otsu_proxy_binary_mask(perturbed)
            j = perturbation.jaccard(proxy, primary_binary)
            contrast_rows.append({"contrast_factor": float(factor), "jaccard": j})
        contrast_df = pd.DataFrame(contrast_rows)
        all_jaccards.extend(contrast_df["jaccard"].dropna().tolist())
        st.plotly_chart(
            px.line(contrast_df, x="contrast_factor", y="jaccard", markers=True, title="Stability vs. contrast factor", range_y=[0, 1]),
            width="stretch",
        )

        st.subheader("Geometric transforms (rotation / flip)")
        geo_rows = []
        for angle in rotation_angles:
            k = int(angle // 90)
            rot_raw = perturbation.rotate90(raw_img, k)
            rot_binary = perturbation.rotate90(primary_binary, k)
            proxy = perturbation.otsu_proxy_binary_mask(rot_raw)
            geo_rows.append({"transform": f"Rotate {angle} deg", "jaccard": perturbation.jaccard(proxy, rot_binary)})
        if include_flips:
            for axis, name in ((0, "Flip vertical"), (1, "Flip horizontal")):
                flip_raw = perturbation.flip(raw_img, axis)
                flip_binary = perturbation.flip(primary_binary, axis)
                proxy = perturbation.otsu_proxy_binary_mask(flip_raw)
                geo_rows.append({"transform": name, "jaccard": perturbation.jaccard(proxy, flip_binary)})
        if geo_rows:
            geo_df = pd.DataFrame(geo_rows)
            all_jaccards.extend(geo_df["jaccard"].dropna().tolist())
            st.plotly_chart(
                px.bar(geo_df, x="transform", y="jaccard", title="Stability under geometric transforms", range_y=[0, 1]),
                width="stretch",
            )
        else:
            st.info("Select at least one rotation angle or enable flips in the sidebar.")

        if all_jaccards:
            st.metric("Overall Stability Index (mean Jaccard across all perturbations)", f"{np.mean(all_jaccards):.3f}")

        with st.expander("Optional: compare against real predictions on perturbed images"):
            st.caption(
                "If you ran Cellpose (or another model) on a perturbed version of the raw image outside this app, "
                "upload its predicted mask here for a direct (non-proxy) Jaccard comparison."
            )
            transform_choice = st.selectbox(
                "Which perturbation does this predicted mask correspond to?",
                ["Gaussian noise", "Contrast adjustment", "Rotate 90", "Rotate 180", "Rotate 270", "Flip vertical", "Flip horizontal"],
            )
            real_pred_file = st.file_uploader("Predicted mask on the perturbed image", type=["tif", "tiff", "npy"], key="adv_perturb_pred")
            if real_pred_file is not None:
                pred_mask = io_utils.load_array_from_upload(real_pred_file)
                if pred_mask.ndim != 2:
                    pred_mask = io_utils.reduce_to_2d(pred_mask, "first_axis_index", 0)

                if transform_choice == "Gaussian noise" or transform_choice == "Contrast adjustment":
                    reference_binary = primary_binary
                elif transform_choice.startswith("Rotate"):
                    k = int(transform_choice.split()[1]) // 90
                    reference_binary = perturbation.rotate90(primary_binary, k)
                else:
                    axis = 0 if transform_choice == "Flip vertical" else 1
                    reference_binary = perturbation.flip(primary_binary, axis)

                if pred_mask.shape != reference_binary.shape:
                    st.error(f"Predicted mask shape {pred_mask.shape} does not match expected shape {reference_binary.shape}.")
                else:
                    real_j = perturbation.jaccard(pred_mask, reference_binary)
                    st.metric(f"Real Jaccard -- {transform_choice}", f"{real_j:.3f}")

# ===========================================================================
# TAB 3 -- Unsupervised Boundary & Homogeneity Quality Metrics
# ===========================================================================
with tab3:
    st.header("Unsupervised Boundary & Homogeneity Quality Metrics")
    st.markdown(
        "Score each object using only intrinsic properties of the raw image and mask boundary -- no ground "
        "truth required."
    )

    if raw_img is None or primary_mask is None:
        st.info("Upload a raw image and the primary mask in the sidebar to use this tab.")
    else:
        st.subheader("Composite score weights")
        w1, w2, w3 = st.columns(3)
        with w1:
            weight_sharpness = st.slider("Weight: boundary sharpness", 0.0, 3.0, 1.0, 0.1)
        with w2:
            weight_homogeneity = st.slider("Weight: intra-object homogeneity", 0.0, 3.0, 1.0, 0.1)
        with w3:
            weight_contrast = st.slider("Weight: signal-to-surround contrast", 0.0, 3.0, 1.0, 0.1)

        sensitivity_k = st.slider(
            "Detection sensitivity k (x background sigma)", 0.5, 6.0, 2.0, 0.1,
            help="Passed through to the local signal-to-background computation.",
        )

        conf_df = quality_metrics.compute_composite_confidence(
            primary_mask, raw_img, ring_width, ring_gap, sensitivity_k,
            weights={"sharpness": weight_sharpness, "homogeneity": weight_homogeneity, "contrast": weight_contrast},
        )

        if conf_df.empty:
            st.warning("No labeled objects found in the mask.")
        else:
            suspicious_pct = st.slider("Flag objects at or below this confidence percentile as suspicious", 1, 50, 10)
            conf_df = quality_metrics.flag_low_confidence(conf_df, suspicious_pct)

            st.subheader("Per-object composite confidence table")
            st.dataframe(conf_df, width="stretch")

            st.plotly_chart(
                px.histogram(
                    conf_df, x="confidence_score", nbins=30, color="suspicious",
                    title="Composite Segmentation Confidence Score distribution",
                ),
                width="stretch",
            )

            st.subheader("Suspicious objects highlighted on raw image")
            centroids = {p.label: p.centroid for p in regionprops(primary_mask.astype(np.int32))}
            conf_df["row"] = conf_df["label"].map(lambda lbl: centroids[lbl][0])
            conf_df["col"] = conf_df["label"].map(lambda lbl: centroids[lbl][1])

            fig = px.imshow(io_utils.normalize_for_display(raw_img), color_continuous_scale="gray", title="Object confidence overlay")
            for is_suspicious, color, name in [(False, "lime", "Confident"), (True, "red", "Suspicious")]:
                subset = conf_df[conf_df["suspicious"] == is_suspicious]
                fig.add_trace(
                    go.Scatter(
                        x=subset["col"], y=subset["row"], mode="markers",
                        marker=dict(size=9, color=color, symbol="circle-open", line=dict(width=2)),
                        text=[f"label {l}, score {s:.2f}" for l, s in zip(subset["label"], subset["confidence_score"])],
                        name=name,
                    )
                )
            st.plotly_chart(fig, width="stretch")

            suspicious_table = conf_df[conf_df["suspicious"]].sort_values("confidence_score")
            st.subheader("Suspicious object table")
            st.dataframe(suspicious_table.drop(columns=["row", "col"]), width="stretch")
            if not suspicious_table.empty:
                st.download_button(
                    "Download suspicious objects (CSV)",
                    suspicious_table.drop(columns=["row", "col"]).to_csv(index=False),
                    "suspicious_confidence_objects.csv",
                    "text/csv",
                )

# ===========================================================================
# TAB 4 -- Reverse Classification Accuracy (RCA) Proxy
# ===========================================================================
with tab4:
    st.header("Reverse Classification Accuracy (RCA) Proxy")
    st.markdown(
        "Train a lightweight classifier to separate the primary mask's foreground objects from sampled "
        "background patches. High cross-validated separability is a statistical proxy for how cleanly "
        "consistent the mask's predictions are."
    )

    if raw_img is None or primary_mask is None:
        st.info("Upload a raw image and the primary mask in the sidebar to use this tab.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            classifier_name = st.radio("Classifier", ["Random Forest", "Logistic Regression"])
        with c2:
            negative_ratio = st.slider("Background : Foreground sample ratio", 0.5, 3.0, 1.0, 0.1)
            n_splits = st.slider("Cross-validation folds", 3, 10, 5)
        with c3:
            rca_seed = st.number_input("Random seed", 0, 999999, 0)

        if st.button("Run RCA cross-validation"):
            with st.spinner("Building pseudo-labeled dataset and cross-validating..."):
                data = rca.build_training_set(primary_mask, raw_img, ring_width, ring_gap, negative_ratio, int(rca_seed))
                result = rca.run_rca_cross_validation(data, classifier_name, n_splits, int(rca_seed))
            st.session_state["rca_result"] = result
            st.session_state["rca_data"] = data

        if "rca_result" in st.session_state:
            result = st.session_state["rca_result"]
            data = st.session_state["rca_data"]

            st.caption(f"Trained on {result['n_positive']} foreground objects vs. {result['n_negative']} sampled background patches.")
            m1, m2 = st.columns(2)
            m1.metric("Cross-validated ROC-AUC", f"{result['roc_auc']:.3f}")
            m2.metric("Cross-validated PR-AUC", f"{result['pr_auc']:.3f}")

            col1, col2 = st.columns(2)
            with col1:
                roc_fig = go.Figure()
                roc_fig.add_trace(go.Scatter(x=result["fpr"], y=result["tpr"], mode="lines", name="ROC"))
                roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Chance"))
                roc_fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                st.plotly_chart(roc_fig, width="stretch")
            with col2:
                pr_fig = go.Figure()
                pr_fig.add_trace(go.Scatter(x=result["recall"], y=result["precision"], mode="lines", name="PR"))
                pr_fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
                st.plotly_chart(pr_fig, width="stretch")

            st.subheader("Feature importance")
            st.plotly_chart(
                px.bar(result["feature_importance"], x="feature", y="importance", title=f"{classifier_name} feature importance"),
                width="stretch",
            )

            st.subheader("Feature distributions (foreground vs. background)")
            feature_to_plot = st.selectbox("Feature", rca.FEATURE_COLUMNS)
            st.plotly_chart(
                px.histogram(
                    data, x=feature_to_plot, color="class", barmode="overlay", nbins=40,
                    color_discrete_map={0: "gray", 1: "royalblue"},
                    title=f"{feature_to_plot} distribution (0 = background, 1 = foreground)",
                ),
                width="stretch",
            )

            st.caption(
                "High ROC-AUC/PR-AUC indicate the primary mask's foreground objects are cleanly, consistently "
                "separable from background using simple intensity/shape/gradient features -- a statistical proxy "
                "for segmentation quality when no manual ground truth exists."
            )
