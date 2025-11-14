# app.py
"""
Streamlit app for log upload, parsing, anomaly detection and visualization.

Features:
- Dashboard-style home page
- Top navigation bar with links to modules:
    Dashboard, Upload files, Run detection, Anomaly Explorer, Model diagnostics
- Auto-detection of log type (HDFS / BGL / GENERIC / UNKNOWN) via parse_log_file
- Two models: IsolationForest and Autoencoder
- Global metrics (since app start): total logs, anomalies, files, runs
- Per-file metrics on Dashboard
- Charts: anomalies over time, by log level
- Score / reconstruction error distribution only in Model diagnostics
"""

import io

import streamlit as st
import pandas as pd

from utils.preprocess import parse_log_file, build_tfidf_pipeline
from utils.models import IsolationForestDetector, AutoencoderDetector
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np

from utils.visuals import (
    plot_anomaly_timeline,
    plot_anomaly_counts,
    plot_score_distribution,
)

import streamlit as st
# ... your other imports ...

def inject_custom_css():
    st.markdown(
        """
        <style>
        /* ---------- Global layout ---------- */
        .stApp {
            background: #f3f4f6; /* light gray */
            color: #111827;     /* near-black */
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
        }

        [data-testid="stHeader"] {
            background: #f3f4f6;
        }

        h1, h2, h3 {
            font-weight: 600;
            letter-spacing: 0.01em;
            color: #111827;
        }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid rgba(0,0,0,0.08);
        }

        section[data-testid="stSidebar"] label {
            font-weight: 500;
            color: #111827 !important;
        }

        /* ---------- Nav buttons on top ---------- */
        .stButton > button {
            border-radius: 8px !important;
            padding: 0.40rem 1.1rem !important;
            border: 1px solid #d5d9e0 !important;
            background: #ffffff !important;
            color: #111827 !important;
            font-size: 0.92rem !important;
            font-weight: 500 !important;
            transition: all 0.18s ease-in-out !important;
        }

        .stButton > button:hover {
            background: #e8f2ff !important;
            border-color: #3b82f6 !important;
        }

        .stButton > button:disabled {
            opacity: 0.4 !important;
            cursor: not-allowed;
        }

        /* ---------- Metric cards ---------- */
        [data-testid="stMetric"] {
            background: #ffffff;
            border-radius: 10px;
            padding: 0.75rem 0.9rem;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 7px rgba(0,0,0,0.05);
        }

        [data-testid="stMetric"] label,
        [data-testid="stMetric"] span {
            color: #111827 !important;
        }


        /* ---------- Dataframes / tables ---------- */
        .stDataFrame {
            border-radius: 6px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 7px rgba(0,0,0,0.05);
            background: white;
        }

        /* ---------- File uploader ---------- */
        [data-testid="stFileUploader"] > div:first-child {
            border-radius: 8px;
            border: 2px dashed #cbd5e1;
            background: #ffffff;
        }

        /* ---------- Plot containers ---------- */
        div[data-testid="stPlotlyChart"] {
            background: #ffffff;
            border-radius: 6px;
            padding: 0.2rem;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 7px rgba(0,0,0,0.05);
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Streamlit basic config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AI Log Analyzer", layout="wide")
inject_custom_css()
st.title("AI-Powered Log Analyzer")

# -----------------------------------------------------------------------------
# Session state initialization
# -----------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state["df"] = None
if "log_type" not in st.session_state:
    st.session_state["log_type"] = None
if "df_result" not in st.session_state:
    st.session_state["df_result"] = None
if "model_used" not in st.session_state:
    st.session_state["model_used"] = None
if "contamination_used" not in st.session_state:
    st.session_state["contamination_used"] = None

# Global counters
if "global_logs_count" not in st.session_state:
    st.session_state["global_logs_count"] = 0
if "global_anomaly_count" not in st.session_state:
    st.session_state["global_anomaly_count"] = 0
if "global_files_uploaded" not in st.session_state:
    st.session_state["global_files_uploaded"] = 0
if "global_detection_runs" not in st.session_state:
    st.session_state["global_detection_runs"] = 0
if "last_upload_signature" not in st.session_state:
    st.session_state["last_upload_signature"] = None

# Which view is active
if "view" not in st.session_state:
    st.session_state["view"] = "Dashboard"


def set_view(name: str):
    """Change current view (used by top navbar buttons)."""
    st.session_state["view"] = name


view = st.session_state["view"]
df = st.session_state["df"]
log_type = st.session_state["log_type"]
df_result = st.session_state["df_result"]
model_used = st.session_state["model_used"]

# -----------------------------------------------------------------------------
# Sidebar: model configuration + Run Detection button
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Model settings")

    model_choice = st.selectbox(
        "Model",
        ["IsolationForest", "Autoencoder"],
        index=0,
        help="IsolationForest is fast and tree-based; Autoencoder is neural-network based.",
    )

    contamination = st.slider(
        "Contamination (expected anomaly fraction)",
        min_value=0.001,
        max_value=0.20,
        value=0.05,
        step=0.005,
        help="Approximate fraction of anomalies in the data.",
    )

    if model_choice == "Autoencoder":
        with st.expander("Autoencoder hyperparameters (advanced)", expanded=False):
            encoding_dim = st.slider(
                "Encoding dimension",
                min_value=8,
                max_value=128,
                value=32,
                step=8,
            )
            ae_epochs = st.slider(
                "Training epochs",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
            )
            ae_batch_size = st.selectbox(
                "Batch size",
                [16, 32, 64, 128],
                index=1,
            )
    else:
        encoding_dim = 32
        ae_epochs = 20
        ae_batch_size = 32

    st.markdown("---")
    run_button = st.button("Run detection", type="primary")

# -----------------------------------------------------------------------------
# Compute has_data BEFORE drawing navbar
# -----------------------------------------------------------------------------
has_data = st.session_state["df"] is not None

# -----------------------------------------------------------------------------
# Top navigation bar (always at top, uses up-to-date has_data)
# -----------------------------------------------------------------------------
nav_cols = st.columns(5)
with nav_cols[0]:
    st.button(
        "Dashboard",
        type="secondary",
        on_click=set_view,
        args=("Dashboard",),
    )
with nav_cols[1]:
    st.button(
        "Upload files",
        type="secondary",
        on_click=set_view,
        args=("Upload files",),
    )
with nav_cols[2]:
    st.button(
        "Run detection",
        type="secondary",
        disabled=not has_data,
        on_click=set_view,
        args=("Run detection",),
    )
with nav_cols[3]:
    st.button(
        "Anomaly Explorer",
        type="secondary",
        disabled=not has_data,
        on_click=set_view,
        args=("Anomaly Explorer",),
    )
with nav_cols[4]:
    st.button(
        "Model diagnostics",
        type="secondary",
        disabled=not has_data,
        on_click=set_view,
        args=("Model diagnostics",),
    )

st.markdown("---")

# Refresh locals in case view changed this run
view = st.session_state["view"]
df = st.session_state["df"]
log_type = st.session_state["log_type"]
df_result = st.session_state["df_result"]
model_used = st.session_state["model_used"]
has_data = df is not None



# After possible upload, refresh references
df = st.session_state["df"]
log_type = st.session_state["log_type"]
df_result = st.session_state["df_result"]
model_used = st.session_state["model_used"]
has_data = df is not None

# -----------------------------------------------------------------------------
# Helper: run anomaly detection and store results
# -----------------------------------------------------------------------------
def run_detection_if_needed():
    """
    Triggered when Run detection button is pressed.
    - Builds TF-IDF + SVD features
    - Trains chosen model
    - Stores df_result + updates global metrics
    """
    if df is None:
        st.warning("Please upload a log file before running detection.")
        return

    pipeline, X = build_tfidf_pipeline(df["message"].astype(str))

    if model_choice == "IsolationForest":
        st.info("Running IsolationForest anomaly detection...")
        detector = IsolationForestDetector(
            contamination=contamination,
            random_state=42,
        )
        preds, scores = detector.fit_predict(X)
    else:
        st.info("Training Autoencoder and computing reconstruction errors...")
        detector = AutoencoderDetector(
            encoding_dim=encoding_dim,
            epochs=ae_epochs,
            batch_size=ae_batch_size,
            contamination=contamination,
            random_state=42,
        )
        preds, scores = detector.fit_predict(X)

    df_result_local = df.copy()
    df_result_local["anomaly"] = preds
    df_result_local["anomaly_score"] = scores

    st.session_state["df_result"] = df_result_local
    st.session_state["model_used"] = model_choice
    st.session_state["contamination_used"] = contamination

    total_anom = int(df_result_local["anomaly"].sum())
    st.session_state["global_detection_runs"] += 1
    st.session_state["global_anomaly_count"] += total_anom


if run_button and df is not None:
    run_detection_if_needed()

df_result = st.session_state["df_result"]
model_used = st.session_state["model_used"]

# -----------------------------------------------------------------------------
# Metrics for current file
# -----------------------------------------------------------------------------
def compute_current_metrics(df_logs: pd.DataFrame, df_res: pd.DataFrame | None):
    """Compute simple metrics for the current log file."""
    metrics = {}

    if df_logs is None or df_logs.empty:
        return metrics

    metrics["total_logs"] = len(df_logs)
    metrics["log_type"] = log_type or "UNKNOWN"

    metrics["time_start"] = df_logs["timestamp"].min()
    metrics["time_end"] = df_logs["timestamp"].max()

    level_counts = df_logs["log_level"].value_counts()
    metrics["info_count"] = int(level_counts.get("INFO", 0))
    metrics["warn_count"] = int(level_counts.get("WARN", 0))
    metrics["error_count"] = int(level_counts.get("ERROR", 0))

    if df_res is not None and "anomaly" in df_res.columns:
        total_anom = int(df_res["anomaly"].sum())
        metrics["total_anomalies"] = total_anom
        metrics["anomaly_rate"] = total_anom / len(df_logs) if len(df_logs) > 0 else 0.0
    else:
        metrics["total_anomalies"] = None
        metrics["anomaly_rate"] = None

    return metrics


current_metrics = compute_current_metrics(df, df_result)

# -----------------------------------------------------------------------------
# VIEW: Dashboard
# -----------------------------------------------------------------------------
if view == "Dashboard":
    st.subheader("Dashboard overview")

    # Global metrics
    g_logs = st.session_state["global_logs_count"]
    g_anoms = st.session_state["global_anomaly_count"]
    g_files = st.session_state["global_files_uploaded"]
    g_runs = st.session_state["global_detection_runs"]

    st.markdown("#### Global metrics (since app started)")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Total log entries processed", value=g_logs)
    g2.metric("Total anomalies detected", value=g_anoms)
    g3.metric("Files uploaded", value=g_files)
    g4.metric("Detection runs", value=g_runs)

    st.markdown("---")

    # Current file metrics
    st.markdown("#### Current file metrics")
    if df is None:
        st.info("Upload a log file and run detection to see detailed metrics.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current log entries", current_metrics.get("total_logs", 0))
        c2.metric(
            "Current anomalies",
            current_metrics.get("total_anomalies", 0)
            if current_metrics.get("total_anomalies") is not None
            else "—",
        )
        c3.metric(
            "Current anomaly rate",
            f"{current_metrics['anomaly_rate']*100:.2f} %"
            if current_metrics.get("anomaly_rate") is not None
            else "—",
        )
        c4.metric(
            "Log format",
            current_metrics.get("log_type", "UNKNOWN"),
        )

        lc1, lc2, lc3 = st.columns(3)
        lc1.metric("INFO entries", current_metrics.get("info_count", 0))
        lc2.metric("WARN entries", current_metrics.get("warn_count", 0))
        lc3.metric("ERROR entries", current_metrics.get("error_count", 0))

        if current_metrics.get("time_start") is not None:
            st.caption(
                f"Time range (current file): {current_metrics['time_start']} → {current_metrics['time_end']}"
            )

    st.markdown("---")
    st.markdown("### How to use this application optimally")

    st.write(
        """
        **Prepare your logs**

        - Export HDFS / BGL / generic application logs into `.log`, `.txt` or `.csv` files.
        - Each line should represent a log event (timestamp, level, message).
        - For very large logs, consider sampling to keep things responsive.

        **Upload and verify parsing**

        - Use the **Upload files** link in the titlebar and the uploader on that page.
        - The app will auto-detect the log format (HDFS / BGL / GENERIC).
        - Check the parsed preview for:
          - Correct timestamps,
          - Correct `log_level` (INFO/WARN/ERROR/...),
          - Reasonable `message` content.

        **Configure the model**

        - In the sidebar, choose:
          - **IsolationForest** for a fast, solid baseline,
          - **Autoencoder** for more expressive, neural anomaly scores.
        - Tune the **contamination** slider:
          - Lower → stricter (fewer anomalies),
          - Higher → looser (more anomalies).

        **Run and interpret**

        - Hit **Run detection** in the sidebar.
        - Then use **Anomaly Explorer** to:
          - Inspect individual anomalies,
          - View anomaly counts over time,
          - See anomaly counts by log level.

        - Use **Model diagnostics** for a focused view of score / reconstruction error distributions.

        **Iterate**

        - Adjust contamination / model choice when you see too many or too few anomalies.
        - Compare IsolationForest vs Autoencoder behavior on the same log set.
        """
    )

# -----------------------------------------------------------------------------
# VIEW: Upload files
# -----------------------------------------------------------------------------
elif view == "Upload files":
    st.subheader("Upload log file")

    uploaded = st.file_uploader(
        "Upload a log file (.log / .txt / .csv)",
        type=["log", "txt", "csv"],
    )

    if uploaded is not None:
        raw_bytes = uploaded.read()

        try:
            df_new, log_type_new = parse_log_file(io.BytesIO(raw_bytes))
        except Exception as e:
            st.error(f"Failed to parse log file: {e}")
            st.stop()

        sig = (uploaded.name, uploaded.size)
        is_new_file = st.session_state["last_upload_signature"] != sig

        if is_new_file:
            st.session_state["global_logs_count"] += len(df_new)
            st.session_state["global_files_uploaded"] += 1
            st.session_state["last_upload_signature"] = sig

        st.session_state["df"] = df_new
        st.session_state["log_type"] = log_type_new

        # Just show a small preview, no rerun
        st.caption(f"Detected log format: **{log_type_new}**")
        st.dataframe(df_new.head(10), use_container_width=True)

    else:
        if st.session_state["df"] is None:
            st.info("Upload a log file to start. The app will parse and detect log format automatically.")
        else:
            # Show existing data (top 10 only)
            df = st.session_state["df"]
            log_type = st.session_state["log_type"]
            st.caption(f"Currently loaded data (log format: **{log_type}**, {len(df)} entries)")
            st.dataframe(df.head(10), use_container_width=True)


# -----------------------------------------------------------------------------
# VIEW: Run detection
# -----------------------------------------------------------------------------
elif view == "Run detection":
    st.subheader("Run anomaly detection")

    if df is None:
        st.warning("Upload a log file first. The 'Run detection' link is disabled until a file is uploaded.")
    else:
        st.write(
            """
            Use the **Run detection** button in the sidebar to execute the selected model
            on the currently uploaded log file.
            """
        )
        if df_result is None:
            st.info("No detection has been run yet in this session.")
        else:
            total_anom = int(df_result["anomaly"].sum())
            st.success(f"Latest run ({model_used}) detected {total_anom} anomalies.")
            st.dataframe(
                df_result.head(50)[
                    ["timestamp", "log_level", "message", "anomaly", "anomaly_score"]
                ],
                use_container_width=True,
            )

# -----------------------------------------------------------------------------
# VIEW: Anomaly Explorer
# -----------------------------------------------------------------------------
elif view == "Anomaly Explorer":
    st.subheader("Anomaly explorer")

    if df_result is None:
        st.info("Run anomaly detection first to explore anomalies.")
    else:
        st.caption(
            f"Model used: {model_used} | "
            f"Contamination: {st.session_state['contamination_used']}"
        )

        # ------------------------------------------------------------------
        # METRICS: precision / recall / F1 / accuracy (+ counts)
        #
        # Case 1: if you have a true label column:
        #   - use df_result['label'] as ground truth (0 = normal, 1 = anomaly)
        # Case 2: if no label column:
        #   - derive pseudo-labels from log_level:
        #       ERROR / FATAL / CRITICAL -> 1 (anomaly)
        #       everything else         -> 0 (normal)
        # ------------------------------------------------------------------
        st.markdown("#### Detection metrics")

        if "label" in df_result.columns:
            y_true = df_result["label"].astype(int)
            label_source = "ground_truth"
        else:
            # heuristic labels based on log_level
            error_like = df_result["log_level"].isin(["ERROR", "FATAL", "CRITICAL"])
            y_true = error_like.astype(int)
            label_source = "heuristic"

        try:
            y_pred = df_result["anomaly"].astype(int)

            # core metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, pos_label=1, average="binary"
            )

            # confusion matrix for accuracy and counts
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            total = tp + tn + fp + fn
            accuracy = (tp + tn) / total if total > 0 else np.nan

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Precision (anomaly)", f"{precision:.3f}")
            m2.metric("Recall (anomaly)", f"{recall:.3f}")
            m3.metric("F1-score (anomaly)", f"{f1:.3f}")
            m4.metric("Accuracy", f"{accuracy:.3f}")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("True anomalies (y=1)", int(tp + fn))
            c2.metric("True normals (y=0)", int(tn + fp))
            c3.metric("Predicted anomalies", int(tp + fp))
            c4.metric("Predicted normals", int(tn + fn))

            if label_source == "ground_truth":
                st.caption(
                    "Metrics computed against **ground-truth labels** in the `label` column "
                    "(1 = anomaly, 0 = normal)."
                )
            else:
                st.caption(
                    "Metrics computed against **heuristic labels**: "
                    "`ERROR` / `FATAL` / `CRITICAL` treated as anomalies (1), "
                    "all other levels treated as normal (0)."
                )

        except Exception as e:
            st.warning(f"Could not compute metrics: {e}")

        st.markdown("---")

        # ------------------------------------------------------------------
        # Anomalies table
        # ------------------------------------------------------------------
        anomalies = (
            df_result[df_result["anomaly"] == 1]
            .sort_values("anomaly_score", ascending=False)
        )

        st.markdown("#### Anomalies table")
        st.dataframe(
            anomalies[["timestamp", "log_level", "message", "anomaly_score"]].head(500),
            width="stretch",
        )

        # ------------------------------------------------------------------
        # Charts
        # ------------------------------------------------------------------
        st.markdown("#### Charts")
        tab1, tab2 = st.tabs(["Anomalies over time", "By log level"])

        with tab1:
            st.plotly_chart(
                plot_anomaly_timeline(df_result),
                use_container_width=True,
            )

        with tab2:
            st.plotly_chart(
                plot_anomaly_counts(df_result),
                use_container_width=True,
            )

        # ------------------------------------------------------------------
        # Download
        # ------------------------------------------------------------------
        csv = df_result.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download full results (CSV)",
            data=csv,
            file_name="anomaly_results.csv",
            mime="text/csv",
        )

# -----------------------------------------------------------------------------
# VIEW: Model diagnostics
# -----------------------------------------------------------------------------
elif view == "Model diagnostics":
    st.subheader("Model diagnostics")

    if df_result is None:
        st.info("Run anomaly detection first to see diagnostics.")
    else:
        st.caption(f"Diagnostics for model: {model_used}")

        st.markdown("#### Score / reconstruction error distribution")
        title = (
            "Reconstruction error distribution (Autoencoder)"
            if model_used == "Autoencoder"
            else "Anomaly score distribution (IsolationForest)"
        )
        st.plotly_chart(
            plot_score_distribution(
                df_result,
                score_col="anomaly_score",
                title=title,
            ),
            use_container_width=True,
        )

        if model_used == "Autoencoder":
            st.write(
                """
                Higher reconstruction error means the Autoencoder struggled to reconstruct that log entry, 
                suggesting it's unusual compared to the overall pattern.
                """
            )
        else:
            st.write(
                """
                IsolationForest assigns more extreme scores to points that are isolated early in the trees.
                Here, higher scores correspond to more anomalous entries after score inversion.
                """
            )

# -----------------------------------------------------------------------------
# Fallback: About
# -----------------------------------------------------------------------------
else:
    st.subheader("About this app")
    st.write(
        """
        This AI Log Analyzer is designed to work with:
        - HDFS logs (multiple formats),
        - BGL / BlueGene-style logs,
        - generic application logs (timestamp + level + message).
        """
    )

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption("AI Log Analyzer · Built with Streamlit · Session stats reset when the server restarts.")
