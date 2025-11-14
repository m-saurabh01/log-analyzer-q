# utils/preprocess.py
"""
Parsing utilities for log files + TF-IDF feature extraction pipeline.
The parser is designed to handle common log patterns:
  YYYY-MM-DD HH:MM:SS [LEVEL] message
If no timestamp/level match, the whole line is treated as a log message (fallback mode).
"""

import re
import pandas as pd
from dateutil import parser as dateparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

# --------------------------------------------------------------------
# Log type labels (used in UI)
# --------------------------------------------------------------------
LOG_TYPE_GENERIC = "GENERIC"
LOG_TYPE_HDFS = "HDFS"
LOG_TYPE_BGL = "BGL"
LOG_TYPE_UNKNOWN = "UNKNOWN"

# --------------------------------------------------------------------
# Regex patterns for different log formats based on your samples
# --------------------------------------------------------------------

# HDFS (covers BOTH your formats):
# Format 1:
#   240926 103015 INFO NameNode startup completed
# Format 2:
#   081109 203615 143 INFO dfs.DataNode$PacketResponder: PacketResponder ...
#
# Logic:
#   date(YYMMDD) time(HHMMSS) [optional pid] LEVEL component[:] message
HDFS_RE = re.compile(
    r"^(?P<date>\d{6})\s+"
    r"(?P<time>\d{6})\s+"
    r"(?:(?P<pid>\d+)\s+)?"           # optional PID
    r"(?P<level>[A-Z]+)\s+"
    r"(?P<component>\S+):?\s+"
    r"(?P<message>.*)$"
)

# BGL format 1:
#   - 1131221850 2005-06-01 10:17:30 R04-M1-N1-C:J12-U01 RAS KERNEL INFO Node 1 is up
BGL1_RE = re.compile(
    r"^[\-A]\s+\d+\s+"
    r"(?P<date>\d{4}-\d{2}-\d{2})\s+"
    r"(?P<time>\d{2}:\d{2}:\d{2})\s+"
    r"\S+\s+RAS\s+\S+\s+"
    r"(?P<level>[A-Z]+)\s+"
    r"(?P<message>.*)$"
)

# BGL short format:
#   1727347815 001 INFO kernel: System boot sequence initiated
BGL_SHORT_RE = re.compile(
    r"^(?P<epoch>\d{10})\s+\S+\s+"
    r"(?P<level>[A-Z]+)\s+"
    r"(?P<component>\S+):\s+"
    r"(?P<message>.*)$"
)

# BGL long composite format:
#   - 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.675872 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction ...
BGL_LONG_RE = re.compile(
    r"^[\-A]\s+\d+\s+\S+\s+\S+\s+"
    r"(?P<ts>\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d+)\s+\S+\s+RAS\s+\S+\s+"
    r"(?P<level>[A-Z]+)\s+"
    r"(?P<message>.*)$"
)

# Generic format (your example):
#   2024-09-26 10:30:15 INFO Application server started on port 8080
GENERIC_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+"
    r"(?P<level>[A-Z]+)\s+"
    r"(?P<message>.*)$"
)

def detect_log_type(lines, sample_size=200):
    """
    Inspect up to `sample_size` lines and decide whether the file looks like:
      - HDFS
      - BGL
      - GENERIC
      - UNKNOWN

    We only count regex matches here; no heavy parsing.
    """
    hdfs_hits = 0
    bgl_hits = 0
    generic_hits = 0

    for i, line in enumerate(lines):
        if i >= sample_size:
            break
        if HDFS_RE.match(line):
            hdfs_hits += 1
        elif BGL1_RE.match(line) or BGL_SHORT_RE.match(line) or BGL_LONG_RE.match(line):
            bgl_hits += 1
        elif GENERIC_RE.match(line):
            generic_hits += 1

    if hdfs_hits > bgl_hits and hdfs_hits > generic_hits and hdfs_hits > 0:
        return LOG_TYPE_HDFS
    if bgl_hits > hdfs_hits and bgl_hits > generic_hits and bgl_hits > 0:
        return LOG_TYPE_BGL
    if generic_hits > 0:
        return LOG_TYPE_GENERIC
    return LOG_TYPE_UNKNOWN

def parse_log_file(fobj):
    """
    Parse uploaded log file into a DataFrame with columns:
        timestamp | log_level | message

    Also auto-detects the log type (HDFS / BGL / GENERIC / UNKNOWN).

    Returns:
        df       : pandas.DataFrame
        log_type : one of LOG_TYPE_* constants
    """
    # Read raw content (Streamlit uploads give a bytes buffer)
    if hasattr(fobj, "read"):
        raw = fobj.read().decode("utf-8", errors="ignore")
    else:
        with open(fobj, "r", encoding="utf-8", errors="ignore") as fh:
            raw = fh.read()

    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    # Decide overall type (for display only)
    log_type = detect_log_type(lines)

    rows = []

    for line in lines:
        # ------------------------------------------------------------------
        # 1) HDFS (both formats)
        # ------------------------------------------------------------------
        m = HDFS_RE.match(line)
        if m:
            date_str = m.group("date")   # YYMMDD
            time_str = m.group("time")   # HHMMSS
            # Assume 20xx. If you really care about 19xx, change this logic.
            ts_str = (
                f"20{date_str[0:2]}-{date_str[2:4]}-{date_str[4:6]} "
                f"{time_str[0:2]}:{time_str[2:4]}:{time_str[4:6]}"
            )
            try:
                ts = dateparser.parse(ts_str)
            except Exception:
                ts = None

            rows.append({
                "timestamp": ts,
                "log_level": m.group("level"),
                "message": m.group("message"),
            })
            continue

        # ------------------------------------------------------------------
        # 2) BGL format 1 (label + event id + date + time ...)
        # ------------------------------------------------------------------
        m = BGL1_RE.match(line)
        if m:
            ts_str = f"{m.group('date')} {m.group('time')}"
            try:
                ts = dateparser.parse(ts_str)
            except Exception:
                ts = None

            rows.append({
                "timestamp": ts,
                "log_level": m.group("level"),
                "message": m.group("message"),
            })
            continue

        # ------------------------------------------------------------------
        # 3) BGL short format (epoch + id + level + component: msg)
        # ------------------------------------------------------------------
        m = BGL_SHORT_RE.match(line)
        if m:
            epoch = int(m.group("epoch"))
            try:
                ts = pd.to_datetime(epoch, unit="s")
            except Exception:
                ts = None

            rows.append({
                "timestamp": ts,
                "log_level": m.group("level"),
                "message": m.group("message"),
            })
            continue

        # ------------------------------------------------------------------
        # 4) BGL long composite format (label + epoch + ... + detailed ts)
        # ------------------------------------------------------------------
        m = BGL_LONG_RE.match(line)
        if m:
            ts_str = m.group("ts")
            # Example: 2005-06-03-15.42.50.675872
            # Replace '-' between date and time with space, '.' with ':'
            # up to seconds; microseconds can stay as-is.
            ts_str_clean = ts_str.replace("-", " ", 1).replace(".", ":", 2)
            try:
                ts = dateparser.parse(ts_str_clean)
            except Exception:
                ts = None

            rows.append({
                "timestamp": ts,
                "log_level": m.group("level"),
                "message": m.group("message"),
            })
            continue

        # ------------------------------------------------------------------
        # 5) Generic format: YYYY-MM-DD HH:MM:SS LEVEL message
        # ------------------------------------------------------------------
        m = GENERIC_RE.match(line)
        if m:
            ts_str = m.group("timestamp")
            try:
                ts = dateparser.parse(ts_str)
            except Exception:
                ts = None

            rows.append({
                "timestamp": ts,
                "log_level": m.group("level"),
                "message": m.group("message"),
            })
            continue

        # ------------------------------------------------------------------
        # 6) Fallbacks: try "timestamp rest_of_line", or just raw message
        # ------------------------------------------------------------------
        parts = line.split(" ", 2)
        if len(parts) >= 3:
            possible_ts = parts[0]
            try:
                ts = dateparser.parse(possible_ts)
                rows.append({
                    "timestamp": ts,
                    "log_level": None,
                    "message": parts[2],
                })
                continue
            except Exception:
                pass

        # Ultimate fallback: entire line as message
        rows.append({
            "timestamp": None,
            "log_level": None,
            "message": line,
        })

    df = pd.DataFrame(rows)

    # Handle timestamps
    if df["timestamp"].isnull().all():
        df["timestamp"] = pd.date_range(
            start=pd.Timestamp.now(), periods=len(df), freq="S"
        )
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["timestamp"] = df["timestamp"].ffill().bfill()


    # Normalize columns
    df["log_level"] = df["log_level"].fillna("UNKNOWN")
    df["message"] = df["message"].astype(str)

    return df, log_type


def build_tfidf_pipeline(messages, max_features=2000, n_components=100):
    """
    Build TF-IDF -> (optional) TruncatedSVD -> StandardScaler.

    IMPORTANT:
    - We first fit TF-IDF to know the actual number of features.
    - Then we choose SVD components <= n_features - 1 to avoid
      ValueError: n_components must be <= n_features.

    Returns:
        pipeline : dict with 'tfidf', 'svd', 'scaler' (svd/scaler may be None)
        X        : numpy array of transformed features
    """
    # 1) Fit TF-IDF first
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X_tfidf = tfidf.fit_transform(messages)
    n_features = X_tfidf.shape[1]

    # Edge case: almost no vocabulary
    if n_features <= 1:
        # Just return TF-IDF as dense array, no SVD/scaling
        X = X_tfidf.toarray()
        pipeline = {
            "tfidf": tfidf,
            "svd": None,
            "scaler": None,
        }
        return pipeline, X

    # 2) Choose a safe number of SVD components
    effective_n_components = min(n_components, n_features - 1)
    if effective_n_components < 1:
        effective_n_components = 1  # ultra-defensive, but practically not hit

    # 3) SVD + scaling
    svd = TruncatedSVD(
        n_components=effective_n_components,
        random_state=42  # FIX: make SVD deterministic
    )
    X_svd = svd.fit_transform(X_tfidf)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_svd)

    # We return a simple dict; you are not using it later anyway.
    pipeline = {
        "tfidf": tfidf,
        "svd": svd,
        "scaler": scaler,
    }

    return pipeline, X_scaled

