"""Sidecar configuration: model pins, front-end constants, class-order map.

Every model is pinned by URL + size + sha256. Pins were captured with the
hash-and-reconfirm methodology (download, hash, re-download from an
independent network path, confirm match) on 2026-09-12. Weights are NEVER
committed (CC BY-NC-SA) — they are downloaded at first start into the data
dir and verified against these pins.

Model URLs use the per-head SUBDIR paths — flat paths 404. Never trust the
.json `link` fields (they are stale .pb paths; the embedding link points at
music-style-classification while the actual ONNX lives under
feature-extractors).
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
HOST = "0.0.0.0"
PORT = 8099

# Shared-secret auth. Empty token → /analyze answers 401 (logged hint once);
# /health stays open (auto-detect + config-flow status line need it).
API_TOKEN = os.environ.get("AXBPM_API_TOKEN", "")

# /analyze input caps (OOM + latency protection in a ~1 GB container).
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB multipart cap → 413
MAX_ANALYZE_SECONDS = float(os.environ.get("AXBPM_MAX_ANALYZE_SECONDS", "60"))

# onnxruntime session options — the sidecar shares host CPU with HA core;
# unbounded thread use can stutter core. These are the only lever.
INTRA_OP_THREADS = int(os.environ.get("AXBPM_INTRA_OP_THREADS", "2"))
INTER_OP_THREADS = 1

# Single inference worker, queue depth 1: busy → 503 immediately.
ANALYZE_TIMEOUT_SECONDS = 30.0

# Model download retry/backoff.
DOWNLOAD_RETRIES = 5
DOWNLOAD_BACKOFF_SECONDS = 3.0

# Data dir for model weights (add-on maps /data).
DATA_DIR = Path(os.environ.get("AXBPM_DATA_DIR", "/data"))

# ---------------------------------------------------------------------------
# Front-end constants (TensorflowInputMusiCNN, confirmed from Essentia
# source; identical to sidecar/tests/goldens/goldens_meta.json)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
FRAME_SIZE = 512
HOP_SIZE = 256
NUMBER_BANDS = 96
LOW_FREQUENCY_BOUND = 0.0
HIGH_FREQUENCY_BOUND = 8000.0
WARPING_FORMULA = "slaneyMel"
WEIGHTING = "linear"
NORMALIZE = "unit_tri"
BANDS_TYPE = "power"
WINDOW_TYPE = "hann"
WINDOW_NORMALIZED = False
WINDOW_ZERO_PHASE = True
SHIFT = 1.0
SCALE = 10000.0
COMPRESSION = "log10"

# Patching (TensorflowPredictEffnetDiscogs defaults).
PATCH_SIZE = 128
PATCH_HOP_SIZE = 62
LAST_PATCH_MODE = "repeat"

# ---------------------------------------------------------------------------
# Model pins (URL + size + sha256; release metadata from the model cards)
# All classification heads: v2, release 2022-08-25, CC BY-NC-SA.
# effnet bsdynamic sha256 matches sidecar/tests/goldens/goldens_meta.json.
# ---------------------------------------------------------------------------
MODELS_BASE = "https://essentia.upf.edu/models"

MODEL_PINS: dict[str, dict] = {
    "effnet": {
        "url": f"{MODELS_BASE}/feature-extractors/discogs-effnet/discogs-effnet-bsdynamic-1.onnx",
        "filename": "discogs-effnet-bsdynamic-1.onnx",
        "size": 18027718,
        "sha256": "a280825b334797cf677939db8cd5762c0392aedd0ca6415dbc1cd083f045e43c",
        "version": "1",
        "release_date": "2021-11",
    },
    "moodtheme": {
        "url": f"{MODELS_BASE}/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.onnx",
        "filename": "mtg_jamendo_moodtheme-discogs-effnet-1.onnx",
        "size": 2739322,
        "sha256": "7d6270acaa5f4bba4b115a0d6849aca05ed6bd153dcb6d9da4f6ab9f99ef10ff",
        "version": "1",
        "release_date": "2022-08-25",
    },
    "danceability": {
        "url": f"{MODELS_BASE}/classification-heads/danceability/danceability-discogs-effnet-1.onnx",
        "filename": "danceability-discogs-effnet-1.onnx",
        "size": 514101,
        "sha256": "9ce9b8c44f1dd5df5ffc124e5d41d67acf254232c1b90c7e057e079ab7cead73",
        "version": "1",
        "release_date": "2022-08-25",
    },
    "mood_aggressive": {
        "url": f"{MODELS_BASE}/classification-heads/mood_aggressive/mood_aggressive-discogs-effnet-1.onnx",
        "filename": "mood_aggressive-discogs-effnet-1.onnx",
        "size": 514107,
        "sha256": "de36550b5d1660791ad732ed6de6ebfdc3e65dcf50b928b2578ddf103dbfb400",
        "version": "1",
        "release_date": "2022-08-25",
    },
    "mood_party": {
        "url": f"{MODELS_BASE}/classification-heads/mood_party/mood_party-discogs-effnet-1.onnx",
        "filename": "mood_party-discogs-effnet-1.onnx",
        "size": 514097,
        "sha256": "c50ac2106ec2f209dd04ad48756582df0e3f3512235310d1a4a3fcc453745f04",
        "version": "1",
        "release_date": "2022-08-25",
    },
    "mood_relaxed": {
        "url": f"{MODELS_BASE}/classification-heads/mood_relaxed/mood_relaxed-discogs-effnet-1.onnx",
        "filename": "mood_relaxed-discogs-effnet-1.onnx",
        "size": 514101,
        "sha256": "8ba6515a1e5943a72b3b475e3a25fc7a2ff04142c3eaa6aa0716fca371efdfff",
        "version": "1",
        "release_date": "2022-08-25",
    },
    "mood_electronic": {
        "url": f"{MODELS_BASE}/classification-heads/mood_electronic/mood_electronic-discogs-effnet-1.onnx",
        "filename": "mood_electronic-discogs-effnet-1.onnx",
        "size": 514107,
        "sha256": "4cc09140d5b078cf39e55975a23cba813e6bd4014092b78e8c415fa32a382149",
        "version": "1",
        "release_date": "2022-08-25",
    },
    "mood_acoustic": {
        "url": f"{MODELS_BASE}/classification-heads/mood_acoustic/mood_acoustic-discogs-effnet-1.onnx",
        "filename": "mood_acoustic-discogs-effnet-1.onnx",
        "size": 514103,
        "sha256": "56b02abf772b9c1cf528e4d6521d6e09515cedccce47895ee5de8de33fbdf848",
        "version": "1",
        "release_date": "2022-08-25",
    },
}

# The five gating signals — mood_scores is ATOMIC over exactly this set.
MOOD_HEADS = ("aggressive", "party", "relaxed", "electronic", "acoustic")

# ---------------------------------------------------------------------------
# Class-order map (HARDCODED — class order is INCONSISTENT across heads).
#
# Provenance: per-head model cards (v2, release 2022-08-25) at
# https://essentia.upf.edu/models/classification-heads/<head>/<head>-discogs-effnet-1.json
# e.g. mood_party classes = ["non_party", "party"] → positive class index 1.
# Never index by assumption; never derive at runtime.
# ---------------------------------------------------------------------------
POSITIVE_CLASS_INDEX: dict[str, int] = {
    "mood_aggressive": 0,  # classes = ["aggressive", "not_aggressive"]
    "mood_party": 1,       # classes = ["non_party", "party"]
    "mood_relaxed": 1,     # classes = ["non_relaxed", "relaxed"]
    "mood_electronic": 0,  # classes = ["electronic", "non_electronic"]
    "mood_acoustic": 0,    # classes = ["acoustic", "non_acoustic"]
    "danceability": 0,     # classes = ["danceable", "not_danceable"]
}

# ---------------------------------------------------------------------------
# Jamendo moodtheme tensor pin (Catch 2).
#
# The moodtheme ONNX has TWO 56-d outputs: `model/Sigmoid` (predictions,
# in [0,1]) and `model/dense_1/BiasAdd` (logits, unbounded). Shape alone
# cannot disambiguate. The correct graph-output INDEX was resolved during
# development/CI by matching the committed golden probs within ≤1e-3
# (logits fail this hard) and is pinned here. Production loads by pinned
# index — never first-by-shape, never runtime guessing.
# ---------------------------------------------------------------------------
# Resolved in Tier 2/3 against goldens; see test_inference_goldens.py.
MOODTHEME_PROB_OUTPUT_INDEX = int(
    os.environ.get("AXBPM_MOODTHEME_OUTPUT_INDEX", "0")
)

# mood_tags: classes above this threshold, top-N sorted (attribute layer
# only — moodtheme PR-AUC is 0.14; tags are hints, never gating-grade).
MOOD_TAG_THRESHOLD = 0.25
MOOD_TAG_TOP_N = 5

# Jamendo moodtheme class list (from the model card, order preserved).
MOODTHEME_CLASSES: tuple[str, ...] = (
    "action", "adventure", "advertising", "background", "ballad", "calm",
    "children", "christmas", "commercial", "cool", "corporate", "dark",
    "deep", "documentary", "drama", "dramatic", "dream", "emotional",
    "energetic", "epic", "fast", "film", "fun", "funny", "game", "groovy",
    "happy", "heavy", "holiday", "hopeful", "inspiring", "love",
    "meditative", "melancholic", "melodic", "motivational", "movie",
    "nature", "party", "positive", "powerful", "relaxing", "retro",
    "romantic", "sad", "sexy", "slow", "soft", "soundscape", "space",
    "sport", "summer", "trailer", "travel", "upbeat", "uplifting",
)