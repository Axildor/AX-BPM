"""Constants for the AX BPM integration.

All tunable values live here. The octave-disambiguation windows, thresholds
and genre whitelists are the core of the correction logic — tune them with
care, see math.py for the exact decision order.
"""

DOMAIN = "ax_bpm"
NAME = "AX BPM"

# ---------------------------------------------------------------------------
# Configuration keys (config flow + options flow)
# ---------------------------------------------------------------------------
CONF_MEDIA_PLAYER = "media_player"

# Octave disambiguation mode (single dropdown, replaces the legacy
# genre_correction / mood_correction toggle pair).
CONF_OCTAVE_DISAMBIGUATION = "octave_disambiguation"
OCTAVE_OFF = "off"
OCTAVE_GENRE_ONLY = "genre_only"
OCTAVE_GENRE_MOOD = "genre_mood"
OCTAVE_MODES = [OCTAVE_OFF, OCTAVE_GENRE_ONLY, OCTAVE_GENRE_MOOD]

# Manual analyzer URL override (empty = auto-detect only).
CONF_ANALYZER_URL = "analyzer_url"

# URL learned from HA-native add-on discovery (Supervisor). Internal —
# not a user-facing field; written into the entry at discovery time.
CONF_DISCOVERED_ANALYZER_URL = "discovered_analyzer_url"

# ---------------------------------------------------------------------------
# Octave disambiguation — core tunables (see math.py for the decision order)
# ---------------------------------------------------------------------------

# Raw local-analyzer readings in [65, 110) where the real tempo might be
# 2x raw (NumPy floor or analyzer aubio — both feed the same gate).
LOW_WINDOW = (65.0, 110.0)
# Raw local-analyzer readings in (150, 200] where the real tempo might be
# raw / 2.
HIGH_WINDOW = (150.0, 200.0)

# Intensity threshold to double a low reading (mood path).
T_APPLY = 0.6
# Calmness threshold to halve a high reading (mood path).
T_CALM = 0.6

# Deezer album genres that indicate the true tempo is 2x a low raw reading.
FAST_GENRES = frozenset({
    "drum & bass", "drum and bass", "dnb", "jungle", "neurofunk",
    "hardcore", "gabber", "happy hardcore", "breakcore", "speedcore",
})

# Deezer album genres that indicate the true tempo is half a high raw reading.
SLOW_GENRES = frozenset({
    "ballad", "ambient", "downtempo", "chillout", "acoustic",
    "lullaby", "new age",
})

# ---------------------------------------------------------------------------
# Timeouts / budgets (seconds)
# ---------------------------------------------------------------------------
NETWORK_TIMEOUT = 10.0        # per Deezer HTTP request
ANALYSIS_TIMEOUT = 15.0       # per local analyzer (NumPy floor)
# Per analyzer request (hard, single attempt). 25 s: aarch64 inference
# can exceed 8 s; mood is fully async post-publish on the Deezer path, and
# the local path stays bounded by OVERALL_BUDGET.
ANALYZER_TIMEOUT = 25.0
OVERALL_BUDGET = 25.0         # whole per-track resolution budget

# Track-change debounce: wait this long after playback starts / track changes
# before resolving, so players settle their metadata.
TRACK_DEBOUNCE = 3.0

# Candidate duration tolerance for Deezer matching (seconds). Hard filter
# when media_duration is available.
DURATION_TOLERANCE = 3.0

# ---------------------------------------------------------------------------
# Local analysis — decode + tempo chain
# ---------------------------------------------------------------------------
# Sample rate the local analysis chain decodes to (mono float32).
DECODE_SAMPLE_RATE = 22050

# ---------------------------------------------------------------------------
# Platforms / sensor metadata
# ---------------------------------------------------------------------------
PLATFORMS = ["sensor"]
UNIT_BPM = "BPM"
SOURCE_DEEZER = "deezer_metadata"
SOURCE_ANALYZER = "analyzer"
SOURCE_NUMPY = "numpy"
SOURCE_CACHE = "cache"

# AX BPM Analyzer add-on endpoints.
# Auto-detect order: manual analyzer_url override → discovered URL
# (Supervisor discovery) → homeassistant.local fallback. Empty manual URL
# = auto-detect only.
ANALYZER_PORT = 8099
ANALYZER_URLS = (
    f"http://homeassistant.local:{ANALYZER_PORT}",  # external analyzer mode
)
ANALYZE_PATH = "/analyze"
HEALTH_PATH = "/health"

# Shared-secret auth for the analyzer /analyze endpoint (Bearer token).
# Set the same token in the add-on config and here; empty = no token sent
# (the analyzer answers 401 when its own token is set).
CONF_ANALYZER_API_TOKEN = "analyzer_api_token"

# The five gating signals. mood_scores from the analyzer is ATOMIC over
# exactly this set: consumed only when present AND complete — a missing
# key read as 0.0 would mean "maximally non-X" and bias octave gating
# toward intensity during partial failure. Incomplete → genre-only.
EXPECTED_MOOD_SCORES = frozenset({
    "aggressive", "party", "relaxed", "electronic", "acoustic",
})

# Deezer endpoints (anonymous, no key required for public read endpoints).
DEEZER_API = "https://api.deezer.com"
