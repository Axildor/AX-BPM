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
CONF_GENRE_CORRECTION = "genre_correction"
CONF_MOOD_CORRECTION = "mood_correction"
CONF_AUBIO_BINARY = "aubio_binary"
CONF_HELPER_SCRIPT = "helper_script"
CONF_GETSONGKEY_API_KEY = "getsongkey_api_key"

# ---------------------------------------------------------------------------
# Octave disambiguation — core tunables (see math.py for the decision order)
# ---------------------------------------------------------------------------

# Raw aubio readings in [65, 110) where the real tempo might be 2x raw.
LOW_WINDOW = (65.0, 110.0)
# Raw aubio readings in (150, 200] where the real tempo might be raw / 2.
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
ANALYSIS_TIMEOUT = 15.0       # per analyzer (aubio / essentia)
OVERALL_BUDGET = 25.0         # whole per-track resolution budget

# Track-change debounce: wait this long after playback starts / track changes
# before resolving, so players settle their metadata.
TRACK_DEBOUNCE = 3.0

# Candidate duration tolerance for Deezer matching (seconds). Hard filter
# when media_duration is available.
DURATION_TOLERANCE = 3.0

# ---------------------------------------------------------------------------
# Platforms / sensor metadata
# ---------------------------------------------------------------------------
PLATFORMS = ["sensor"]
UNIT_BPM = "BPM"
SOURCE_DEEZER = "deezer_metadata"
SOURCE_AUBIO = "aubio"
SOURCE_CACHE = "cache"

# Essentia SVM mood classifiers used (five S_x signals).
MOOD_KEYS = ("aggressive", "party", "electronic", "relaxed", "acoustic")

# Essentia SVM model file names (Gaia .history format, CC BY-NC-ND licensed).
ESSENTIA_MODEL_BASE_URL = "https://essentia.upf.edu/models/svm-models/mood"
ESSENTIA_MODELS = {
    "aggressive": "mood_aggressive.history",
    "party": "mood_party.history",
    "electronic": "mood_electronic.history",
    "relaxed": "mood_relaxed.history",
    "acoustic": "mood_acoustic.history",
}

# Deezer endpoints (anonymous, no key required for public read endpoints).
DEEZER_API = "https://api.deezer.com"