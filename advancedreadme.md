# AX BPM — Advanced Documentation

Deep-dive reference for the AX BPM integration and the AX BPM Analyzer
add-on: octave-correction math, analyzer internals, dependency footprint,
and development workflow. For setup and everyday use, see the main
[README.md](README.md).

---

## Table of Contents

- [Octave disambiguation — exact math](#octave-disambiguation--exact-math)
- [Deezer coverage deep-dive](#deezer-coverage-deep-dive)
- [Analyzer internals](#analyzer-internals)
  - [Decode-once pipeline](#decode-once-pipeline)
  - [Mood output contract](#mood-output-contract)
  - [Bit-exactness scope](#bit-exactness-scope)
  - [Image size](#image-size)
- [Dependency footprint](#dependency-footprint)
- [Migration & legacy notes](#migration--legacy-notes)
- [Development](#development)
- [Non-goals](#non-goals)

---

## Octave disambiguation — exact math

Beat trackers sometimes report half or double the real tempo. AX BPM only
ever corrects a raw estimate inside two narrow windows, and only when the
music itself says so. The full decision logic lives in
[`custom_components/ax_bpm/math.py`](custom_components/ax_bpm/math.py);
all tunables live in [`custom_components/ax_bpm/const.py`](custom_components/ax_bpm/const.py).

**Signals** (computed per track):

- `genre_fast` / `genre_slow` — 1 if any Deezer album genre is in the
  `FAST_GENRES` / `SLOW_GENRES` whitelist, else 0.
- `intensity  I = (S_aggressive + S_party + S_electronic + genre_fast) / 4`
- `calmness   C = (S_relaxed + S_acoustic + genre_slow) / 3`

where `S_x` are the analyzer's five gating mood scores in [0, 1]
(`aggressive`, `party`, `relaxed`, `electronic`, `acoustic`).

**Decision order:**

- **Raw reading in [65, 110) BPM** — might really be twice as fast.
  Doubled only if `genre_fast = 1` **or** `I ≥ 0.6`.
- **Raw reading in (150, 200] BPM** — might really be half as fast.
  Halved only if `genre_slow = 1` **or** `C ≥ 0.6`.
- **Everything else is published unchanged.**

**Safety properties:**

- At most ONE correction per track.
- Ambiguity always resolves to "no correction" — a wrong dance band from
  a false correction is worse than one from an ambiguous estimate.
- Graceful degradation: missing mood scores reduce to pure genre means;
  missing genres reduce to pure mood means; both missing → raw estimate
  published.
- Deezer metadata BPM is **never** corrected. Corrections apply only to
  the locally analyzed estimate.

---

## Deezer coverage deep-dive

Deezer's metadata BPM only covers the part of its catalog that reports a
non-zero `bpm` field — coverage is incomplete and varies by release (many
tracks, including top-tier ones, report `bpm: 0`). The local analysis
tier exists precisely for this gap.

Matching pipeline (see [`custom_components/ax_bpm/deezer.py`](custom_components/ax_bpm/deezer.py)):

1. Anonymous field-quoted search `artist:"X" track:"Y"`, with a
   cleaned-title retry for "(Remix)"/"feat." variants.
2. Duration-filtered candidate pick (±3 s tolerance; highest popularity
   wins among candidates).
3. The track's `bpm` field is read as-is when non-zero.

A track with no Deezer BPM and no decodable preview publishes `unknown`,
never 0. On any failure the sensor goes `unknown` — it **never
publishes 0**.

---

## Analyzer internals

The AX BPM Analyzer add-on is a small FastAPI + ONNX Runtime service
(source in [`analyzer/ax_bpm_analyzer/`](analyzer/ax_bpm_analyzer/)).
No TensorFlow, no essentia library — the DSP front end is a pure-NumPy
port of the Essentia C++ algorithms, and the ONNX models are downloaded
from Essentia's public model zoo at first start.

### Decode-once pipeline

The service decodes the track preview **once** at 44.1 kHz, runs the
aubio tempo detector on that buffer, then downsamples to 16 kHz
(scipy `resample_poly`, rational factor via gcd) for the ONNX front end.
One decode, two consumers — a single `/analyze` call returns both tempo
and mood.

### Mood output contract

- `mood_scores` — the **five gating signals** (`aggressive`, `party`,
  `relaxed`, `electronic`, `acoustic`) from dedicated ONNX mood heads.
  This object is **atomic**: it is emitted only when all five heads are
  healthy, and the integration consumes it only when complete — a
  partial failure falls back to genre-only gating rather than reading a
  missing signal as 0.0 (a missing key read as 0.0 would mean
  "maximally non-X" and bias octave gating toward intensity).
- `mood_tags` — rich jamendo moodtheme tags (56 classes, top-N above
  threshold). Attribute layer only, never used for gating.
- `danceability` — degrades independently (omitted if its head is down).
- `valence` / `arousal` (+ `valence_std` / `arousal_std`) — attribute
  layer only.
- `bpm` + `bpm_confidence` — aubio tempo on the 44.1 kHz preview,
  **independent** of mood atomicity: a tempo failure omits these
  without touching `mood_scores`, and vice versa. A tempo failure
  omits bpm entirely — never 0.
- `model_versions` — per-head model version + release date.

Class order is **hardcoded per head** (class order is inconsistent
across Essentia heads — see `POSITIVE_CLASS_INDEX` in
[`analyzer/ax_bpm_analyzer/config.py`](analyzer/ax_bpm_analyzer/config.py));
never indexed by assumption.

The jamendo moodtheme ONNX has TWO 56-d outputs (Sigmoid predictions vs
dense logits); the correct graph-output index is pinned via
`MOODTHEME_PROB_OUTPUT_INDEX` — shape alone cannot disambiguate.

### Bit-exactness scope

The add-on's DSP front end is validated **bit-exact against the Essentia
reference on the committed golden clips only** (per-clip patch sha256s).
Production decode/resample (miniaudio/ffmpeg) is NOT Essentia's
MonoLoader resample path — live-audio mood outputs are
tolerance-bounded equivalents, never claimed bit-exact. The golden clips
bypass the production decode chain entirely (synthesized + resampled
via libsamplerate in the test harness).

Front-end constants (all confirmed from Essentia source, master branch):
sample rate 16000, frameSize 512, hopSize 256, 96 mel bands, 0–8000 Hz,
slaneyMel warping / linear weighting / unit_tri normalization / power
bands, hann window (normalized=false, zeroPhase=true),
`log10(10000 * mel + 1)`, patch 128 / hop 62 / lastPatchMode repeat.

### Image size

The published add-on image is **516 MB (amd64)** / **539 MB (aarch64)**
uncompressed — dominated by the ONNX Runtime + scipy + numpy stack. No
dev/test dependencies, no compiler, no model weights (models download to
`/data` at first start). CI verifies on every build (`probe-runtime-image`
job) that the runtime image contains no dev dependencies: it asserts
`pytest`/`ruff`/`httpx` are unimportable and the test suite is absent.

Trim decisions on record: scipy retained (~70 MB compressed not worth
swapping a battle-tested dep), static-ffmpeg not applicable (input
policy: pre-decoded buffers only), BuildKit wheel mount +
`PIP_NO_COMPILE` skipped (negligible savings; .pyc recompile on
container recreation = first-start CPU spike on Pi).

---

## Dependency footprint

The base integration install has **no extra dependencies** — Deezer
matching and the cache work out of the box, and local BPM analysis works
out of the box too: a built-in NumPy tempo estimator (spectral-flux
onsets + autocorrelation) decodes previews via the `ffmpeg` binary that
ships with official Home Assistant images (or the `miniaudio`/
`soundfile` wheels when installed).

The analyzer is a **two-tier design**:

1. **AX BPM Analyzer add-on (premium tier)** — when installed and
   reachable, ONE `/analyze` call provides aubio-grade tempo AND mood in
   a single request (sensor source attribute: `analyzer`).
2. **Built-in NumPy floor (basic tier)** — always available, no compiled
   dependencies (sensor source attribute: `numpy`).

A missing analyzer never delays or blocks the sensor. aubio lives
exclusively inside the AX BPM Analyzer add-on — the integration never
imports it. When the add-on is unreachable, mood attributes are omitted
and octave disambiguation degrades to genre-only — one info log, no
retry, the BPM sensor is unaffected.

---

## Migration & legacy notes

- **Old "AX-BPM Sidecar" add-on** — the add-on was renamed to
  **AX BPM Analyzer** (new slug `ax_bpm_analyzer`). Uninstall the old
  add-on and install the new one; integration settings migrate
  automatically.
- **Config-entry v1 → v2** — the legacy `mood_analyzer_url` /
  `mood_api_token` keys are renamed to `analyzer_url` /
  `analyzer_api_token` (values preserved) by
  [`custom_components/ax_bpm/migration.py`](custom_components/ax_bpm/migration.py).
- **Legacy genre/mood toggles** — existing installs with the old
  `genre_correction` / `mood_correction` toggle pair migrate to the
  single `octave_disambiguation` dropdown on the next options save
  (genre+mood → "Genre + mood", genre-only → "Genre only", both off →
  "Off").
- **Cache schema v1 → v2** — legacy Essentia SVM mood fields
  (`mood_scores`, `mood_label`) are dropped on load; BPM fields are
  kept (self-healing migration in
  [`custom_components/ax_bpm/store.py`](custom_components/ax_bpm/store.py)).
- **Old cache `source: "sidecar"` strings stay valid** — the sensor
  publishes whatever the cache holds; new writes use `analyzer`.

---

## Development

The analyzer service lives in [`analyzer/`](analyzer/) (FastAPI service)
with HA add-on packaging in [`addon/`](addon/). The devcontainer is
Alpine/musl, where `onnxruntime` is not installable (no musllinux
wheels) — testing follows a three-tier model:

- **Tier 1 (musl workspace)**: everything except real ONNX sessions.

  ```sh
  pytest analyzer/tests -q
  ```

  ONNX tests skip with a reason.

- **Tier 2 (glibc container, iteration only, never the gate)**:

  ```sh
  docker run --rm -p 8099:8099 -v axbpm-models:/data \
    -e AXBPM_API_TOKEN=devtoken \
    -v "$PWD/analyzer:/src" -w /src \
    python:3.12-slim sh -c \
    "pip install -r requirements.txt && python -m ax_bpm_analyzer.api"
  ```

- **Tier 3 (CI, authoritative)**:
  [`.github/workflows/analyzer-golden-validation.yml`](.github/workflows/analyzer-golden-validation.yml)
  runs the full suite with `AXBPM_REQUIRE_ONNX=1` and a zero-skip guard
  on the golden files; any skip fails the job.

**Model weights are never committed** — they download at first start
into `/data` (or the `axbpm-models` volume) and are verified against
pinned sha256 checksums + sizes from
[`analyzer/ax_bpm_analyzer/config.py`](analyzer/ax_bpm_analyzer/config.py)
(`MODEL_PINS`, sourced from `https://essentia.upf.edu/models`).

---

## Non-goals

No blanket tempo folding (only the whitelisted, mood/genre-gated math
above), no TensorFlow models, no essentia library, no Spotify Web API,
no realtime beat streaming, no Music Assistant coupling.