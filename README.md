# AX BPM for Home Assistant

A custom Home Assistant integration that publishes a sensor holding the
currently playing track's tempo (BPM). Built for the
[AXiDOS Avatar Card](https://github.com/adix992/AXiDOS-Avatar-Card) dance
engine, but works with any automation that needs a live BPM value.

**All network requests are anonymous** — the Deezer public API needs no API
key, no account, and no OAuth. Setup is UI-only; no YAML required.

## How it works

When the tracked `media_player` starts a new track (debounced a few seconds),
AX BPM resolves its tempo in this order:

1. **Cache** — a persistent store (`.storage`, survives restarts) keyed on
   the track's ISRC (or a hash of artist/title/duration). A hit publishes
   instantly with zero network calls.
2. **Deezer metadata** — an anonymous field-quoted search
   (`artist:"X" track:"Y"`, with a cleaned-title retry for
   "(Remix)"/"feat." variants), a duration-filtered candidate pick
   (±3 s, highest popularity wins), then the track's `bpm` field.
   If Deezer reports a BPM, it is published as-is.
3. **Local analysis** — Deezer frequently reports `bpm: 0` even for
   top-tier catalog. In that case AX BPM downloads the track's ~30 s
   preview once and runs the **aubio** tempo analyzer on it (60 / median
   inter-beat interval). When octave disambiguation is set to
   "Genre + mood", the same preview is POSTed to the sidecar mood
   analyzer concurrently, so the first published BPM is already
   mood-corrected.

On any failure the sensor goes `unknown` — it **never publishes 0**.

## Octave disambiguation (plain language)

Beat trackers sometimes report half or double the real tempo. AX BPM only
ever corrects a raw estimate inside two narrow windows, and only when the
music itself says so:

- **Raw reading between 65–110 BPM** might really be twice as fast.
  It is doubled only if the album genre is a fast genre (drum & bass,
  jungle, hardcore, gabber, breakcore, …) **or** the sidecar mood
  analyzer scores the track as intense (aggressive/party/electronic
  average ≥ 0.6).
- **Raw reading between 150–200 BPM** might really be half as fast.
  It is halved only if the album genre is a slow genre (ballad, ambient,
  downtempo, acoustic, …) **or** the mood analyzer scores it as calm
  (relaxed/acoustic average ≥ 0.6).
- **Everything else is published unchanged.** At most one correction per
  track; ambiguity always resolves to "no correction" — a wrong dance band
  from a false correction is worse than one from an ambiguous estimate.

Deezer metadata BPM is never corrected. Corrections apply only to the
locally analyzed estimate.

## Installation

**HACS (recommended)**
1. HACS → ⋮ → *Custom repositories*.
2. Add this repository, category **Integration**.
3. Download, restart Home Assistant.

**Manual**
1. Copy `custom_components/ax_bpm` into your `config/custom_components`.
2. Restart Home Assistant.

## Sidecar add-on (mood analysis)

Mood analysis runs in the **AX BPM sidecar** add-on — a small FastAPI +
ONNX Runtime service (no TensorFlow, no essentia) that analyzes the
track preview and returns mood scores, mood tags, and danceability.

**Install the add-on**
1. Settings → Add-ons → ⋮ → *Repositories* → add this repository URL.
2. The **AX BPM Sidecar** add-on appears in the list — install it
   (first start downloads ~26 MB of ONNX model weights into `/data`
   and verifies their sha256 checksums).
3. Start the add-on. It listens on port **8099** (host-mapped by
   default, so `homeassistant.local:8099` works from the integration).

**Add-on options**

| Option | Default | Description |
| --- | --- | --- |
| `api_token` | *(empty)* | Shared secret for `/analyze`. **Set this** — with an empty token the sidecar answers 401 to every analysis request (one clear log hint). `/health` stays open for auto-detect. |
| `max_analyze_seconds` | 60 | Previews are truncated to this length before inference (latency guard, mainly for aarch64). |
| `intra_op_threads` | 2 | ONNX Runtime intra-op threads. The sidecar shares the host CPU with Home Assistant core — keep this small. |

**Connect the integration**
1. In the AX BPM integration options, set octave disambiguation to
   **Genre + mood**.
2. Paste the same token into the new **Sidecar API token** field.
3. Leave the sidecar URL empty for auto-detect (add-on internal
   hostname, then `homeassistant.local:8099`).

**What the sidecar returns**

- `mood_scores` — five signals (`aggressive`, `party`, `relaxed`,
  `electronic`, `acoustic`) from dedicated ONNX mood heads. This object
  is **atomic**: it is emitted only when all five heads are healthy, and
  the integration uses it only when complete — a partial failure falls
  back to genre-only gating rather than reading a missing signal as 0.0.
- `mood_tags` — rich jamendo moodtheme tags (attribute layer only,
  never used for gating).
- `danceability` — degrades independently (omitted if its head is down).

**Bit-exactness scope**: the sidecar's DSP front end is validated
bit-exact against the essentia reference on the committed golden clips
only. Production decode/resample (miniaudio/ffmpeg) is a
tolerance-bounded equivalent — live-audio mood outputs are never claimed
bit-exact.

## Configuration

Settings → Devices & Services → **Add Integration** → **AX BPM**:

| Option | Description |
| --- | --- |
| Media player | The `media_player` entity to track (required). |
| Octave disambiguation | One dropdown: **Off** / **Genre only** / **Genre + mood** (default: Genre only). "Genre + mood" needs the AX BPM sidecar add-on and degrades to genre-only while it is unreachable. |
| Sidecar mood analyzer URL | Optional manual override. Leave empty to auto-detect (add-on internal hostname, then `homeassistant.local:8099`). |
| Sidecar API token | Shared secret matching the add-on's `api_token`. Required when the add-on has a token set. |
| External aubio binary | Path to an `aubio` CLI binary, used when the Python package is missing. |

The setup form shows a connection status line with the sidecar
auto-detect result. Existing installs with the legacy genre/mood toggles
migrate automatically on the next options save (genre+mood → "Genre +
mood", genre-only → "Genre only", both off → "Off").

## Sensor

`sensor.ax_bpm` — state is the final BPM (unit `BPM`, measurement class).
Attributes include the source (`deezer_metadata` / `aubio` / `cache`),
track name, ISRC, Deezer track id, match rank, the pre-correction raw BPM,
album genre, mood scores, intensity/calmness, the octave rule applied, and
the last update time.

When the sidecar mood analyzer responds (Deezer-metadata path, or the
concurrent local path), additional mood attributes are published:
`mood_tags` (top-N above threshold), `valence`, `arousal`,
`valence_std`/`arousal_std`, `danceability`, `analyzed_seconds`,
`model_versions`, and `source="sidecar"`. The `intensity`/`calmness`
attribute names are kept for automation back-compat — they are now
computed from the sidecar tags (aggressive/party/electronic vs
relaxed/acoustic averages) with arousal as tie-breaker.

- **Pause** retains the last value. **Stop / off** sets `unknown`.
- Mood attributes never delay the BPM publish: on the Deezer-metadata
  path they arrive as a second state write after the BPM is published.

## Dependency footprint

The base install has **no extra dependencies** — Deezer matching and the
cache work out of the box, and **local BPM analysis works out of the box**
too: a built-in NumPy tempo estimator (spectral-flux onsets +
autocorrelation) decodes previews via the `ffmpeg` binary that ships with
official Home Assistant images (or the `miniaudio`/`soundfile` wheels when
installed).

Optional accuracy upgrades, picked up automatically when importable:

- `aubio` (Python package, or any external `aubio` CLI binary) — tempo.

The analyzer chain is aubio → aubio CLI → built-in NumPy; the first
backend that produces a BPM wins (shown in the sensor's source attribute
as `aubio` or `numpy`). A missing analyzer never delays or blocks the
sensor.

Mood analysis is provided by the **AX BPM sidecar add-on** (ONNX-based,
no TensorFlow in Home Assistant). When the sidecar is unreachable, mood
attributes are omitted and octave disambiguation degrades to genre-only —
one info log, no retry, the BPM sensor is unaffected.

## Non-goals

No blanket tempo folding (only the whitelisted, mood/genre-gated math
above), no TensorFlow models, no Spotify Web API, no realtime beat
streaming, no Music Assistant coupling.

## Development

The sidecar lives in `sidecar/` (FastAPI service) with HA add-on
packaging in `addon/`. The devcontainer is Alpine/musl, where
`onnxruntime` is not installable (no musllinux wheels) — testing follows
a three-tier model:

- **Tier 1 (musl workspace)**: everything except real ONNX sessions.
  `pytest sidecar/tests -q` — ONNX tests skip with a reason.
- **Tier 2 (glibc container, iteration only, never the gate)**:

  ```sh
  docker run --rm -p 8099:8099 -v axbpm-models:/data \
    -e AXBPM_API_TOKEN=devtoken \
    -v "$PWD/sidecar:/src" -w /src \
    python:3.12-slim sh -c \
    "pip install -r requirements.txt && python -m ax_bpm_sidecar.api"
  ```

- **Tier 3 (CI, authoritative)**: `.github/workflows/sidecar-golden-validation.yml`
  runs the full suite with `AXBPM_REQUIRE_ONNX=1` and a zero-skip guard
  on the golden files; any skip fails the job.

Model weights are never committed — they download at first start into
`/data` (or the `axbpm-models` volume) and are verified against pinned
sha256 checksums from `sidecar/ax_bpm_sidecar/config.py`.
