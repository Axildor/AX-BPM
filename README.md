[![GitHub Release](https://img.shields.io/github/v/release/adix992/AX-BPM?style=flat-square)](https://github.com/adix992/AX-BPM/releases)
[![HACS Status](https://img.shields.io/badge/HACS-Custom-orange.svg?style=flat-square)](https://github.com/hacs/integration)
[![Add-on Build](https://img.shields.io/github/actions/workflow/status/adix992/AX-BPM/addon-build.yml?branch=main&label=Add-on%20Build&style=flat-square)](https://github.com/adix992/AX-BPM/actions/workflows/addon-build.yml)
[![Buy me a tea](https://img.shields.io/badge/Buy_me_a_tea-☕-FF5E5B?style=flat-square&logo=ko-fi&logoColor=white)](https://ko-fi.com/axildor)

# AX BPM for Home Assistant

A custom Home Assistant integration that publishes a sensor holding the
currently playing track's tempo (BPM). Built for the
[AXiDOS Avatar Card](https://github.com/adix992/AXiDOS-Avatar-Card) dance
engine, but works with any automation that needs a live BPM value.

**All network requests are anonymous** — the Deezer public API needs no API
key, no account, and no OAuth. Setup is UI-only; no YAML required.

> 🔧 Deep-dive reference (octave-correction math, analyzer internals,
> development workflow): see **[advancedreadme.md](advancedreadme.md)**.

---

## How it works

When the tracked `media_player` starts a new track (debounced a few seconds),
AX BPM resolves its tempo in this order:

1. **Cache** — a persistent store (`.storage`, survives restarts). A hit
   publishes instantly with zero network calls.
2. **Deezer metadata** — an anonymous search, then the track's `bpm`
   field. If Deezer reports a BPM, it is published as-is.
3. **Local analysis** — Deezer frequently reports `bpm: 0` even for
   top-tier catalog. In that case AX BPM downloads the track's ~30 s
   preview once and analyzes it locally. With the **AX BPM Analyzer
   add-on** installed, ONE `/analyze` call returns both the **aubio**
   tempo and the mood scores, so the first published BPM is already
   mood-corrected. Without the add-on, a built-in **NumPy** tempo
   estimator provides the BPM, degrading octave gating to genre-only.

On any failure the sensor goes `unknown` — it **never publishes 0**.

## Octave disambiguation (plain language)

Beat trackers sometimes report half or double the real tempo. AX BPM only
ever corrects a raw estimate inside two narrow windows, and only when the
music itself says so:

- **Raw reading between 65–110 BPM** might really be twice as fast.
  It is doubled only if the album genre is a fast genre (drum & bass,
  jungle, hardcore, gabber, breakcore, …) **or** the analyzer scores the
  track as intense (aggressive/party/electronic average ≥ 0.6).
- **Raw reading between 150–200 BPM** might really be half as fast.
  It is halved only if the album genre is a slow genre (ballad, ambient,
  downtempo, acoustic, …) **or** the analyzer scores it as calm
  (relaxed/acoustic average ≥ 0.6).
- **Everything else is published unchanged.** At most one correction per
  track; ambiguity always resolves to "no correction".

Deezer metadata BPM is never corrected. Corrections apply only to the
locally analyzed estimate. The exact windows, thresholds, and signal
formulas are documented in
[advancedreadme.md](advancedreadme.md#octave-disambiguation--exact-math).

## Installation

**HACS (recommended)**
1. HACS → ⋮ → *Custom repositories*.
2. Add this repository, category **Integration**.
3. Download, restart Home Assistant.

**Manual**
1. Copy `custom_components/ax_bpm` into your `config/custom_components`.
2. Restart Home Assistant.

## AX BPM Analyzer add-on (tempo + mood analysis)

Tempo (aubio-grade) and mood analysis both run in the **AX BPM Analyzer**
add-on — a small FastAPI + ONNX Runtime service (no TensorFlow, no
essentia library — the DSP front end is a pure-NumPy port of the Essentia
algorithms, and the ONNX models are downloaded from Essentia's public
model zoo at first start). It decodes the track preview once at 44.1 kHz,
runs the aubio tempo detector on it, downsamples to 16 kHz, and returns
`bpm` (+ `bpm_confidence`), mood scores, mood tags, and danceability in
one response.

**Install the add-on**
1. Settings → Add-ons → ⋮ → *Repositories* → add this repository URL.
2. The **AX BPM Analyzer** add-on appears in the list — install it
   (first start downloads ~26 MB of ONNX model weights into `/data`
   and verifies their sha256 checksums).
3. Start the add-on. It listens on port **8099** (host-mapped by
   default, so `homeassistant.local:8099` works from the integration).

> **Upgrading from the old "AX-BPM Sidecar" add-on?** The add-on was
> renamed to **AX BPM Analyzer** (new slug `ax_bpm_analyzer`). Uninstall
> the old add-on and install the new one — your integration settings
> migrate automatically.

**Add-on options**

| Option | Default | Description |
| --- | --- | --- |
| `api_token` | *(empty)* | Shared secret for `/analyze`. **Set this** — with an empty token the add-on answers 401 to every analysis request (one clear log hint). `/health` stays open for auto-detect. |
| `max_analyze_seconds` | 60 | Previews are truncated to this length before inference (latency guard, mainly for aarch64). |
| `intra_op_threads` | 2 | ONNX Runtime intra-op threads. The add-on shares the host CPU with Home Assistant core — keep this small. |

**Connect the integration**
1. In the AX BPM integration options, set octave disambiguation to
   **Genre + mood**.
2. Paste the same token into the new **Analyzer API token** field.
3. Leave the analyzer URL empty for auto-detect. On Home Assistant OS the
   add-on announces itself to the Supervisor, so the integration finds it
   automatically; otherwise it falls back to `homeassistant.local:8099`.

**What the add-on returns**

- `bpm` + `bpm_confidence` — aubio tempo on the 44.1 kHz preview
  (independent of mood: a tempo failure omits these without touching
  `mood_scores`, and vice versa).
- `mood_scores` — the **five gating signals** (`aggressive`, `party`,
  `relaxed`, `electronic`, `acoustic`) from dedicated ONNX mood heads.
  These five are the set used for octave gating; the add-on additionally
  returns 56 jamendo `mood_tags`, `danceability`, and
  `valence`/`arousal` as attribute-layer data. The `mood_scores` object
  is **atomic**: it is emitted only when all five heads are healthy, and
  the integration uses it only when complete — a partial failure falls
  back to genre-only gating rather than reading a missing signal as 0.0.
- `mood_tags` — rich jamendo moodtheme tags (attribute layer only,
  never used for gating).
- `danceability` — degrades independently (omitted if its head is down).

## Configuration

Settings → Devices & Services → **Add Integration** → **AX BPM**:

| Option | Description |
| --- | --- |
| Media player | The `media_player` entity to track (required). |
| Octave disambiguation | One dropdown: **Off** / **Genre only** / **Genre + mood** (default: Genre only). "Genre + mood" needs the AX BPM Analyzer add-on and degrades to genre-only while it is unreachable. |
| Analyzer URL | Optional manual override. Leave empty to auto-detect (Supervisor discovery, then `homeassistant.local:8099`). |
| Analyzer API token | Shared secret matching the add-on's `api_token`. Required when the add-on has a token set. |

The setup form shows a connection status line with the analyzer
auto-detect result. Existing installs with the legacy genre/mood toggles
or the old `mood_analyzer_url` / `mood_api_token` settings migrate
automatically on upgrade.

## Sensor

`sensor.ax_bpm` — state is the final BPM (unit `BPM`, measurement class).
Attributes include the source (`deezer_metadata` / `analyzer` / `numpy` /
`cache`), track name, ISRC, Deezer track id, match rank, the
pre-correction raw BPM, album genre, mood scores, intensity/calmness, the
octave rule applied, and the last update time.

When the analyzer responds (Deezer-metadata path, or the concurrent local
path), additional mood attributes are published:
`mood_tags` (top-N above threshold), `valence`, `arousal`,
`valence_std`/`arousal_std`, `danceability`, `analyzed_seconds`,
`model_versions`, and `source="analyzer"`. The `intensity`/`calmness`
attribute names are kept for automation back-compat — they are now
computed from the analyzer tags (aggressive/party/electronic vs
relaxed/acoustic averages) with arousal as tie-breaker.

- **Pause** retains the last value. **Stop / off** sets `unknown`.
- Mood attributes never delay the BPM publish: on the Deezer-metadata
  path they arrive as a second state write after the BPM is published.

---

## ☕ Support the Project

I'm a solo developer on disability building Home Assistant integrations
and add-ons independently. Your support keeps servers online, API quotas
funded, and the black tea brewing while I debug Python.

If this integration is useful to you, there's no obligation — but any
support is highly appreciated.

[![Buy me a tea](https://img.shields.io/badge/Buy_me_a_tea-on_Ko--fi-FF5E5B?style=for-the-badge&logo=ko-fi&logoColor=white)](https://ko-fi.com/axildor)

---

## License

MIT
