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
   preview once and runs two analyzers on it:
   - **aubio** for tempo (60 / median inter-beat interval), and
   - **Essentia SVM mood models** for octave-disambiguation signals.

On any failure the sensor goes `unknown` — it **never publishes 0**.

## Octave disambiguation (plain language)

Beat trackers sometimes report half or double the real tempo. AX BPM only
ever corrects a raw estimate inside two narrow windows, and only when the
music itself says so:

- **Raw reading between 65–110 BPM** might really be twice as fast.
  It is doubled only if the album genre is a fast genre (drum & bass,
  jungle, hardcore, gabber, breakcore, …) **or** the Essentia mood
  classifiers score the track as intense (aggressive/party/electronic
  average ≥ 0.6).
- **Raw reading between 150–200 BPM** might really be half as fast.
  It is halved only if the album genre is a slow genre (ballad, ambient,
  downtempo, acoustic, …) **or** the mood classifiers score it as calm
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

## Configuration

Settings → Devices & Services → **Add Integration** → **AX BPM**:

| Option | Description |
| --- | --- |
| Media player | The `media_player` entity to track (required). |
| Genre-based octave disambiguation | Use Deezer album genres in the correction rules (default: on). |
| Mood-based octave disambiguation | Use Essentia mood classifiers (default: on; auto-disables with a warning if Essentia is unavailable). |
| External aubio binary | Path to an `aubio` CLI binary, used when the Python package is missing. |
| External analyzer helper | Path to an external analyzer helper script/binary (subprocess mode). |
| GetSongKEY API key | Optional additional metadata fallback (disabled by default). |

## Sensor

`sensor.ax_bpm` — state is the final BPM (unit `BPM`, measurement class).
Attributes include the source (`deezer_metadata` / `aubio` / `cache`),
track name, ISRC, Deezer track id, match rank, the pre-correction raw BPM,
album genre, mood scores, intensity/calmness, the octave rule applied, and
the last update time.

- **Pause** retains the last value. **Stop / off** sets `unknown`.

## Dependency footprint

The base install has **no extra dependencies** — Deezer matching and the
cache work out of the box. Optional local analysis adds:

- `aubio` (Python package, or any external `aubio` CLI binary) — tempo.
- `essentia` (Python package) — SVM mood classifiers. The five mood model
  files (`.history`, CC BY-NC-ND licensed) are looked up in the wheel's
  package data; if absent they are downloaded **once** into the
  integration's storage directory at first setup.

Missing analyzers degrade gracefully: mood+genre → genre-only → raw.
A missing analyzer never delays or blocks the sensor.

## Non-goals

No blanket tempo folding (only the whitelisted, mood/genre-gated math
above), no TensorFlow models, no Spotify Web API, no realtime beat
streaming, no Music Assistant coupling.
