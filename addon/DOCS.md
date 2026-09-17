# AX BPM Analyzer — Documentation

Local tempo (BPM) and mood analysis service for the
[AX BPM](https://github.com/Axildor/AX-BPM) Home Assistant integration.

## What it does

When the AX BPM integration needs to analyze a track locally (Deezer metadata
reports `bpm: 0`), it sends the track's ~30 s preview to this add-on. One
`POST /analyze` call returns:

- `bpm` + `bpm_confidence` — the **aubio** tempo detector on the 44.1 kHz preview
- `mood_scores` — five gating signals (`aggressive`, `party`, `relaxed`,
  `electronic`, `acoustic`) from dedicated ONNX mood heads
- `mood_tags`, `valence`, `arousal`, `danceability`

The integration uses these for mood-corrected octave disambiguation
(**Genre + mood** mode) and richer sensor attributes.

## ✅ No API key required

**You do not need any API key, account, or subscription to use this add-on.**
Everything runs locally:

- Audio analysis: 100% local (aubio + ONNX Runtime on your machine)
- Model weights: downloaded once from Essentia's **public** model zoo
  (~26 MB total, sha256-verified) — no registration, no key
- The AX BPM integration's Deezer lookups are anonymous public API reads

The `api_token` option below is **not** an API key from any provider. It is an
optional shared secret *you make up* to lock the analyzer's `/analyze`
endpoint. Leaving it empty is perfectly fine.

## Setup

1. **Install & start** the add-on. First start downloads the ONNX model
   weights into `/data` and verifies their sha256 checksums (needs internet
   once; subsequent starts are fully offline).
2. **Done.** On Home Assistant OS/Supervised, the add-on announces itself to
   the Supervisor at startup, and the AX BPM integration discovers it
   automatically — no URL or token to configure.
3. In the AX BPM integration, set octave disambiguation to **Genre + mood**
   to use the analyzer's mood scores.

Non-Supervisor installs (e.g. Home Assistant Container) can reach the
analyzer at `homeassistant.local:8099` (port 8099 is host-mapped), or by
entering a manual URL in the integration's *Analyzer URL* field.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `api_token` | *(empty)* | **Optional.** NOT an API key — a shared secret you invent yourself. When set, `/analyze` requires this value as a Bearer token; paste the same value into the integration's *Analyzer API token* field. When left **empty, authentication is disabled** and the add-on works with zero configuration. `/health` is always open (used for auto-detect). |
| `max_analyze_seconds` | 60 | Previews are truncated to this length before inference (latency guard, mainly for aarch64). |
| `intra_op_threads` | 2 | ONNX Runtime intra-op threads. The add-on shares the host CPU with Home Assistant core — keep this small. |

### When should I set `api_token`?

Only if you want to restrict who can call the analyzer. Port 8099 is
host-mapped, so anything on your LAN can reach it. With no token, `/analyze`
accepts unauthenticated requests (each capped at 20 MB upload and CPU-bound
inference). If your network is shared or you expose ports beyond your router,
set a token in the add-on **and** the integration options — they must match.

## How discovery works

At startup the add-on calls the Supervisor discovery API
(`POST /services/discovery`, service `ax_bpm`), so the integration learns the
real resolvable hostname + port instead of guessing. The integration's
detection order is:

1. Manual *Analyzer URL* override (if set)
2. Supervisor-discovered URL (HA-native discovery)
3. `http://homeassistant.local:8099` fallback

## Endpoints

| Endpoint | Auth | Purpose |
| --- | --- | --- |
| `GET /health` | none | Per-model state + tempo availability (auto-detect + diagnostics) |
| `POST /analyze` | Bearer token **only if** `api_token` is set | Tempo + mood analysis of an uploaded audio preview |

## Troubleshooting

- **Sensor stays `unknown` for tracks Deezer knows** — check the add-on log;
  first start must download model weights (internet required once).
- **`/analyze` returns 401** — a token is set in the add-on but not (or
  differently) in the integration. Set the same value in both, or clear the
  add-on's `api_token` to disable auth.
- **Analyzer not detected** — verify the add-on is started, then check the
  integration's status line in its options dialog; it shows the resolved
  analyzer URL.

## License

The add-on code is licensed under the repository's
[LICENSE](https://github.com/Axildor/AX-BPM/blob/main/LICENSE). The ONNX
model weights are downloaded from Essentia's public model zoo and are
licensed CC BY-NC-SA by their authors — they are never redistributed by
this repository.