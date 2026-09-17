# AX BPM Analyzer

Tempo (BPM) and mood analyzer for the [AX BPM](https://github.com/Axildor/AX-BPM)
Home Assistant integration. Runs the **aubio** tempo detector plus Essentia-grade
ONNX mood models as a small local sidecar service, so the integration gets
aubio-grade tempo and mood-corrected octave disambiguation without any cloud
dependency.

## ✅ No API key required

**This add-on needs no API key, no account, and no sign-up of any kind.** All
audio analysis runs locally on your Home Assistant machine. The only network
traffic is a one-time download of the public ONNX model weights from Essentia's
model zoo at first start.

The optional `api_token` option is **not** an API key from any provider — it is
a shared secret *you invent yourself* to protect the analyzer's `/analyze`
endpoint. Leave it empty and everything works out of the box.

## Quick start

1. Install the add-on and start it (first start downloads ~26 MB of model
   weights into `/data` and verifies their sha256 checksums).
2. That's it. The AX BPM integration discovers the analyzer automatically via
   the Supervisor — no URL, no token, no configuration needed.

For full details, see the **Documentation** tab.