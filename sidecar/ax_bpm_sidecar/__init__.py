"""AX-BPM sidecar mood analyzer service (Phase 2).

FastAPI service exposing POST /analyze and GET /health for the AX-BPM
Home Assistant integration. Pure NumPy front end + ONNX Runtime inference;
no essentia, no TensorFlow.
"""

__version__ = "1.0.0"