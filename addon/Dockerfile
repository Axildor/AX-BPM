# AX-BPM Sidecar — multi-stage build.
# Runtime stage: python:3.12-slim (Debian/glibc) + pinned requirements ONLY.
# Dev stage: adds requirements-dev.txt (pytest/httpx/ruff) for Tier 2
# iteration (docker compose exec <sidecar-svc> pytest sidecar/tests).
# NO essentia, NO TensorFlow — pure NumPy front end + ONNX Runtime.
FROM python:3.12-slim AS runtime

WORKDIR /usr/src/axbpm
COPY sidecar/requirements.txt sidecar/requirements.txt
RUN pip install --no-cache-dir -r sidecar/requirements.txt

COPY sidecar/ax_bpm_sidecar sidecar/ax_bpm_sidecar
COPY addon/run.sh addon/run.sh
RUN chmod +x addon/run.sh

ENV AXBPM_DATA_DIR=/data
VOLUME /data
EXPOSE 8099

CMD ["python", "-m", "ax_bpm_sidecar.api"]

# --- Dev target (Tier 2 iteration only — never published) -------------------
FROM runtime AS dev
COPY sidecar/requirements-dev.txt sidecar/requirements-dev.txt
RUN pip install --no-cache-dir -r sidecar/requirements-dev.txt
COPY sidecar/tests sidecar/tests
COPY pytest.ini pytest.ini