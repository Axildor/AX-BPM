# AX-BPM Sidecar — multi-stage build.
# Runtime stage: python:3.12-slim (Debian/glibc) + pinned requirements ONLY.
# Dev stage: adds requirements-dev.txt (pytest/httpx/ruff) for Tier 2
# iteration (docker compose exec <sidecar-svc> pytest sidecar/tests).
# NO essentia, NO TensorFlow — pure NumPy front end + ONNX Runtime.

# Wheel-build stage: miniaudio ships no aarch64 wheels on PyPI, so on arm64
# pip must compile it from source (needs a C++ toolchain). Build the wheel
# here with build-essential, then install the prebuilt wheel into the slim
# runtime stage so the compiler never ships in the published image.
FROM python:3.12-slim AS wheelbuild
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip wheel --no-cache-dir miniaudio==1.71 -w /wheels

FROM python:3.12-slim AS runtime

WORKDIR /usr/src/axbpm
COPY --from=wheelbuild /wheels /wheels
COPY sidecar/requirements.txt sidecar/requirements.txt
RUN pip install --no-cache-dir -r sidecar/requirements.txt && rm -rf /wheels

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