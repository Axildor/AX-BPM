# AX-BPM Sidecar — multi-stage build.
# Runtime stage: python:3.12-slim (Debian/glibc) + pinned requirements ONLY.
# Dev stage: adds requirements-dev.txt (pytest/httpx/ruff) for Tier 2
# iteration (docker compose exec <sidecar-svc> pytest sidecar/tests).
# NO essentia, NO TensorFlow — pure NumPy front end + ONNX Runtime.

# Wheel-build stage: miniaudio ships no aarch64 wheels on PyPI, and aubio is
# source-only for ALL platforms (no wheels at all), so on any arch pip must
# compile them from source (needs a C/C++ toolchain; aubio also wants libav
# headers only when its own file decoding is used — NOT our case: we feed
# pre-decoded samples, so no ffmpeg/libav linkage). Build the wheels here
# with build-essential, then install the prebuilt wheels into the slim
# runtime stage so the compiler never ships in the published image.
FROM python:3.12-slim AS wheelbuild
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip wheel --no-cache-dir miniaudio==1.71 -w /wheels
# aubio 0.4.9 builds via setup.py + distutils build_ext (NOT waf — waf
# only builds the standalone libaubio we never link). Its C extension
# includes numpy/arrayobject.h unconditionally, and python:3.12 ships no
# setuptools — so --no-build-isolation requires setuptools + wheel + the
# pinned numpy present IN the stage before the aubio wheel build.
# aubio 0.4.9 predates numpy 2.0's const-qualified PyUFuncGenericFunction
# signature; GCC 14 (trixie base) makes -Wincompatible-pointer-types a
# hard error, so downgrade it to a warning for this legacy source
# (building against numpy 1.x instead would be ABI-incompatible at
# runtime with the pinned numpy 2.5.3).
ENV CFLAGS="-Wno-error=incompatible-pointer-types"
RUN pip install --no-cache-dir numpy==2.5.3 setuptools wheel \
    && pip wheel --no-cache-dir --no-build-isolation aubio==0.4.9 -w /wheels

FROM python:3.12-slim AS runtime

WORKDIR /usr/src/axbpm
COPY --from=wheelbuild /wheels /wheels
COPY sidecar/requirements.txt sidecar/requirements.txt
# --find-links makes pip prefer the prebuilt miniaudio wheel over the PyPI
# source tarball (no aarch64 wheels exist upstream).
RUN pip install --no-cache-dir --find-links=/wheels -r sidecar/requirements.txt \
    && rm -rf /wheels

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