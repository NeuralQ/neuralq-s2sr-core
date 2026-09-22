# syntax=docker/dockerfile:1.19
# NeuralQ S2SR - dedicated inference image.
#
# Runs on GPU (CUDA, via the NVIDIA Container Toolkit on the host) or CPU
# only; device selection happens at run time through the scripts'
# `--device auto|cuda|cpu` flag (default: auto-detect via torch.cuda).
#
# The checkpoint `models/s2sr-v3.0.0.pt` is baked into the image at build
# time (see .dockerignore — `models/` must NOT be excluded). No network
# access, Hugging Face account, or HF_TOKEN is required at run time.
# Override with `--model /path/to/checkpoint.pt` (bind-mount an alternative).
#
# Build (from the repo root, `models/s2sr-v3.0.0.pt` must exist):
#   docker build -t neuralq-s2sr-core .
#
# Run (GPU):
#   docker run --rm --gpus all \
#     -v "$PWD/outputs:/app/outputs" -v s2sr-cache:/home/mambauser/.cache/neuralq-s2sr-core \
#     neuralq-s2sr-core python scripts/run_location.py --lon 51.531 --lat 25.2886 --date 2026-08-14
#
# Run (CPU only):
#   docker run --rm -v "$PWD/outputs:/app/outputs" \
#     neuralq-s2sr-core python scripts/run_location.py --device cpu ...
#
# Prefer `docker compose` (see docker-compose.yml) for day-to-day use.
FROM mambaorg/micromamba:1.5.10-jammy

ARG MAMBA_DOCKERFILE_ACTIVATE=1
USER root

# GDAL CLI tools (gdalbuildvrt/gdalwarp/gdal_translate) come from the conda
# environment below; only base build tooling is needed from apt.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --chown=$MAMBA_USER:$MAMBA_USER docker/environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba clean --all --yes

# Two layers: code first, 800 MB weights last. Editing code rebuilds only
# the small layer; the weights layer is reused until models/ changes.
# (COPY --exclude needs frontend 1.19+ pinned above.)
COPY --exclude=models --chown=$MAMBA_USER:$MAMBA_USER . /app
COPY --chown=$MAMBA_USER:$MAMBA_USER models/ /app/models/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NEURALQ_ENGINE_MODULE=local_engine \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fail the build early on missing/corrupt weights, then fix perms.
RUN cd /app/models && sha256sum -c s2sr-v3.0.0.pt.sha256 \
    && mkdir -p /app/outputs && chown -R $MAMBA_USER:$MAMBA_USER /app

USER $MAMBA_USER

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["python", "scripts/run_location.py", "--help"]
