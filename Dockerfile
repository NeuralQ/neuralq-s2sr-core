# syntax=docker/dockerfile:1
# NeuralQ S2SR - dedicated inference image.
#
# Runs on GPU (CUDA, via the NVIDIA Container Toolkit on the host) or CPU
# only; device selection happens at run time through the scripts'
# `--device auto|cuda|cpu` flag (default: auto-detect via torch.cuda).
#
# The checkpoint is downloaded from the private Hugging Face repo
# Khlaifiabilel/neuralq-s2sr-core on first use, so HF_TOKEN (a Hugging Face
# read token) must be set in the environment.
#
# Build:
#   docker build -t neuralq-s2sr-core .
#
# Run (GPU):
#   docker run --rm --gpus all -e HF_TOKEN \
#     -v "$PWD/outputs:/app/outputs" -v s2sr-cache:/home/mambauser/.cache/neuralq-s2sr-core \
#     -v hf-cache:/home/mambauser/.cache/huggingface \
#     neuralq-s2sr-core python scripts/run_location.py --lon 51.531 --lat 25.2886 --date 2026-08-14
#
# Run (CPU only):
#   docker run --rm -e HF_TOKEN -v "$PWD/outputs:/app/outputs" \
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

COPY --chown=$MAMBA_USER:$MAMBA_USER . /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NEURALQ_ENGINE_MODULE=local_engine \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN mkdir -p /app/outputs && chown -R $MAMBA_USER:$MAMBA_USER /app

USER $MAMBA_USER

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["python", "scripts/run_location.py", "--help"]
