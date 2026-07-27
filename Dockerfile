# SmarTrain runtime image (CUDA 12.8 / cu128 torch policy).
# Build: docker build -t smartrain:local .
# GPU:   docker run --rm --gpus all smartrain:local --help
# CPU:   docker run --rm smartrain:local --help

ARG CUDA_IMAGE=nvidia/cuda:12.8.0-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG PYTHON_VERSION=3.11
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_VERSION=2.7.1
ARG TORCHVISION_VERSION=0.22.1
ARG TORCHAUDIO_VERSION=2.7.1
ARG ULTRALYTICS_VERSION=8.3.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SMART_TRAIN_SKIP_TORCH_POLICY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python3-pip \
        git \
        libgl1 \
        libglib2.0-0 \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY smartrain ./smartrain

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
        --index-url ${TORCH_INDEX_URL} \
        torch==${TORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        torchaudio==${TORCHAUDIO_VERSION} \
    && python -m pip install "ultralytics==${ULTRALYTICS_VERSION}" \
    && python -m pip install -e ".[dev]"

ENTRYPOINT ["smartrain"]
CMD ["--help"]
