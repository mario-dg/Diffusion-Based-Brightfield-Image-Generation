FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ARG WANDB_TOKEN
ARG HF_TOKEN

RUN test -n "$WANDB_TOKEN" || (echo "WANDB_TOKEN  not set" && false)
RUN test -n "$HF_TOKEN" || (echo "HF_TOKEN  not set" && false)

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3.10 \
    python3-pip \
    python3.10-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configure Poetry
ENV POETRY_VERSION=1.8.2
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache

# Install poetry separated from system interpreter
RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

# Add `poetry` to PATH
ENV PATH="${PATH}:${POETRY_VENV}/bin"

WORKDIR /data

COPY src/ /data/src/
COPY pyproject.toml /data/
COPY poetry.lock /data/

RUN poetry install
RUN poetry run wandb login ${WANDB_TOKEN}
RUN poetry run huggingface-cli login --token ${HF_TOKEN}

CMD ["/bin/bash"]

# docker build --build-arg WANDB_TOKEN=e56c562c8f2e2daf5c6f7ed076f0439155900d4c --build-arg HF_TOKEN=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 -t pldiffusion .
# docker run --rm -it --gpus '"device=0,1"' --shm-size=256g -v /mnt/beegfs/mario.dejesusdagraca/brightfield-microscopy-scc:/data/.cache pldiffusion