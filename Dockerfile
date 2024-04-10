FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3.10 \
    python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app

COPY . /data
RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]

# This image let's you create an isolated dev environment for POSTER_V2
# docker build -t pv2-train .
# docker run --rm -it --gpus all --shm-size=8g -v .:/app/POSTER_V2 pv2-train