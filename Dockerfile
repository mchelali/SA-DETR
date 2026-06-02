# ARG PYTHON_VERSION=3.10-slim
# FROM python:${PYTHON_VERSION}

FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3.10 python3.10-dev python3.10-venv \
    python3-pip \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    git \
    ffmpeg \
    curl \
    wget \
    nano \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

RUN python -m pip install --upgrade pip

ARG SRC_DIR=hisolo
WORKDIR /src
RUN pip install poetry==1.8.3
RUN poetry config virtualenvs.create false

COPY ./pyproject.toml ./pyproject.toml
RUN pip install -U pip
RUN poetry install --no-interaction --no-ansi

COPY . /src
RUN python -m pip install --no-build-isolation -e detectron2
RUN python -m pip install --no-build-isolation -e .

EXPOSE 8000

CMD ["tail", "-f", "/dev/null"]
