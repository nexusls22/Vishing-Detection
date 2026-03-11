# Base image with CUDA
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base

# Env for auto detect
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 100 \
    && update-alternatives --set python3 /usr/bin/python3.11 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

#Update pip
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch with CUDA Support (fallback to cpu without gpu)
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Backend directory
COPY . /Backend
WORKDIR /workspace

# Start up script searching for GPU
RUN echo '#!/bin/bash\npython -c "import torch; print(f\"GPU available: {torch.cuda.is_available()}\")"\nexec "$@"' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]

RUN pip install ipykernel jupyter && \
    python -m ipykernel install --name "vishing-env" --display-name "Python (Vishing Docker)"