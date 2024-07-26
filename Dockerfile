FROM nvidia/cuda:12.3.2-base-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /work
RUN apt-get update && apt-get -y --no-install-recommends install \
    sudo \
    vim \
    wget \
    build-essential \
    git \
    liblapacke \
    pkg-config \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-apt \
    python3-setuptools \
    python3-pip \
    unzip
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /work/
RUN pip install -r requirements.txt
COPY aimet_zoo_torch-1.5.0-py3-none-any.whl /work/
RUN pip install aimet_zoo_torch-1.5.0-py3-none-any.whl
COPY stuff/ImageNet2012-download-main.zip /work/
RUN unzip ImageNet2012-download-main.zip
COPY stuff/ILSVRC2012_img_val.tar /work/ImageNet2012-download-main
RUN bash /work/ImageNet2012-download-main/DB_arrangement.bash
