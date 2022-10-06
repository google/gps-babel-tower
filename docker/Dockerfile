FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

COPY ./models /models

RUN apt-get update && apt-get install -y build-essential wget curl && rm -rf /var/lib/apt/lists/*
WORKDIR /root

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

SHELL ["/bin/bash", "--login", "-c"]
# Create the environment:
RUN conda create -n python37 python=3.7
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "python37", "/bin/bash", "-c"]

# force no cache for all command below
# ARG CACHEBUST=1
RUN pip3 install tensorflow gps-babel-tower[tensorflow,torch,nlp,image,google]
RUN python -c "import gps_babel_tower"

