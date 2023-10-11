FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y tzdata && \
    apt-get install -y \
    git \
    vim \
    tmux \
    libsndfile-dev \
    apt-utils \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    python3-tk && \
    apt clean autoclean && \
    apt autoremove -y

RUN ln -fns /usr/bin/python3.8 /usr/bin/python && \
    ln -fns /usr/bin/python3.8 /usr/bin/python3 && \
    ln -fns /usr/bin/pip3 /usr/bin/pip

ENV TZ Asia/Tokyo
ENV LANG ja_JP.UTF-8
RUN apt-get -y install language-pack-ja-base language-pack-ja

RUN pip install --upgrade pip && \
    pip install --upgrade setuptools

RUN python3 -m pip install --user numpy scipy matplotlib sklearn pandas plotly tqdm torch transformers tensorflow

WORKDIR /work
COPY src/ /work/src/
COPY data/ /work/data/
#COPY results/ /work/results/
#COPY models/ /work/models/
#COPY scripts/ /work/scripts/
