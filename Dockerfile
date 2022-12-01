FROM nvidia/cuda:11.4.0-devel-ubuntu20.04
LABEL maintainer="haruki@hacarus.com"
ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]
ENV HOME=/home \
    WORKDIR=/work
RUN mkdir -p $WORKDIR
WORKDIR $WORKDIR

# for CUDA
RUN apt-key adv --fetch-keys https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-get update \
 && apt-get install --no-install-recommends -y fonts-ipaexfont libglib2.0-0 git gcc vim pip curl wget \
 # for opencv
 && apt-get install --no-install-recommends -y \
    build-essential \
    libsm-dev \
    libxrender-dev \
    libxext-dev \
    libgl1-mesa-dev \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    python-dev \
    python3-venv \
    libssl-dev \
    libffi-dev \
 # for pyenv
 && apt-get install --no-install-recommends -y \
    gcc \
    make \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    llvm \
    libncurses5-dev \
    xz-utils \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
 # openslide
 && apt-get install --no-install-recommends -y python3-openslide \
 && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# install CUDA Toolkit for CuPy
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
#  && dpkg -i cuda-keyring_1.0-1_all.deb \
#  && apt-get update \
#  && apt-get -y install cuda

RUN wget https://www.python.org/ftp/python/3.8.6/Python-3.8.6.tar.xz \
 && tar xJf Python-3.8.6.tar.xz \
 && cd Python-3.8.6 \
 && ./configure \
 && make \
 && make install \
 && cd ../ \
 && rm -rf Python*

# poetry
COPY pyproject.toml poetry.lock ./
ENV POETRY_HOME=/usr/local/poetry \
    POETRY_VERSION=1.1.13
RUN /usr/local/bin/python3.8 -m pip install --upgrade pip \
 && curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python3 - \
 && echo 'export PATH="/usr/local/poetry/bin:$PATH"' >> $HOME/.bashrc \
 && /usr/local/poetry/bin/poetry config virtualenvs.create false \
 && /usr/local/poetry/bin/poetry install --no-root

# install original packages with separate cache
COPY modules ./modules
RUN /usr/local/poetry/bin/poetry install \
 && chmod -R 777 $POETRY_HOME \
 && chmod -R 777 $HOME

EXPOSE 22
EXPOSE 8888

COPY entrypoint.sh ./entrypoint.sh
ENTRYPOINT /bin/bash entrypoint.sh
