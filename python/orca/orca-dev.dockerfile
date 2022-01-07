# Ubuntu 20.04 (focal)
# https://hub.docker.com/_/ubuntu/?tab=tags&name=focal
ARG ROOT_CONTAINER=ubuntu:focal

FROM $ROOT_CONTAINER

LABEL maintainer="The BigDL Authors"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root


# Install OS dependencies for bigdl orca
# - tini is installed as a helpful container entrypoint that reaps zombie
#   processes and such of the actual executable we want to start, see
#   https://github.com/krallin/tini#why-tini for details.
# - apt-get upgrade is run to patch known vulnerabilities in apt-get packages as
#   the ubuntu base image is rebuilt too seldom sometimes (less than once a month)
ENV DEBIAN_FRONTEND noninteractive
RUN apt update --yes && \
    apt upgrade --yes && \
    apt install --yes \
    --no-install-recommends \
    tini \
    wget \
    sudo \
    locales \
    fonts-liberation \
    openssh-server \
    git \
    htop apt-utils ca-certificates vim \
    openjdk-8-jdk \
    scala \
    maven && \
    apt clean && rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen; 

# Configure environment
ENV CONDA_DIR=/opt/conda \
    SHELL=/bin/bash \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8
ENV PATH="${CONDA_DIR}/bin:${PATH}:/usr/local/scala/bin" \
    HOME="/root" \
    JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64


# Enable prompt color in the skeleton .bashrc
# hadolint ignore=SC2016
RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /etc/skel/.bashrc && \
   # Add call to conda init script see https://stackoverflow.com/a/58081608/4413446
   echo 'eval "$(command conda shell.bash hook 2> /dev/null)"' >> /etc/skel/.bashrc && \
   sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' ${HOME}/.bashrc 


ARG PYTHON_VERSION=default
# Install conda and check the sha256 sum provided on the download site
WORKDIR /tmp

# CONDA_MIRROR is a mirror prefix to speed up downloading
# For example, people from mainland China could set it as
# https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda
ARG CONDA_MIRROR=https://repo.anaconda.com/miniconda
# ARG CONDA_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda

# ---- Miniconda installer ----
# Check https://docs.conda.io/en/latest/miniconda.html
# Package Manager and Python implementation to use (https://docs.conda.io/en/latest/miniconda.html)
RUN set -x && \
    # Miniconda3 installer
    mkdir -p "${CONDA_DIR}" && \
    miniconda_arch=$(uname -m) && \
    miniconda_installer="Miniconda3-latest-Linux-${miniconda_arch}.sh" && \
    wget "${CONDA_MIRROR}/${miniconda_installer}" && \
    /bin/bash "${miniconda_installer}" -f -b -p "${CONDA_DIR}" && \
    rm "${miniconda_installer}" && \
    # Conda configuration see https://conda.io/projects/conda/en/latest/configuration.html
    conda config --system --set auto_update_conda false && \
    conda config --system --set show_channel_urls true && \
    if [[ "${PYTHON_VERSION}" != "default" ]]; then conda install --quiet --yes python="${PYTHON_VERSION}"; fi && \
    conda list python | grep '^python ' | tr -s ' ' | cut -d ' ' -f 1,2 >> "${CONDA_DIR}/conda-meta/pinned" && \
    # Using conda to update all packages: https://github.com/mamba-org/mamba/issues/1092
    conda update --all --quiet --yes && \
    conda clean --all -f -y && \
    rm -rf "/root/.cache/yarn" && \
    conda init bash


# Configure container startup
ENTRYPOINT ["/bin/bash"]

WORKDIR "${HOME}"
