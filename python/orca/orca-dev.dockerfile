# Ubuntu 20.04 (focal)
# https://hub.docker.com/_/ubuntu/?tab=tags&name=focal
ARG ROOT_CONTAINER=ubuntu:focal

FROM $ROOT_CONTAINER

LABEL maintainer="The BigDL Authors"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

RUN if [ ! -z "$HTTPS_PROXY" ] || [ ! -z "$HTTP_PROXY" ]; then \
    echo "Setting Proxy..."; \
    echo "export http_proxy=${HTTP_PROXY}" >> /etc/profile.d/02-proxy.sh; \
    echo "export https_proxy=${HTTPS_PROXY}" >> /etc/profile.d/02-proxy.sh; \
    echo "export HTTP_PROXY=${HTTP_PROXY}" >> /etc/profile.d/02-proxy.sh; \
    echo "export HTTPS_PROXY=${HTTPS_PROXY}" >> /etc/profile.d/02-proxy.sh; \
    source /etc/profile.d/02-proxy.sh; \
    echo "Acquire::http::Proxy \"${HTTP_PROXY}\";" >> /etc/apt/apt.conf; \
    echo "Acquire::https::Proxy \"${HTTPS_PROXY}\";" >> /etc/apt/apt.conf; \
    cat /etc/apt/apt.conf; \
    fi

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
# https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease
ARG CONDA_MIRROR=https://github.com/conda-forge/miniforge/releases/latest/download
# ARG CONDA_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease


# ---- Miniforge installer ----
# Check https://github.com/conda-forge/miniforge/releases
# Package Manager and Python implementation to use (https://github.com/conda-forge/miniforge)
# We're using Mambaforge installer, possible options:
# - conda only: either Miniforge3 to use Python or Miniforge-pypy3 to use PyPy
# - conda + mamba: either Mambaforge to use Python or Mambaforge-pypy3 to use PyPy
# Installation: conda, mamba, pip
RUN set -x && \
    # Miniforge installer
    if [ -f "/etc/profile.d/02-proxy.sh" ];then \
    source /etc/profile.d/02-proxy.sh; \
    fi; \
    mkdir -p "${CONDA_DIR}" && \
    miniforge_arch=$(uname -m) && \
    miniforge_installer="Mambaforge-Linux-${miniforge_arch}.sh" && \
    wget "${CONDA_MIRROR}/${miniforge_installer}" && \
    /bin/bash "${miniforge_installer}" -f -b -p "${CONDA_DIR}" && \
    rm "${miniforge_installer}" && \
    # Conda configuration see https://conda.io/projects/conda/en/latest/configuration.html
    conda config --system --set auto_update_conda false && \
    conda config --system --set show_channel_urls true && \
    if [[ "${PYTHON_VERSION}" != "default" ]]; then mamba install --quiet --yes python="${PYTHON_VERSION}"; fi && \
    mamba list python | grep '^python ' | tr -s ' ' | cut -d ' ' -f 1,2 >> "${CONDA_DIR}/conda-meta/pinned" && \
    # Using conda to update all packages: https://github.com/mamba-org/mamba/issues/1092
    conda update --all --quiet --yes && \
    conda clean --all -f -y && \
    rm -rf "/root/.cache/yarn"

RUN if [ ! -z "$HTTPS_PROXY" ] || [ ! -z "$HTTP_PROXY" ]; then \
    conda config --set proxy_servers.http ${HTTP_PROXY}; \
    conda config --set proxy_servers.https ${HTTPS_PROXY}; \
    pip config set global.proxy ${HTTPS_PROXY}; \
    git config --global https.proxy ${HTTPS_PROXY}; \
    git config --global http.proxy ${HTTP_PROXY}; \
    # sed '<\/proxies>/i/' file # Add Proxy configuration to maven setting
    fi; \
    conda init bash


# Configure container startup
ENTRYPOINT ["/bin/bash"]

WORKDIR "${HOME}"
