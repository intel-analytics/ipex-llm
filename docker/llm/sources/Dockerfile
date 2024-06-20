FROM ubuntu:22.04

# Add deb-src entries to the sources.list
RUN echo "deb-src http://archive.ubuntu.com/ubuntu/ focal main restricted universe multiverse" >> /etc/apt/sources.list

# Update package lists and install dpkg-dev
RUN apt-get update && apt-get install -y dpkg-dev

# Create a temporary directory, adjust permissions, and download source code for the specified packages
RUN mkdir -p /usr/local/src/git-source && \
    chown _apt:root /usr/local/src/git-source && \
    cd /usr/local/src/git-source && \
    apt-get source \
    git \
    gnupg \
    numactl \
    wget \
    software-properties-common