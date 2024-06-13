FROM ubuntu:22.04

RUN apt-get update -y && apt-get install -y --no-install-recommends --fix-missing \
    git \
    gnupg \
    numactl \
    wget && \
    rm -rf /var/lib/apt/lists/*