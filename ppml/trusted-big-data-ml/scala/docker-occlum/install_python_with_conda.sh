#!/bin/bash
set -e

# Install python and dependencies to specified position
wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh
chmod +x Miniconda3-4.5.4-Linux-x86_64.sh
[ -d miniconda ] || bash ./Miniconda3-4.5.4-Linux-x86_64.sh -b -p /opt/miniconda
/opt/miniconda/bin/conda create --prefix /opt/python-occlum -y python=3.8.10 numpy=1.21.5 scipy=1.7.3 scikit-learn=1.0 pandas=1.3 Cython
