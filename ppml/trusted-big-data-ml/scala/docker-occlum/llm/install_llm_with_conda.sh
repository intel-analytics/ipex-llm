#!/bin/bash
set -e

# Install python and dependencies to specified position

[ -f Miniconda3-latest-Linux-x86_64.sh ] || wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
[ -d miniconda ] || bash ./Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda
/opt/miniconda/bin/conda create --prefix /opt/python-occlum -y python=3.9.11 numpy=1.21.5 scipy=1.7.3 scikit-learn=1.0 pandas=1.3 Cython

export PATH=$PATH:/opt/miniconda/bin
source activate
conda activate /opt/python-occlum
pip install synapseml==0.10.2
mkdir -p /opt/occlum_spark/data1/
cd /opt/occlum_spark/data1/
pip3 install fschat -i https://pypi.tuna.tsinghua.edu.cn/simple/
git clone -b dev-2023-08-01 https://github.com/analytics-zoo/FastChat.git
cd FastChat
pip3 install --upgrade pip  # enable PEP 660 support
pip3 install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple/

pip install --pre --upgrade bigdl-llm[all] -i https://pypi.tuna.tsinghua.edu.cn/simple/
