#!/usr/bin/env bash

set -e

PY_VERSION_SUFFIX=${PY_VERSION_3:+3}
PYTHON=python${PY_VERSION_SUFFIX}
PIP=pip${PY_VERSION_SUFFIX}

apt-get install -y ${PYTHON}-minimal
apt-get install -y build-essential ${PYTHON} ${PYTHON}-setuptools ${PYTHON}-dev ${PYTHON}-pip

${PIP} install --upgrade setuptools
${PIP} install numpy scipy
${PIP} install --no-binary pandas -I pandas
${PIP} install scikit-learn matplotlib seaborn jupyter wordcloud moviepy requests h5py opencv-python tensorflow==1.10.0

if [[ "$PYTHON" != "python3" ]]; then
    ${PYTHON} -m pip install -U ipykernel
    ${PYTHON} -m ipykernel install --user
else
    ln -s /usr/bin/python3 /usr/bin/python
    ln -s /usr/bin/pip3 /usr/local/bin/pip
    #Fix tornado await process
    pip uninstall -y -q tornado
    pip install tornado==5.1.1
fi

${PYTHON} -m ipykernel.kernelspec
