#!/usr/bin/env bash

set -e

$CONDA/bin/conda create -n benchmark-resnet-50 -y python==3.7.10 setuptools==58.0.4
$CONDA/bin/activate benchmark-resnet-50

bash $ANALYTICS_ZOO_ROOT/python/nano/dev/build_and_install.sh linux default false pytorch

echo "Nano_Perf: Running PyTorch Baseline"
python pytorch-cat-dog.py --batch_size 32 --name "PyTorch Baseline"

source bigdl-nano-init
echo "Nano_Perf: Running Nano default"
python pytorch-cat-dog.py --batch_size 32 --name "Nano default env"
source bigdl-nano-unset-env

source bigdl-nano-init
echo "Nano_Perf: Running Nano default with ipex"
python pytorch-cat-dog.py --use_ipex true --batch_size 32 --name "Nano default env with ipex"
source bigdl-nano-unset-env

source bigdl-nano-init
echo "Nano_Perf: Running Nano default with ipex, nano data"
python pytorch-cat-dog.py --nano_data true --use_ipex true --batch_size 32 --name "Nano default env with ipex, nano data"
source bigdl-nano-unset-env

source bigdl-nano-init
echo "Nano_Perf: Running Nano default with ipex 4 processes"
python pytorch-cat-dog.py --nproc 4 --nano_data true --use_ipex true --batch_size 32 --name "Nano default env with ipex, nano data, 4 process"
source bigdl-nano-unset-env

source $CONDA/bin/deactivate
$CONDA/bin/conda remove -n benchmark-resnet-50 --all