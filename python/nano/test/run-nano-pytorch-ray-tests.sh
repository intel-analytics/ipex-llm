#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export NANO_RAY_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/test/ray
export PYTHONPATH=$PYTHONPATH:/home/wym/my_ray/BigDL/python/nano

set -e

# ipex is not installed here. Any tests needs ipex should be moved to next pytest command.
echo "# Start testing"
start=$(date "+%s")

python -m pytest -s ${NANO_RAY_TEST_DIR}/test_ray_trainer.py ${NANO_RAY_TEST_DIR}/test_torch_nano_ray.py
version=$(python -c "import torch;print(torch.__version__)")
if [ ${version} != "2.0.0+cu117" ];then  # workaround as we have not upgrade pl
    python -m pytest -s ${ANALYTICS_ZOO_ROOT}/python/nano/test/pytorch/tests/train/trainer/test_scale_lr.py -k 'ray'
fi

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-nano tests finished"
echo "Time used:$time seconds"
