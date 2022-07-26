#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export NANO_RAY_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/test/ray
export PYTHONPATH=$PYTHONPATH:/home/wym/my_ray/BigDL/python/nano

set -e

# ipex is not installed here. Any tests needs ipex should be moved to next pytest command.
echo "# Start testing"
start=$(date "+%s")
python -m pytest -s ${NANO_RAY_TEST_DIR}/test_ray_trainer.py ${NANO_RAY_TEST_DIR}/test_ray_lite.py

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-nano tests finished"
echo "Time used:$time seconds"
