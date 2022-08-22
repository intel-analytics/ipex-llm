export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export NANO_TUTORIAL_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/tutorial/inference/tensorflow

set -e

export NUM_SHARDS=4
python $NANO_TUTORIAL_TEST_DIR/tensorflow_quantization.py
