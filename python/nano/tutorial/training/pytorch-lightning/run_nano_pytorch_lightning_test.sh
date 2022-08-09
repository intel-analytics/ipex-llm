export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export NANO_TUTORIAL_TEST_DIR=${ANALYTICS_ZOO_ROOT}python/nano/tutorial/training/pytorch-lightning/

set -e

python $NANO_TUTORIAL_TEST_DIR/lightning_channel_last.py

python $NANO_TUTORIAL_TEST_DIR/lightning_cv_data_pipeline.py