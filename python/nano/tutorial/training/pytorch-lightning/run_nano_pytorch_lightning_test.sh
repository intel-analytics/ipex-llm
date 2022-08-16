export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export NANO_TUTORIAL_TEST_DIR=${ANALYTICS_ZOO_ROOT}python/nano/tutorial/training/pytorch-lightning

set -e

sed -i s/Trainer\(max_epochs=5,\ channels_last=True\)/Trainer\(max_epochs=5,\ fast_dev_run=True,\ channels_last=True\)/ $NANO_TUTORIAL_TEST_DIR/lightning_channel_last.py
python $NANO_TUTORIAL_TEST_DIR/lightning_channel_last.py

sed -i s/Trainer\(max_epochs=5\)/Trainer\(max_epochs=5,\ fast_dev_run=True\)/ $NANO_TUTORIAL_TEST_DIR/lightning_cv_data_pipeline.py
python $NANO_TUTORIAL_TEST_DIR/lightning_cv_data_pipeline.py