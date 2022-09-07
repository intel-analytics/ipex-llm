export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export NANO_TUTORIAL_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/tutorial/training/pytorch-lightning

TORCH_VERSION=`python -c "from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_12; print(TORCH_VERSION_LESS_1_12)"`

set -e

sed -i s/Trainer\(max_epochs=5,\ channels_last=True\)/Trainer\(max_epochs=5,\ fast_dev_run=True,\ channels_last=True\)/ $NANO_TUTORIAL_TEST_DIR/lightning_channel_last.py
python $NANO_TUTORIAL_TEST_DIR/lightning_channel_last.py

sed -i s/Trainer\(max_epochs=5\)/Trainer\(max_epochs=5,\ fast_dev_run=True\)/ $NANO_TUTORIAL_TEST_DIR/lightning_cv_data_pipeline.py
python $NANO_TUTORIAL_TEST_DIR/lightning_cv_data_pipeline.py

sed -i s/max_epochs=5,/max_epochs=5,\ fast_dev_run=True,/ $NANO_TUTORIAL_TEST_DIR/lightning_train_ipex.py
python $NANO_TUTORIAL_TEST_DIR/lightning_train_ipex.py

sed -i s/max_epochs=5,/max_epochs=5,\ fast_dev_run=True,/ $NANO_TUTORIAL_TEST_DIR/lightning_train_multi_instance.py
python $NANO_TUTORIAL_TEST_DIR/lightning_train_multi_instance.py

sed -i s/max_epochs=5,/max_epochs=5,\ fast_dev_run=True,/ $NANO_TUTORIAL_TEST_DIR/lightning_train_bf16.py
if [ $TORCH_VERSION == True ]
then
    sed -i '106,108d' $NANO_TUTORIAL_TEST_DIR/lightning_train_bf16.py
fi
python $NANO_TUTORIAL_TEST_DIR/lightning_train_bf16.py
