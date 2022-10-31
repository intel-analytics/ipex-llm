export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export NANO_TUTORIAL_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/tutorial/training/pytorch

TORCH_VERSION=`python -c "from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_12; print(TORCH_VERSION_LESS_1_12)"`

set -e

echo " Start testing PyTorch CV data pipeline tutorial"
sed -i s/:-val_size/:5/ $NANO_TUTORIAL_TEST_DIR/pytorch_cv_data_pipeline.py
sed -i s/-val_size:/-5:/ $NANO_TUTORIAL_TEST_DIR/pytorch_cv_data_pipeline.py
python $NANO_TUTORIAL_TEST_DIR/pytorch_cv_data_pipeline.py

echo " Start testing PyTorch IPEX training tutorial"
sed -i s/:-val_size/:5/ $NANO_TUTORIAL_TEST_DIR/pytorch_train_ipex.py
sed -i s/-val_size:/-5:/ $NANO_TUTORIAL_TEST_DIR/pytorch_train_ipex.py
python $NANO_TUTORIAL_TEST_DIR/pytorch_train_ipex.py

echo " Start testing PyTorch multi-instance training tutorial"
sed -i s/:-val_size/:5/ $NANO_TUTORIAL_TEST_DIR/pytorch_train_multi_instance.py
sed -i s/-val_size:/-5:/ $NANO_TUTORIAL_TEST_DIR/pytorch_train_multi_instance.py
python $NANO_TUTORIAL_TEST_DIR/pytorch_train_multi_instance.py

echo " Start testing PyTorch BFloat16 training tutorial"
sed -i s/:-val_size/:5/ $NANO_TUTORIAL_TEST_DIR/pytorch_train_bf16.py
sed -i s/-val_size:/-5:/ $NANO_TUTORIAL_TEST_DIR/pytorch_train_bf16.py
if [ $TORCH_VERSION == True ]
then
    sed -i s/MyNano\(precision=\'bf16\'\)\.train\(\)//  $NANO_TUTORIAL_TEST_DIR/pytorch_train_bf16.py
fi
python $NANO_TUTORIAL_TEST_DIR/pytorch_train_bf16.py
