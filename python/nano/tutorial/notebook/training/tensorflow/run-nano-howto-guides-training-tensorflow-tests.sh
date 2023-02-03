#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export NANO_HOWTO_GUIDES_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/tutorial/notebook/training/tensorflow

set -e

# the number of epoch to run is limited for testing purposes
sed -i 's/epochs=10/epochs=1/' $NANO_HOWTO_GUIDES_TEST_DIR/*.ipynb
sed -i 's/range(10)/range(1)/' $NANO_HOWTO_GUIDES_TEST_DIR/tensorflow_training_bf16.ipynb

# the number of batches to run is limited for testing purposes
sed -i "s/steps_per_epoch=(ds_info.splits\['train'].num_examples \/\/ 32)/steps_per_epoch=(ds_info.splits\['train'].num_examples \/\/ 32 \/\/ 10)/" $NANO_HOWTO_GUIDES_TEST_DIR/accelerate_tensorflow_training_multi_instance.ipynb

# comment out the install commands
sed -i 's/!pip install/#!pip install/' $NANO_HOWTO_GUIDES_TEST_DIR/*.ipynb

# comment out the environment setting commands
sed -i 's/!source bigdl-nano-init/#!source bigdl-nano-init/' $NANO_HOWTO_GUIDES_TEST_DIR/*.ipynb

echo 'Start testing'
start=$(date "+%s")

python -m pytest -s --nbmake --nbmake-timeout=600 --nbmake-kernel=python3 ${NANO_HOWTO_GUIDES_TEST_DIR}
now=$(date "+%s")
time=$((now-start))

echo "BigDL-Nano how-to guides tests for Training TensorFlow finished."
echo "Time used: $time seconds"