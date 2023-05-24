#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export NANO_HOWTO_GUIDES_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/tutorial/notebook/training/pytorch

TORCH_VERSION_LESS_1_12=`python -c "from bigdl.nano.utils.pytorch import TORCH_VERSION_LESS_1_12; print(TORCH_VERSION_LESS_1_12)"`
AVX512_AVAILABLE=`lscpu | grep avx512`

set -e

# disable training with native pytorch bf16 amp if torch<1.12
if [[ $TORCH_VERSION_LESS_1_12 == True ]]
then
    sed -i "s/MyNano(precision='bf16').train()/#MyNano(precision='bf16').train()/" $NANO_HOWTO_GUIDES_TEST_DIR/accelerate_pytorch_training_bf16.ipynb
fi
# disable training with bf16 + ipex when avx512 is not avaliable;
# as it will automatically turn back to fp32 training,
# and no need to test
if [[ -z "$AVX512_AVAILABLE" ]]
then
    sed -i "s/MyNano(use_ipex=True, precision='bf16').train()/#MyNano(use_ipex=True, precision='bf16').train()/" $NANO_HOWTO_GUIDES_TEST_DIR/accelerate_pytorch_training_bf16.ipynb
fi

# It seems windows's bash cannot expand * wildcard
all_ipynb=`find "$NANO_HOWTO_GUIDES_TEST_DIR" -maxdepth 1 -name "*.ipynb"`

# the training dataset size is limited for testing purposes
sed -i "s/DataLoader(train_dataset/DataLoader(torch.utils.data.Subset(train_dataset, torch.randperm(len(train_dataset))[:5])/" $all_ipynb

# comment out the install commands
sed -i "s/!pip install/#!pip install/" $all_ipynb

# comment out the environment setting commands
sed -i "s/!source bigdl-nano-init/#!source bigdl-nano-init/" $all_ipynb

echo "Start testing"
start=$(date "+%s")

python -m pytest -s --nbmake --nbmake-timeout=600 --nbmake-kernel=python3 ${NANO_HOWTO_GUIDES_TEST_DIR}

now=$(date "+%s")
time=$((now-start))

echo "BigDL-Nano how-to guides tests for Training PyTorch finished."
echo "Time used: $time seconds"
