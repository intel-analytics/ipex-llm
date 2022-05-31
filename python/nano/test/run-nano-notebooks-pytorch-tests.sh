#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export PYTORCH_NANO_NOTEBOOKS_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/notebooks/pytorch
export MAX_STEPS=10

wget -nv ${FTP_URI}/analytics-zoo-data/cifar-10-python.tar.gz -P ${PYTORCH_NANO_NOTEBOOKS_DIR}/cifar10/

set -e

# Add your notebook to the Appropriate list after made sure the test environment meets your needs
# Example: new notebook test.ipynb in notebooks/pytorch/test-dataset
# train_notebooks = "${PYTORCH_NANO_NOTEBOOKS_DIR}/cifar10/nano-trainer-example.ipynb \
#                    ${PYTORCH_NANO_NOTEBOOKS_DIR}/test-dataset/test.ipynb "
train_notebooks="${PYTORCH_NANO_NOTEBOOKS_DIR}/cifar10/nano-trainer-example.ipynb "
inference_notebooks="${PYTORCH_NANO_NOTEBOOKS_DIR}/cifar10/nano-inference-example.ipynb "

function read_dir() {
for file in `ls $1`
do
 if [ -d $1"/"$file ]
 then
 read_dir $1"/"$file
 else
 if [ "${file##*.}"x = "ipynb"x ] && [[ ! "$train_notebooks" =~ "$1/$file" ]] && [[ ! "$inference_notebooks" =~ "$1/$file" ]]
 then
 echo "$file is added but the notebook test is not config"
 exit 1
 fi
 fi
done
}

read_dir ${PYTORCH_NANO_NOTEBOOKS_DIR}

echo "# Start Testing ${train_notebooks}"
start=$(date "+%s")

python -m pytest --nbmake --nbmake-timeout=1000 --nbmake-kernel=python3 ${train_notebooks}

now=$(date "+%s")
time=$((now-start))

echo "# Start Testing ${inference_notebooks}"
start=$(date "+%s")

python -m pytest --nbmake --nbmake-timeout=1000 --nbmake-kernel=python3 ${inference_notebooks}

now=$(date "+%s")
time=$((now-start))

