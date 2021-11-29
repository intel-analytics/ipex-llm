#!/usr/bin/env bash
clear_up() {
  echo "Clearing up environment. Uninstalling analytics-zoo"
  pip uninstall -y bigdl-dllib
  pip uninstall -y bigdl-orca
  pip uninstall -y pyspark
}

set -e

echo "#1 start test for tf2_lenet_mnist.ipynb"
#replace '!pip install --pre' to '#pip install --pre', here we test pr with built whl package. In nightly-build job, we test only use "ipython notebook" for pre-release Analytics Zoo
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/python/orca/dev/colab-notebook/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/python/orca/colab-notebook/quickstart/tf2_keras_lenet_mnist
sed -i '/get_ipython/s/^/#/' $ANALYTICS_ZOO_HOME/python/orca/colab-notebook/quickstart/tf2_keras_lenet_mnist.py
python ${ANALYTICS_ZOO_HOME}/python/orca/colab-notebook/quickstart/tf2_keras_lenet_mnist.py

exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "tf2_lenet_mnist failed"
  exit $exit_status
fi

now=$(date "+%s")
time1=$((now - start))

echo "#2 start test for ncf_dataframe.ipynb"
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/python/orca/dev/colab-notebook/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/python/orca/colab-notebook/quickstart/ncf_dataframe
sed -i '/get_ipython/s/^/#/' ${ANALYTICS_ZOO_HOME}/python/orca/colab-notebook/quickstart/ncf_dataframe.py
python ${ANALYTICS_ZOO_HOME}/python/orca/colab-notebook/quickstart/ncf_dataframe.py

exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "ncf_dataframe failed"
  exit $exit_status
fi

now=$(date "+%s")
time2=$((now - start))

echo "#1 tf2_keras_lenet_mnist time used: $time1 seconds"
echo "#2 ncf_dataframe time used: $time2 seconds"
