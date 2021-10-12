#!/bin/bash

# orca test
echo "orca test start"

dir=${ANALYTICS_ZOO_HOME}/python/orca/colab-notebook/quickstart
pytorchFiles=("pytorch_lenet_mnist_data_creator_func" "pytorch_lenet_mnist" "pytorch_distributed_lenet_mnist" "autoestimator_pytorch_lenet_mnist" "autoxgboost_regressor_sklearn_boston")
index=1

set -e

for f in "${pytorchFiles[@]}"
do
   
	filename="${dir}/${f}"
	echo "#${index} start example for ${f}"
	#timer
	start=$(date "+%s")

	${ANALYTICS_ZOO_HOME}/python/orca/dev/colab-notebook/ipynb2py.sh ${filename}
	sed -i "s/get_ipython()/#/g"  ${filename}.py
	sed -i "s/import os/#import os/g" ${filename}.py
	sed -i "s/import sys/#import sys/g" ${filename}.py
	sed -i 's/^[^#].*environ*/#&/g' ${filename}.py
	sed -i 's/^[^#].*__future__ */#&/g' ${filename}.py
	sed -i "s/_ = (sys.path/#_ = (sys.path/g" ${filename}.py
	sed -i "s/.append/#.append/g" ${filename}.py
	sed -i 's/^[^#].*site-packages*/#&/g' ${filename}.py
	sed -i 's/version_info/#version_info/g' ${filename}.py
	sed -i 's/python_version/#python_version/g' ${filename}.py
	sed -i 's/epochs=30/epochs=1/g' ${filename}.py

	python ${filename}.py

	now=$(date "+%s")
	time=$((now-start))
	echo "Complete #${index} with time ${time} seconds"	
	index=$((index+1))
done

