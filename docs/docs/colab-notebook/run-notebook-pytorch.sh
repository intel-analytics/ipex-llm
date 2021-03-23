#!/bin/bash

dir=${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart
pytorchFiles=("pytorch_lenet_mnist" "pytorch_distributed_lenet_mnist")
index=1
for f in $pytorchFiles
do
   
	filename="${dir}/${f}"
	echo "#${index} start example for ${f}"
	#timer
	start=$(date "+%s")

	${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${filename}
	sed -i "s/get_ipython()/#/g"  ${filename}.py
	sed -i "s/import os/#import os/g" ${filename}.py
	sed -i "s/import sys/#import sys/g" ${filename}.py
	sed -i 's/^[^#].*environ*/#&/g' ${filename}.py
	sed -i 's/^[^#].*__future__ */#&/g' ${filename}.py
	sed -i "s/_ = (sys.path/#_ = (sys.path/g" ${filename}.py
	sed -i 's/^[^#].*site-packages*/#&/g' ${filename}.py

	wget -nv $FTP_URI/analytics-zoo-data/mnist/train-labels-idx1-ubyte.gz -P ./dataset
	wget -nv $FTP_URI/analytics-zoo-data/mnist/train-images-idx3-ubyte.gz -P ./dataset
	wget -nv $FTP_URI/analytics-zoo-data/mnist/t10k-labels-idx1-ubyte.gz -P ./dataset
	wget -nv $FTP_URI/analytics-zoo-data/mnist/t10k-images-idx3-ubyte.gz -P ./dataset

	python ${filename}.py

	now=$(date "+%s")
	time=$((now-start))
	index=$((index+1))
	echo "Complete #${index} with time ${time}"	
done
