echo "#2 start example for pytorch minist"
#timer
start=$(date "+%s")	

${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/pytorch_lenet_mnist
sed -i "s/get_ipython()/#/g"  ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/pytorch_lenet_mnist.py
sed -i "s/import os/#import os/g" ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/pytorch_lenet_mnist.py
sed -i "s/import sys/#import sys/g" ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/pytorch_lenet_mnist.py
sed -i 's/^[^#].*environ*/#&/g' ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/pytorch_lenet_mnist.py
sed -i 's/^[^#].*__future__ */#&/g' ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/pytorch_lenet_mnist.py
sed -i "s/_ = (sys.path/#_ = (sys.path/g" ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/pytorch_lenet_mnist.py
sed -i 's/^[^#].*site-packages*/#&/g' ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/pytorch_lenet_mnist.py

wget -nv $FTP_URI/analytics-zoo-data/mnist/train-labels-idx1-ubyte.gz -P ./dataset
wget -nv $FTP_URI/analytics-zoo-data/mnist/train-images-idx3-ubyte.gz -P ./dataset
wget -nv $FTP_URI/analytics-zoo-data/mnist/t10k-labels-idx1-ubyte.gz -P ./dataset
wget -nv $FTP_URI/analytics-zoo-data/mnist/t10k-images-idx3-ubyte.gz -P ./dataset

python ${ANALYTICS_ZOO_HOME}/docs/docs/colab-notebook/orca/quickstart/pytorch_lenet_mnist.py

now=$(date "+%s")	
time2=$((now-start))	

