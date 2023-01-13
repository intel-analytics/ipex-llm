#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


set -ex

export FTP_URI=$FTP_URI
export PYSPARK_PYTHON=python
export PYSPARK_DRIVER_PYTHON=python

ray stop -f

cd "`dirname $0`"
cd ../../tutorial/xshards/images

echo "#1 image classification of tensorflow"
#timer
start=$(date "+%s")

if [ -f ${BIGDL_ROOT}/python/orca/tutorial/xshards/dogs-vs-cats/ ]
then
    echo "dogs-vs-cats already exists"
else
    wget -r -np -nH --cut-dirs=2 $FTP_URI/analytics-zoo-data/xshards/dogs-vs-cats/ -P ${BIGDL_ROOT}/python/orca/tutorial/xshards/
fi

filename=image_classification_tf
${BIGDL_ROOT}/python/orca/dev/colab-notebook/ipynb2py.sh ${filename}
	sed -i "s/get_ipython()/#/g"  ${filename}.py
	sed -i "s/EPOCHS = 50/EPOCHS = 1/g" ${filename}.py

python ${filename}.py
echo "finished notebook image_classification_tf.ipynb"
now=$(date "+%s")
time1=$((now - start))


echo "#2 image classification of pytorch"
#timer
start=$(date "+%s")

if [ -f ${BIGDL_ROOT}/python/orca/tutorial/xshards/dogs-vs-cats/ ]
then
    echo "dogs-vs-cats already exists"
else
    wget -r -np -nH --cut-dirs=2  $FTP_URI/analytics-zoo-data/xshards/dogs-vs-cats/ -P ${BIGDL_ROOT}/python/orca/tutorial/xshards/
fi

filename=image_classification_py
${BIGDL_ROOT}/python/orca/dev/colab-notebook/ipynb2py.sh ${filename}
	sed -i "s/get_ipython()/#/g"  ${filename}.py
	sed -i "s/EPOCHS = 50/EPOCHS = 1/g" ${filename}.py

python ${filename}.py
echo "finished notebook image_classification_py.ipynb"
now=$(date "+%s")
time2=$((now - start))

echo "#3 image segmentation of tensorflow"
start=$(date "+%s")
if [ -f ${BIGDL_ROOT}/python/orca/tutorial/xshards/petsdata/ ]
then
    echo "petsdata already exists"
else
    wget -r -np -nH --cut-dirs=2  $FTP_URI/analytics-zoo-data/xshards/petsdata/ -P ${BIGDL_ROOT}/python/orca/tutorial/xshards/
fi

filename=segmentation_tf
${BIGDL_ROOT}/python/orca/dev/colab-notebook/ipynb2py.sh ${filename}
	sed -i "s/get_ipython()/#/g"  ${filename}.py
	sed -i "s/EPOCHS = 50/EPOCHS = 1/g" ${filename}.py

python ${filename}.py
cat ${filename}.py
echo "finished notebook segmentation_tf.ipynb"
now=$(date "+%s")
time3=$((now - start))


echo "#1 Running image_classification_tf time used: $time1 seconds"
echo "#2 Running image_classification_py time used: $time2 seconds"
echo "#3 Running segmentation_tf time used: $time3 seconds"
