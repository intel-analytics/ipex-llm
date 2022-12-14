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
cd ../../tutorial/xshards/notebooks

echo "#1 Running run-tabular_playground_series"
#timer
start=$(date "+%s")

if [ -f ${BIGDL_ROOT}/python/orca/tutorial/xshards/train.csv ]
then
    echo "train.csv already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/xshards/train.csv -P ${BIGDL_ROOT}/python/orca/tutorial/xshards/
fi

filename=tabular_playground_series
${BIGDL_ROOT}/python/orca/dev/colab-notebook/ipynb2py.sh ${filename}
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
	sed -i 's/batch_size = 32/batch_size = 320/g' ${filename}.py
	sed -i 's/epochs = 30/epochs = 1/g' ${filename}.py

python ${filename}.py

now=$(date "+%s")
time1=$((now - start))
