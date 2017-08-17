#!/bin/bash

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

set -e

apt-get update && apt-get install -y python-dev build-essential libfreetype6-dev unzip
curl https://bootstrap.pypa.io/get-pip.py | python

PIP_PACKAGES=$(curl -f -s -H Metadata-Flavor:Google http://metadata/computeMetadata/v1/instance/attributes/PIP_PACKAGES || true)
NUM_EXECUTORS=$(curl -f -s -H Metadata-Flavor:Google http://metadata/computeMetadata/v1/instance/attributes/NUM_EXECUTORS || true)
NOTEBOOK_DIR="/root/notebooks"
TENSORBOARD_LOGDIR="/tmp/bigdl_summaries"

ROLE=$(/usr/share/google/get_metadata_value attributes/dataproc-role)
if [[ "${ROLE}" == 'Master' ]]; then
    pip install IPython==5.0 jupyter tensorboard virtualenv numpy scipy pandas scikit-learn matplotlib seaborn wordcloud opencv-python nltk
    wget https://repo1.maven.org/maven2/com/intel/analytics/bigdl/dist-spark-2.0.2-scala-2.11.8-linux64/0.2.0/dist-spark-2.0.2-scala-2.11.8-linux64-0.2.0-dist.zip -P /root/ 
    unzip /root/dist-spark-2.0.2-scala-2.11.8-linux64-0.2.0-dist.zip -d /root/
    rm /root/dist-spark-2.0.2-scala-2.11.8-linux64-0.2.0-dist.zip   
    [[ ! -d "~/.jupyter" ]] && mkdir -p ~/.jupyter
    [[ ! -d ${NOTEBOOK_DIR} ]] && mkdir -p $NOTEBOOK_DIR
    
    # uncomment the following two lines if you want the bigdl tutorials are in the notebook by default
    #git clone https://github.com/intel-analytics/BigDL-Tutorials.git /root/bigdl-tutorials
    #cp /root/bigdl-tutorials/notebooks/*/*.ipynb $NOTEBOOK_DIR/
    gsutil cp gs://dataproc-initial/examples/*.ipynb $NOTEBOOK_DIR/

    # package the virtual environment
    VENV="/root/venv"
    virtualenv $VENV
    virtualenv --relocatable $VENV
    . $VENV/bin/activate
    zip -q -r $VENV.zip $VENV
 
    echo "c.NotebookApp.token = u''" >> ~/.jupyter/jupyter_notebook_config.py
    if [ -z "${NUM_EXECUTORS}" ]; then
        NUM_EXECUTORS=$(yarn node -list | sed -n 's/Total Nodes:\(.*\)/\1/p')
    fi
    cat << EOF > "/usr/local/share/jupyter/kernels/python2/kernel.json"
{
  "argv": [
    "python",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "PySpark",
  "language": "python",
  "env": {
    "SPARK_HOME": "/usr/lib/spark",
    "PYTHONPATH": "/root/lib/bigdl-0.2.0-python-api.zip",
    "PYSPARK_SUBMIT_ARGS": "--master yarn --deploy-mode client --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./venv.zip/venv/bin/python --driver-memory 13g --executor-memory 13g --driver-cores 4 --executor-cores 4 --num-executors ${NUM_EXECUTORS} --properties-file /root/conf/spark-bigdl.conf --jars /root/lib/bigdl-SPARK_2.0-0.2.0-jar-with-dependencies.jar --archives /root/venv.zip --py-files /root/lib/bigdl-0.2.0-python-api.zip --conf spark.driver.extraClassPath=/root/lib/bigdl-SPARK_2.0-0.2.0-jar-with-dependencies.jar --conf spark.executor.extraClassPath=bigdl-SPARK_2.0-0.2.0-jar-with-dependencies.jar pyspark-shell"
  }
}
EOF
    # Lanuch tensorboard and Jupyter
    cat << EOF > "/usr/lib/systemd/system/tensorboard.service"
[Unit]
Description=Tensorboard Server
[Service]
Type=simple
ExecStart=/usr/local/bin/tensorboard --logdir=${TENSORBOARD_LOGDIR}
[Install]
WantedBy=multi-user.target
EOF
    cat << EOF > "/usr/lib/systemd/system/jupyter.service"
[Unit]
Description=Jupyter Server
[Service]
Type=simple
ExecStart=/usr/local/bin/jupyter notebook --notebook-dir=${NOTEBOOK_DIR} --ip=* --no-browser --allow-root
[Install]
WantedBy=multi-user.target
EOF
    chmod a+rw /usr/lib/systemd/system/tensorboard.service
    chmod a+rw /usr/lib/systemd/system/jupyter.service 
    systemctl daemon-reload
    systemctl enable tensorboard
    systemctl enable jupyter
    systemctl restart tensorboard
    systemctl restart jupyter
fi

# Install customized packages
if [ -n "${PIP_PACKAGES}" ]; then
    echo "Installing custom pip packages '$(echo ${PIP_PACKAGES} | tr ':' ' ')'"
    pip install $(echo ${PIP_PACKAGES} | tr ':' ' ')
fi
