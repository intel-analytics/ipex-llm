wget https://raw.githubusercontent.com/intel-analytics/BigDL/main/python/requirements/orca/requirements_automl.txt
# python version passed as the first argument
# default version: 3.9
conda create -y -n bigdl python=$1
source activate bigdl
pip install --no-cache-dir -r requirements_automl.txt
