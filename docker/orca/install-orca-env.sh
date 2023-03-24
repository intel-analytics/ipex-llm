wget https://raw.githubusercontent.com/intel-analytics/BigDL/main/python/requirements/orca/requirements_automl.txt && \
conda create -y -n bigdl python=3.7 && \
source activate bigdl && \
pip install --no-cache-dir -r requirements_automl.txt
