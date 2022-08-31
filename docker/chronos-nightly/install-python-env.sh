conda create -y -n chronos python=3.7 setuptools=58.0.4 && \
source activate chronos && \
pip install --no-cache-dir prophet==1.1.0 &&\
pip install --no-cache-dir pmdarima==1.8.4 && \
pip install --no-cache-dir neural_compressor==1.8.1 && \
pip install --no-cache-dir onnxruntime==1.6.0 && \
pip install --no-cache-dir tsfresh==0.17.0 && \
pip install --no-cache-dir numpy==1.19.5 && \
pip install --no-cache-dir ray==1.9.2 ray[tune]==1.9.2 ray[default]==1.9.2 && \
pip install --no-cache-dir pyarrow==6.0.1 && \
pip install --no-cache-dir --pre bigdl-nano[pytorch] && \
pip install --no-cache-dir --pre bigdl-nano[tensorflow] && \
pip install --no-cache-dir --pre bigdl-chronos[all] && \
pip install --no-cache-dir torchmetrics==0.7.2 && \
pip install --no-cache-dir scipy==1.5.4 && \
pip install --no-cache-dir prometheus_pandas==0.3.1 && \
pip install --no-cache-dir xgboost==1.2.0 && \
pip install --no-cache-dir jupyter==1.0.0




