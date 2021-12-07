conda create -y -n pytf1 python=3.7 && \
source activate pytf1 && \
conda install openvino-ie4py-ubuntu18 -c intel && \
pip install --no-cache-dir --upgrade pip && \
pip install --no-cache-dir --upgrade setuptools && \
pip install --no-cache-dir numpy==1.18.1 scipy && \
pip install --no-cache-dir pandas==1.0.3 && \
pip install --no-cache-dir scikit-learn matplotlib seaborn jupyter jupyterlab requests h5py==2.10.0 dm-tree==0.1.5 && \
pip uninstall -y -q tornado && \
pip install --no-cache-dir tornado && \
python3 -m ipykernel.kernelspec && \
pip install --no-cache-dir tensorboard && \
pip install --no-cache-dir jep && \
pip install --no-cache-dir cloudpickle && \
pip install --no-cache-dir opencv-python && \
pip install --no-cache-dir pyyaml && \
pip install --no-cache-dir redis && \
pip install --no-cache-dir ray[tune]==1.2.0 && \
pip install --no-cache-dir gym[atari]==0.17.1 && \
pip install --no-cache-dir Pillow==6.2 && \
pip install --no-cache-dir psutil aiohttp==3.7.0 && \
pip install --no-cache-dir setproctitle && \
pip install --no-cache-dir hiredis==1.1.0 && \
pip install --no-cache-dir async-timeout==3.0.1 && \
pip install --no-cache-dir py4j && \
pip install --no-cache-dir cmake==3.16.3 && \
pip install --no-cache-dir torch==1.7.1 torchvision==0.8.2 && \
pip install --no-cache-dir horovod==0.19.2 && \
pip install --no-cache-dir xgboost && \
pip install --no-cache-dir pyarrow && \
pip install --no-cache-dir opencv-python==4.2.0.34 && \
pip install --no-cache-dir aioredis==1.1.0 && \
pip install --no-cache-dir tensorflow==1.15.0 && \
pip install --no-cache-dir tf_slim && \
pip install --no-cache-dir tensorflow-datasets==3.2.0 && \
pip install --no-cache-dir mxnet==1.6.0 && \
source deactivate
