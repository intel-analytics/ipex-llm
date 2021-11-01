# Install Miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh
chmod +x Miniconda3-4.5.4-Linux-x86_64.sh
./Miniconda3-4.5.4-Linux-x86_64.sh -b -f -p /usr/local

conda create -y -n pytf1 python=3.7 && \
source activate pytf1 && \
pip install --no-cache-dir --upgrade pip && \
pip install --no-cache-dir --upgrade setuptools && \
pip install --no-cache-dir numpy==1.18.1 scipy && \
pip install --no-cache-dir pandas==1.0.3 && \
pip install --no-cache-dir scikit-learn matplotlib seaborn jupyter jupyterlab requests h5py && \
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
pip install --no-cache-dir Pillow==6.2 && \
pip install --no-cache-dir psutil aiohttp && \
pip install --no-cache-dir py4j && \
pip install --no-cache-dir cmake==3.16.3 && \
pip install --no-cache-dir torch==1.7.1 torchvision==0.8.2 && \
pip install --no-cache-dir horovod==0.19.2 && \
pip install --no-cache-dir xgboost && \
pip install --no-cache-dir pyarrow && \
pip install opencv-python==4.2.0.34 && \
pip install aioredis==1.1.0 && \
pip install tensorflow==1.15.0 && \
pip install tensorflow-datasets==2.0 && \
source deactivate && \

conda create -y -n pytf2 python=3.7 && \
source activate pytf2 && \
pip install --no-cache-dir --upgrade pip && \
pip install --no-cache-dir --upgrade setuptools && \
pip install --no-cache-dir numpy==1.18.1 scipy && \
pip install --no-cache-dir pandas==1.0.3 && \
pip install --no-cache-dir scikit-learn matplotlib seaborn jupyter jupyterlab requests h5py && \
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
pip install --no-cache-dir Pillow==6.2 && \
pip install --no-cache-dir psutil aiohttp && \
pip install --no-cache-dir py4j && \
pip install --no-cache-dir cmake==3.16.3 && \
pip install --no-cache-dir torch==1.7.1 torchvision==0.8.2 && \
pip install --no-cache-dir horovod==0.19.2 && \
pip install --no-cache-dir xgboost && \
pip install --no-cache-dir pyarrow && \
pip install opencv-python==4.2.0.34 && \
pip install aioredis==1.1.0 && \
pip install tensorflow==2.4.0 && \
source deactivate
