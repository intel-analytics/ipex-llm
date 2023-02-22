# Use Chronos in Container (docker)
This page helps user to build and use a docker image where Chronos-nightly build version is deployed.

## Download image from Docker Hub
We provide docker image with Chronos-nightly build version deployed in [Docker Hub](https://hub.docker.com/r/intelanalytics/bigdl-chronos/tags). You can directly download it by running command:
```bash
docker pull intelanalytics/bigdl-chronos:latest
```

## Build an image (Optional)
**If you have downloaded docker image, you can just skip this part and go on [Use Chronos](#use-chronos).**

First clone the repo `BigDL` to the local.
```bash
git clone https://github.com/intel-analytics/BigDL.git
```
Then `cd` to the root directory of `BigDL`, and copy the Dockerfile to it. 
```bash
cd BigDL
cp docker/chronos-nightly/Dockerfile ./Dockerfile
```
When building image, you can specify some build args to install chronos with necessary dependencies according to your own needs.
The build args are similar to the install options in [Chronos documentation](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/install.html).

```
model: which model or framework you want. 
       value: pytorch
              tensorflow
              prophet
              arima
              ml (default, for machine learning models).

auto_tuning: whether to enable auto tuning.
             value: y (for yes)
                    n (default, for no).

hardware: run chronos on a single machine or a cluster.
          value: single (default)
                 cluster

inference: whether to install dependencies for inference optimization (e.g. onnx, openvino, ...).
           value: y (for yes)
                  n (default, for no)

extra_dep: whether to install some extra dependencies.
           value: y (for yes)
                  n (default, for no)
           if specified to y, the following dependencies will be installed:
           tsfresh, pyarrow, prometheus_pandas, xgboost, jupyter, matplotlib
```

If you want to build image with the default options, you can simply use the following command:
```bash
sudo docker build -t intelanalytics/bigdl-chronos:latest . # You may choose any NAME:TAG you want.
```

You can also build with other options by specifying the build args:
```bash
sudo docker build \
    --build-arg model=pytorch \
    --build-arg auto_tuning=y \
    --build-arg hardware=single \
    --build-arg inference=n \
    --build-arg extra_dep=n \
     -t intelanalytics/bigdl-chronos:latest . # You may choose any NAME:TAG you want.
```

(Optional) If you need a proxy, you can add two additional build args to specify it:
```bash
# typically, you need a proxy for building since there will be some downloading.
sudo docker build \
    --build-arg http_proxy=http://<your_proxy_ip>:<your_proxy_port> \ #optional
    --build-arg https_proxy=http://<your_proxy_ip>:<your_proxy_port> \ #optional
    -t intelanalytics/bigdl-chronos:latest . # You may choose any NAME:TAG you want.
```
According to your network status, this building will cost **15-30 mins**. 

**Tips:** When errors happen like `failed: Connection timed out.`, it's usually related to the bad network status. Please build with a proxy.

## Run the image
```bash
sudo docker run -it --rm --net=host intelanalytics/bigdl-chronos:latest bash
```

## Use Chronos
A conda environment is created for you automatically. `bigdl-chronos` and the necessary depenencies (based on the build args when you build image) are installed inside this environment.
```bash
(chronos) root@icx-5:/opt/work# 
```
```eval_rst
.. important::

       Considering the image size, we build docker image with the default args and upload it to Docker Hub. If you use it directly, only ``bigdl-chronos`` is installed inside this environment. There are two methods to install other necessary dependencies according to your own needs:

       1. Make sure network is available and run install command following `Install using Conda <https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/install.html#install-using-conda>`_ , such as ``pip install --pre --upgrade bigdl-chronos[pytorch]``.

       2. Make sure network is available and bash ``/opt/install-python-env.sh`` with build args. The values are introduced in `Build an image <## Build an image (Optional)>`_.

       .. code-block:: python

              # bash /opt/install-python-env.sh ${model} ${auto_tuning} ${hardware} ${inference} ${extra_dep}
              # For example, if you want to install bigdl-chronos[pytorch,inference]
              bash /opt/install-python-env.sh pytorch n single y n

```

## Run unittest examples on Jupyter Notebook for a quick use
> Note: To use jupyter notebook, you need to specify the build arg `extra_dep` to `y`.

You can run these on Jupyter Notebook on single node server if you pursue a quick use on Chronos.
```bash
(chronos) root@icx-5:/opt/work# cd /opt/work/colab-notebook #Unittest examples are here.
```
```bash
(chronos) root@icx-5:/opt/work/colab-notebook# jupyter notebook --notebook-dir=./ --ip=* --allow-root #Start the Jupyter Notebook services.
```
After the Jupyter Notebook service is successfully started, you can connect to the Jupyter Notebook service from a browser.
1. Get the IP address of the container
2. Launch a browser, and connect to the Jupyter Notebook service with the URL: 
</br>`https://container-ip-address:port-number/?token=your-token`
</br>As a result, you will see the Jupyter Notebook opened.
3. Open one of these `.ipynb` files, run through the example and learn how to use Chronos to predict time series.

## Shut down docker container
You should shut down the BigDL Docker container after using it.
1. First, use `ctrl+p+q` to quit the container when you are still in it. 
2. Then, you can list all the active Docker containers by command line:
   ```bash
   sudo docker ps
   ```
   You will see your docker containers:
   ```bash
   CONTAINER ID   IMAGE                                 COMMAND   CREATED       STATUS       PORTS     NAMES
   ef133bd732d1   intelanalytics/bigdl-chronos:latest   "bash"    2 hours ago   Up 2 hours             happy_babbage
   ```
3. Shut down the corresponding docker container by its ID:
   ```bash
   sudo docker rm -f ef133bd732d1
   ```
