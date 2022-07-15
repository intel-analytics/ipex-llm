# Chronos nightly build docker image
This dockerfile helps user to build a docker image where Chronos nightly build version is deploied.

## Build an image
First `cd` to this directory. Then build your docker image with Dockerfile:
```bash
sudo docker build -t chronos-nightly:b1 . # You may choose any NAME:TAG you want.
```
Or build with a proxy:
```bash
# typically, you need a proxy for building since there will be some downloading.
sudo docker build \
    --build-arg http_proxy=http://<your_proxy_ip>:<your_proxy_port> \
    --build-arg https_proxy=http://<your_proxy_ip>:<your_proxy_port> \
    -t chronos-nightly:b1 . # You may choose any NAME:TAG you want.
```
According to your network status, this building will cost **around 10 mins**

## Run the image
```bash
sudo docker run -it --rm --net=host chronos-nightly:b1 bash
```

## Use Chronos
A conda environment is created for you automatically. `bigdl-chronos` and all of its depenencies are installed inside this environment.
```bash
(chronos) root@cpx-3:/opt/work#
```

## Run Unitest Examples on Jupyter Notebook for a quick use
There have been some Chronos unitest examples about time-series passed in the docker. You can run these on Jupyter Notebook on single node server if you pursue a quick use on Chronos.
```bash
(chronos) root@cpx-3:/opt/work# cd /opt/work/colab-notebook #Unitest examples are here.
```
```bash
(chronos) root@cpx-3:/opt/work# jupyter notebook --notebook-dir=./ --ip=* --allow-root #Start the Jupyter Notebook services.
```
After the Jupyter Notebook service is successfully started, you can connect to the Jupyter Notebook service from a browser.
    1.Get the IP address of the container
    2.Launch a browser, and connect to the Jupyter Notebook service with the URL: 
    https://container-ip-address:port-number/?token=your-token As a result, you will see the Jupyter Notebook opened.
    3.Open one of these `.ipynb` files, run through the example and learn how to use Chronos to predict time series.

## Shut Down Docker Container
You should shut down the BigDL Docker container after using it.
1. First, use `ctrl+p+q` to quit the container when you are still in it. 
2. Then, you can list all the active Docker containers by command line:
```bash
sudo docker ps
```
You will see your docker containers:
```bash
CONTAINER ID        IMAGE                                        COMMAND                  CREATED             STATUS              PORTS               NAMES
40de2cdad025        chronos-nightly:b1         "/opt/work/"   3 hours ago         Up 3 hours                              upbeat_al
```
3. Shut down the corresponding docker container by its ID:
```bash
sudo docker rm -f 40de2cdad025
```
