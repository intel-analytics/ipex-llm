# Chronos nightly build docker image
This dockerfile helps user to build a docker image where Chronos nightly build version is deploied.

## Build an image
First `cd` to this directory.
```bash
# typically, you need a proxy for building since there will be some downloading.
sudo docker build \
    --build-arg http_proxy=http://child-prc.intel.com:913 \
    --build-arg https_proxy=http://child-prc.intel.com:913 \
    -t chronos-nightly:b20220524 . # You may choose any NAME:TAG you want.
```
According to your network status, this building will cost **around 10 mins**

## Run the image
```bash
sudo docker run -it --rm --net=host chronos-nightly:b20220524 bash
```

## Use Chronos
A conda environment is created for you automatically. `bigdl-chronos` and all of its depenencies are installed inside this environment.
```bash
(chronos) root@cpx-3:/opt/work#
```