# Running Orca TF2 mnist example


## Environment

First TensorFlow need to be installed.

We recommend conda to set up your environment. You can install a conda distribution from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
if you haven't already.

Then you can follow the [horovod document](https://github.com/horovod/horovod/blob/master/docs/install.rst) to install the horovod with Gloo support.

Here are the commands that work on for us on ubuntu 16.04. The exact steps may vary from different machines.

```bash
conda create -n analytics-zoo python==3.7.7
conda activate analytics-zoo
pip install tensorflow==2.3.0
```

Then download and install latest analytics whl by following instructions ([here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/#install-the-latest-nightly-build-wheels-for-pip).

E.g.
```bash
pip install analytics_zoo-${VERSION}-${TIMESTAMP}-py2.py3-none-${OS}_x86_64.whl[ray]
```

## Run examples on local

```bash
python lenet_mnist_keras.py --cluster_mode local 
```

## Run examples on yarn cluster

```bash
python lenet_mnist_keras.py --cluster_mode yarn
```