# Running Orca TF2 ResNet50 example


## Environment

First TensorFlow and horovod (built with gloo) need to be installed.

We recommend conda to set up your environment. You can install a conda distribution from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
if you haven't already.

Then you can follow the [horovod document](https://github.com/horovod/horovod/blob/master/docs/install.rst) to install the horovod with Gloo support.

Here are the commands that work on for us on ubuntu 16.04. The exact steps may vary from different machines.

```bash
conda create -n analytics-zoo python==3.7.7
conda activate analytics-zoo
conda install -y cmake==3.16.0 -c conda-forge
pip install tensorflow==2.3.0
HOROVOD_WITH_TENSORFLOW=1; HOROVOD_WITH_GLOO=1; pip install --no-cache-dir horovod==0.19.2
```

Then download and install latest analytics whl by following instructions ([here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/#install-the-latest-nightly-build-wheels-for-pip).

E.g.
```bash
pip install analytics_zoo-${VERSION}-${TIMESTAMP}-py2.py3-none-${OS}_x86_64.whl[ray]
```

## Training Data

You can follow the instructions [here](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data) here
to generate training and validation tf records data.


## Running multiple process in one machine using Spark Standalone

Example command:

```
python -u resnet-50-imagenet.py --cluster_mode standalone --worker_num 8 --cores 17 --data_dir $TF_RECORDS_PATH --use_bf16 --enable_numa_binding
```

## Running fewer iterations for benchmarking 

Example command:

```
python -u resnet-50-imagenet.py --cluster_mode standalone --worker_num 8 --cores 17 --data_dir $TF_RECORDS_PATH --use_bf16 --enable_numa_binding --benchmark
```