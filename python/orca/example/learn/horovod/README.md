# Running Horovod programs on RayOnSpark

This example demonstrates how to easily run a Horovod program on RayOnSpark using
analytics zoo's HorovodRayTrainer.


## Environment

Running this example requires a Yarn cluster.

Running this example also requires a conda environment. You can install a conda distribution from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
if you haven't already.

Then create a conda an environment and follow the [horovod document](https://github.com/horovod/horovod/blob/master/docs/install.rst) to install the horovod and pytorch with Gloo support.

Here are the commands that work on for us on ubuntu 16.04. The exact steps may vary from different machines.

```bash
conda install -y pytorch==1.4.0 torchvision==0.5.0 cpuonly -c pytorch
conda install -y cmake==3.16.0 -c conda-forge
conda install cxx-compiler==1.0 -c conda-forge
conda install openmpi
HOROVOD_WITH_PYTORCH=1; HOROVOD_WITH_GLOO=1; pip install --no-cache-dir horovod==0.19.1
pip install analytics-zoo[ray]
```

## Running

After creating and activating the conda environment, you can run this example by:

```bash
python simple_horovod_pytorch.py --hadoop_conf $your_hadoop_conf_directory --conda_name $your_conda_env_name

```
