# PyTorch BrainMRI example
We demonstrate how to easily run synchronous distributed PyTorch training using PyTorch Estimator of Project Orca in [BigDL](https://github.com/intel-analytics/BigDL). We use a simple Unet model to train on BrainMRI Segmentation dataset, which is a dataset for image segmentation. See [here](https://www.kaggle.com/s0mnaths/brain-mri-unet-pytorch/notebook) for the original single-node version.


## Prepare the environment

We recommend you to use Anaconda to prepare the environment, especially if you want to run on a yarn cluster:

```
conda create -n bigdl python=3.7  # "bigdl" is conda environment name, you can use any name you like.
conda activate bigdl
pip install torch
pip install torchvision
pip install albumentations
pip install scikit-learn
pip install opencv-python

# For torch_distributed and spark backend:
pip install bigdl-orca[ray]
```

## Run on local after pip install

The default backend is `torch_distributed`

You can run with `spark` backend via:

```
python brainMRI.py --backend spark 
```

## Run on yarn cluster for yarn-client mode after pip install

```
export HADOOP_CONF_DIR=the directory of the hadoop and yarn configurations
python brainMRI.py --cluster_mode yarn-client
```