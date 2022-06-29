# PyTorch BrainMRI example
We demonstrate how to easily run synchronous distributed PyTorch training using PyTorch Estimator of Project Orca in Analytics Zoo. We use a simple Unet model to train on BrainMRI Segmentation dataset, which is a dataset for image segmentation. See [here](https://www.kaggle.com/s0mnaths/brain-mri-unet-pytorch/notebook) for the original single-node version. A lot of thanks to S0MNATHS, the orginal author of the code.


## Trouble Shooting
- Difficult to identify a customized loss function; for example, dice coefficient
    - Require a loss function creator to pass a customized loss function into orca
- Unable to see the plot of matplotlib
- Need to implement function to download dataset if it's not included in pytorch datasets

## TODO
- [ ] bigdl backend development
- [ ] yarn development

## Prepare the environment

We recommend you to use Anaconda to prepare the environment, especially if you want to run on a yarn cluster:

```
conda create -n zoo python=3.7  # "zoo" is conda environment name, you can use any name you like.
conda activate zoo
pip install torch
pip install torchvision
pip install matplotlib

# For bigdl backend:
pip install analytics-zoo  # 0.10.0.dev3 or above
pip install jep==3.9.0
pip install six cloudpickle

# For torch_distributed backend:
pip install analytics-zoo[ray]  # 0.10.0.dev3 or above

# For spark backend
pip install bigdl-orca
```

## Run on local after pip install

`bigdl` backend is still developing.

You can run with `torch_distributed` backend via:

```
python brainMRI.py --backend torch_distributed
```

You can run with `spark` backend via:

```
python brainMRI.py --backend spark
```

## Run on yarn cluster for yarn-client mode after pip install

```
export HADOOP_CONF_DIR=the directory of the hadoop and yarn configurations
python brainMRI.py --cluster_mode yarn-client
```