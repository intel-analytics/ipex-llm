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

# For torch_distributed backend:
pip install bigdl-orca[ray]

# For spark backend:
pip install bigdl-orca
```

## Prepare the dataset
You can download the dataset in [here](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

```
unzip archive.zip
mv kaggle_3m the/example/path

# if you use yarn-client
cd kaggle_3m
zip ../kaggle_3m.zip *
```
## Run example
You can run the example on local and yarn client mode.
### Run on local after pip install

The default backend is `torch_distributed`
```commandline
python brainMRI.py
```
You can run with `spark` backend via:
```
python brainMRI.py --backend spark 
```

### Run on yarn cluster for yarn-client mode after pip install

```
export HADOOP_CONF_DIR=the directory of the hadoop and yarn configurations
python brainMRI.py --cluster_mode yarn-client
```

Options

- `--cluster_mode` The cluster mode, such as local, yarn-client, spark-submit. Default is `local`
- `--backend` The backend of PyTorch Estimator; torch_distributed and spark are supported. Default is `torch_distributed` 
- `--epochs` The number of epochs to train for. Default is 2
- `--batch_size` The number of samples per gradient update. Default is 64
- `--data_dir` The path of the dataset. Default is `./kaggle_3m`
- `--additional_archive` The zip dataset if use `yarn-client` mode. Default is `kaggle_3m.zip#kaggle_3m`
- `--memory` The memory allocated for each node. Default is `4g`