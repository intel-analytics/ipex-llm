# PyTorch BrainMRI example
We demonstrate how to easily run synchronous distributed PyTorch training using PyTorch Estimator of Project Orca in BigDL. We use a simple Unet model to train on BrainMRI Segmentation dataset, which is a dataset for image segmentation. See [here](https://www.kaggle.com/s0mnaths/brain-mri-unet-pytorch/notebook) for the original single-node version.


## Prepare the environment

We recommend you to use Anaconda to prepare the environment, especially if you want to run on a yarn cluster:

```
conda create -n bigdl python=3.7  # "bigdl" is conda environment name, you can use any name you like.
conda activate bigdl
pip install torch
pip install torchvision
pip install 'albumentations<=1.4.0'
pip install scikit-learn
pip install opencv-python
pip install matplotlib
pip install tqdm
pip install pyarrow

# For ray backend:
pip install --pre --upgrade bigdl-orca[ray]

# For spark backend:
pip install --pre --upgrade bigdl-orca
```

## Prepare the dataset
You can download the dataset from [here](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation/download). Put the downloaded data in the current working directory.


```
unzip archive.zip

# If you run in yarn-client mode, you need to zip the images under kaggle_3m to the current working directory as well.
cd kaggle_3m
zip ../kaggle_3m.zip *
cd ../
```


## Run on local after pip install
Note: You should add the current directory into the `PYTHONPATH` first.
```
export PYTHONPATH=./:$PYTHONPATH
```

```commandline
python brainMRI.py
```
The default backend is `ray`. You can run with `spark` backend via:
```
python brainMRI.py --backend spark 
```

## Run on yarn cluster for yarn-client mode after pip install
```
export HADOOP_CONF_DIR=the directory of the hadoop and yarn configurations
python brainMRI.py --cluster_mode yarn-client
```

If you use spark backend, you should input `model_dir` parameter with the hdfs path. 

Note: If you meet `OSError: Unable to load libhdfs: ./libhdfs.so: cannot open shared object file: No such file or directory`, please refer to [here](https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/known_issues.html#oserror-unable-to-load-libhdfs-libhdfs-so-cannot-open-shared-object-file-no-such-file-or-directory) for the solution.
```commandline
python brainMRI.py --cluster_mode yarn-client --backend spark --model_dir hdfs_path_to_save_model
```

Options

- `--cluster_mode` The cluster mode, such as local, yarn-client, spark-submit. Default is `local`.
- `--backend` The backend of PyTorch Estimator; ray, and spark are supported. Default is `ray`.
- `--epochs` The number of epochs to train for. Default is 2.
- `--batch_size` The number of samples per gradient update. Default is 64.
- `--data_dir` The path to the dataset. Default is `./kaggle_3m`.
- `--additional_archive` The zip dataset if use `yarn-client` mode. Default is `kaggle_3m.zip#kaggle_3m`.
- `--model_dir` The model save dir when use spark backend. Default is the current working directory.

## Results

Final test results will be printed at the end:
```
num_samples: 531
dice_coef_metric: 0.8578884796683207
val_loss: 0.16065971297957354
```