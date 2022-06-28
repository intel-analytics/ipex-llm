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
pip install matplotlib

# For torch_distributed backend:
pip install bigdl-orca[ray]

# For spark backend:
pip install bigdl-orca
```

## Prepare the dataset
You can download the dataset [here](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation/download).


```
# You can download the dataset to your project path.
unzip archive.zip

# if you use yarn-client, the dataset should be sent to the excutors. 
# So we need zip the dataset as `.zip` file, then the program will send it to all excutors automatically. 
cd kaggle_3m
zip ../kaggle_3m.zip *
```


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
You can run with `bigdl` backend via:
```
python brainMRI.py --backend bigdl
```
### Run on yarn cluster for yarn-client mode after pip install
```
export HADOOP_CONF_DIR=the directory of the hadoop and yarn configurations
python brainMRI.py --cluster_mode yarn-client
```

If you use the spark as backend, you should give the `model_dir` parameter with the hdfs path. 

Note: if you meet `OSError: Unable to load libhdfs: ./libhdfs.so: cannot open shared object file: No such file or directory` error, read the [document](https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/known_issues.html#oserror-unable-to-load-libhdfs-libhdfs-so-cannot-open-shared-object-file-no-such-file-or-directory).
```commandline
python brainMRI.py --cluster_mode yarn-client --backend spark --model_dir hdfs://url:port/file_path
```

Options

- `--cluster_mode` The cluster mode, such as local, yarn-client, spark-submit. Default is `local`.
- `--backend` The backend of PyTorch Estimator; torch_distributed and spark are supported. Default is `torch_distributed`.
- `--epochs` The number of epochs to train for. Default is 2.
- `--batch_size` The number of samples per gradient update. Default is 64.
- `--data_dir` The path of the dataset. Default is `./kaggle_3m`.
- `--additional_archive` The zip dataset if use `yarn-client` mode. Default is `kaggle_3m.zip#kaggle_3m`.
- `--memory` The memory allocated for each node. Default is `4g`.
- `--model_dir` The model save dir when use spark backend. Default is the current working directory.
  - If you use `yarn-client` as `cluster_mode`, you need give the hdfs path to save the model. 
  - If you use `spark-submit` as `cluster_mode`, you need add `file://` at the beginning of the path.
- `spark_executor_dir` The spark.executorEnv.ARROW_LIBHDFS_DIR is need when use spark backend and yarn cluster_mode.