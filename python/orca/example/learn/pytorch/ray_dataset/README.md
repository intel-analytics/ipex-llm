# PyTorch with Ray Dataset Input Example

This example is adapted from
https://github.com/ray-project/ray/blob/releases/1.9.0/doc/examples/datasets_train/datasets_train.py

In this example, you can learn how to use Ray Datasets as a first-class input in Orca PyTorch Estimator to conduct a distributed training pipeline.

## Prepare the environment

We recommend you to use Anaconda to prepare the environment, especially if you want to run on a yarn cluster:

```bash
conda create -n bigdl python=3.7  # "bigdl" is conda environment name, you can use any name you like.
conda activate bigdl
pip install torch

# For torch_distributed backend:
pip install bigdl-orca[ray]
pip install tqdm  # progress bar
pip install --pre --upgrade ray
```

## Prepare Dataset

In this example, you need to execute `data_generation.py` to prepare the dataset for training pipeline as bellow:

```python
python data_generation.py --data_dir ${data_dir}
```

We also support generating dataset to distributed platform like S3 as follows:

```python
python data_generation.py --data_dir ${data_dir} --use_s3 True
```

## Run the example

Currently, we support Ray Datasets input in `torch_distributed` backend. You can load data to the Ray Dataset from local disk or remote resources like `S3`. For here you should specify the `data_dir` as the location of generated data files.

```python
python ray_dataset_prediction.py --runtime ray --data_dir ${data_dir}
```

If you want to finish testing quickly and simply, we recommend you to use `smoke_test`, which conducts training with only one data file as follows:

```python
python ray_dataset_prediction.py --runtime ray --data_dir ${data_dir} --smoke_test True
```
