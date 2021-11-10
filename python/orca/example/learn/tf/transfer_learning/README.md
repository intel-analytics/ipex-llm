# Transfer Learning with Orca TF Estimator

This is an example to demonstrate how to use Analytics-Zoo's Orca TF Estimator API to run distributed [Transfer Learning](https://github.com/tensorflow/docs/blob/master/site/en/r1/tutorials/images/transfer_learning.ipynb) training and inference task.

## Environment Preparation

Download and install latest analytics whl by following instructions ([here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/#install-the-latest-nightly-build-wheels-for-pip)).

```bash
conda create -n zoo python=3.7
conda activate zoo
pip install tensorflow==1.15
pip install tensorflow_datasets==3.2.0
pip install analytics_zoo-${VERSION}-${TIMESTAMP}-py2.py3-none-${OS}_x86_64.whl
```
Note: conda environment is required to run on Yarn, but not strictly necessary for running on local.

## Data Preparation
Cats_and_dogs_filtered dataset will be auto-downloaded, or you can run prepare_dataset.sh to download dataset in advance.

## Run example on local
```bash
python transfer_learning.py --cluster_mode local
```

## Run example on yarn cluster
```bash
python transfer_learning.py --cluster_mode yarn
```

Options
* `--cluster_mode` The mode for the Spark cluster. local or yarn. Default to be `local`.
* `--data_dir` The path of dataset. Default is "./dataset".
* `--batch_size` The training batch size. Default to be `64`.
* `--epochs` The number of epochs to train. Default to be `2`.

