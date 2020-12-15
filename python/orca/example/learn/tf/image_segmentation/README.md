# Image Segmentation with Orca TF Estimator

This is an example to demonstrate how to use Analytics-Zoo's Orca TF Estimator API to run distributed image segmentation training and inference task.

## Environment Preparation

Download and install latest analytics whl by following instructions ([here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/#install-the-latest-nightly-build-wheels-for-pip)).

```bash
conda create -n zoo python=3.7
conda activate zoo
pip install tensorflow==1.15
pip install Pillow
pip install pandas
pip install matplotlib
pip install sklearn
pip install analytics_zoo-${VERSION}-${TIMESTAMP}-py2.py3-none-${OS}_x86_64.whl
```

Note: conda environment is required to run on Yarn, but not strictly necessary for running on local.

## Data Preparation
You should manually download the dataset from kaggle [carvana-image-masking-challenge](https://www.kaggle.com/c/carvana-image-masking-challenge/data) and save it to `/tmp/carvana/`. We will need three files, train.zip, train_masks.zip and train_masks.csv.zip

## Run example on local
```bash
# linux
python image_segmentation.py --cluster_mode local
# macos
python image_segmentation.py --cluster_mode local --platform mac
```

## Run example on yarn cluster
```bash
# linux
python image_segmentation.py --cluster_mode yarn
# macos
python image_segmentation.py --cluster_mode yarn --platform mac
```

Options
* `--cluster_mode` The mode for the Spark cluster. local or yarn. Default to be `local`.
* `--file_path` The path to carvana train.zip, train_mask.zip and train_mask.csv.zip. Default to be `/tmp/carvana/`.
* `--epochs` The number of epochs to train the model. Default to be 8.
* `--batch_size` Batch size for training and prediction. Default to be 8.
* `--platform` The platform you used to run the example. Default to be `linux`. You should pass `mac` if you use macos.
* `--non_interactive` Flag to not visualize the result.
