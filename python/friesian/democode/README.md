# Running Friesian demo examples

A couple of end to end training pipelines are provided here to showcase how to use Friesian Table to generate features for popular recommender models.

We recommend conda to set up your environment. You can install a conda distribution from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
if you haven't already.

```bash
conda create -n bigdl python==3.7
conda activate bigdl
pip install tensorflow pandas pyarrow pillow numpy
```

Then download and install latest nightly-build BigDL Friesian.
```bash
pip install --pre --upgrade bigdl-friesian[train]
```

## Train wide_and_deep model

```bash
python train_wnd.py
```