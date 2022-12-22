# PyTorch ResNet50 inference example

## Prepare the dataset
You can download the imagenet dataset from [here](https://www.image-net.org/download.php). Only 50000 validation images are needed for this example.

The folder structure is expected to be `/path/to/imagenet/` and under which there is a `val` folder for validation images.

## Prepare the environment

We recommend you to use Anaconda to prepare the environment:

```
conda create -n bigdl python=3.7  # "bigdl" is conda environment name, you can use any name you like.
conda activate bigdl

pip install --pre --upgrade bigdl-orca-spark3[ray]
pip install tqdm  # progress bar
pip install protobuf==3.19.5

conda install pytorch torchvision cpuonly -c pytorch
pip install intel_extension_for_pytorch

conda install -c conda-forge jemalloc
```

## Running commands
- Spark local mode
```bash
python inference.py /path/to/imagenet --cores 8 --workers_per_node 2 --steps 10 --pretrained
```

- Spark standalone mode
```bash
export PYSPARK_PYTHON=...
python inference.py /path/to/imagenet --cluster_mode standalone --master spark://ip:port --num_nodes 2 --cores 4 --workers_per_node 1 --pretrained
```

- Int8 configure file can be downloaded from: https://raw.githubusercontent.com/IntelAI/models/master/models/image_recognition/pytorch/common/resnet50_configure_sym.json