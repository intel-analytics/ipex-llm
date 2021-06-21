# Running Orca TF2 YoloV3 example


## Environment

We recommend conda to set up your environment. You can install a conda distribution from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
if you haven't already.

```bash
conda create -n analytics-zoo python==3.7
conda activate analytics-zoo
pip install tensorflow
pip install pandas
```

Then download and install latest nightly-build Analytics Zoo 

```bash
pip install --pre --upgrade analytics-zoo[ray]
```

## Training Data

Download VOC2009 dataset [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar) 


## Pretrained Weights

Download pretrained weights [here](https://pjreddie.com/media/files/yolov3.weights)

## Running example

Example command:

```
python yoloV3.py --data_dir ${data_dir} --weights ${weights} --class_num ${class_num} --names ${names}
```

