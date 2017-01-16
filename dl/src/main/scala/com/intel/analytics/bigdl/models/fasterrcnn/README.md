# Faster-RCNN

Faster-RCNN is a popular object detection framework, which is described in 
[paper](https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf) published in NIPS 2015.
It's official code can be found [here](https://github.com/rbgirshick/py-faster-rcnn) 
and a python version can be found [here](https://github.com/SeaOfOcean/py-faster-rcnn).

Later, [PVANET](https://arxiv.org/abs/1611.08588) further reduces computational cost with a lighter network.
It's implementation can be found [here](https://github.com/sanghoon/pva-faster-rcnn)

This example demonstrates how to use BigDL to test a Faster-RCNN framework with either pvanet network or vgg16 network.

## Prepare the dataset
Download the test dataset

```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
```

Extract all of these tars into one directory named ```VOCdevkit```

```
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar
```

It should have this basic structure

```
$VOCdevkit/                           # development kit
$VOCdevkit/VOCcode/                   # VOC utility code
$VOCdevkit/VOC2007                    # image sets, annotations, etc.
# ... and several other directories ...
```

## Download pretrained model

You can use [this script](https://github.com/rbgirshick/py-faster-rcnn/blob/master/data/scripts/fetch_faster_rcnn_models.sh) to download
pretrained Faster-RCNN(VGG) models.

Faster-RCNN(PVANET) model can be found [here](https://www.dropbox.com/s/87zu4y6cvgeu8vs/test.model?dl=0), 
its caffe prototxt file can be found [here](https://github.com/sanghoon/pva-faster-rcnn/blob/master/models/pvanet_obsolete/full/test.pt)

## Run the demo

Example command for running as a local Java program
```
./dist/bin/bigdl.sh -- \
java -cp dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar \
com.intel.analytics.bigdl.models.fasterrcnn.Demo \
 -f demo \
 -o demoOut \
 --caffeDefPath data/pvanet/full/test.pt \
 --caffeModelPath data/pvanet/full/test.model \
 -t PVANET  \
 -c physical_core_number \
 -n 1 \
 --env local
```

Example command for running in Spark cluster mode
```
dist/bin/bigdl.sh -- spark-submit \
 --driver-memory 32g \
 --executor-memory 32g \
 --class com.intel.analytics.bigdl.models.fasterrcnn.Demo \
 dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar  \
 -f demo \
 -o demoOut \
 --caffeDefPath data/pvanet/full/test.pt \
 --caffeModelPath data/pvanet/full/test.model \
 -t PVANET  \
 -c 8 \
 -n 4 \
 --env spark
```
In the above commands
* -f: where you put your demo data
* -o: where you put your demo output data
* --caffeDefPath: caffe network definition prototxt file path
* --caffeModelPath: caffe serialized model file path
* -t: network type, it can be PVANET or VGG16
* --c: How many cores of your machine will be used. Note that the core number should be physical core number. If your machine turn on hyper threading, one physical core will map to two OS core.
* -n: Node number.
* --env: It can be local/spark.


## Run the test

Example command for running as a local Java program
```
./dist/bin/bigdl.sh -- \
java -cp dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar \
com.intel.analytics.bigdl.models.fasterrcnn.Test \
 -f VOCdevkit \
 -i voc_2007_test \
 --caffeDefPath data/pvanet/full/test.pt \
 --caffeModelPath data/pvanet/full/test.model \
 -t PVANET  \
 -c physical_core_number \
 -n 1 \
 --env local
```

Example command for running in Spark cluster mode
```
dist/bin/bigdl.sh -- spark-submit \
 --driver-memory 32g \
 --executor-memory 32g \
 --class com.intel.analytics.bigdl.models.fasterrcnn.Test \
 dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar  \
 -f VOCdevkit \
 -i voc_2007_test \
 --caffeDefPath data/pvanet/full/test.pt \
 --caffeModelPath data/pvanet/full/test.model \
 -t PVANET  \
 -c 8 \
 -n 4 \
 --env spark
```
In the above commands
* -f: VOCdevkit path
* -i: image set name with the format ```voc_${year}_${imageset}```, e.g. voc_2007_test
* --caffeDefPath: caffe network definition prototxt file path
* --caffeModelPath: caffe serialized model file path
* -t: network type, it can be PVANET or VGG16
* --c: How many cores of your machine will be used. Note that the core number should be physical core number. If your machine turn on hyper threading, one physical core will map to two OS core.
* -n: Node number.
* --env: It can be local/spark.





