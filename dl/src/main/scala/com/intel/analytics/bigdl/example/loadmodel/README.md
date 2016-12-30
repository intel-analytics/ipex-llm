# Load Pre-trained Model

This example demonstrates how to use BigDL to load pre-trained [Torch](http://torch.ch/) or [Caffe](http://caffe.berkeleyvision.org/) model into Spark program for prediction.

**ModelValidator** provides an integrated example to load models, and test over imagenet validation dataset (running as a local Java program, or a standard Spark program).

## Preparation

To start with this example, you need prepare your model, dataset.

The caffe model used in this example can be found in 
[GoogleNet Caffe Model](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet)
and [Alexnet Caffe Model](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet).

The torch model used in this example can be found in
[Resnet Torch Model](https://github.com/facebook/fb.resnet.torch/tree/master/pretrained#trained-resnet-torch-models).

The imagenet validation dataset preparation can be found from
[BigDL inception Prepare the data](https://github.com/intel-analytics/BigDL/tree/master/dl/src/main/scala/com/intel/analytics/bigdl/models/inception#prepare-the-data).

## Run as a local Java program

When running as a local Java program, we use original imagenet image folder as input.

Command to run the example:

```
modelType=caffe
folder=imagenet
modelName=inception
pathToCaffePrototxt=data/model/googlenet/deploy.prototxt
pathToModel=data/model/googlenet/bvlc_googlenet.caffemodel
batchSize=64
dist/bin/bigdl.sh -- java -cp dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar \
com.intel.analytics.bigdl.example.loadmodel.ModelValidator \
-t $modelType -f $folder -m $modelName --caffeDefPath $pathToCaffePrototxt \
--modelPath $pathToModel -b $batchSize --env local
```

where 

* ```-t``` is the type of model to load, it can be torch, caffe.
* ```-f``` is the folder holding validation data,
* ```-m``` is the name of model to use, it can be inception, alexnet, or resenet in this example.
* ```--caffeDefPath``` is the path of caffe prototxt file, this is only needed for caffe model
* ```--modelPath``` is the path to serialized model
* ```-b``` is the batch size to use when do the validation.
* ```--env``` is the execution environment

Some other parameters

* ```-n```: node number to do the validation
* ```--meanFile```: mean values that is needed in alexnet caffe model preprocess part

## Run as a Spark program

When running as a Spark program, we use transformed imagenet sequence file as input.

For Caffe Inception model and Alexnet model, the command to transform the sequence file is (validation data is the different with the [BigDL inception Prepare the data](https://github.com/intel-analytics/BigDL/tree/master/dl/src/main/scala/com/intel/analytics/bigdl/models/inception#prepare-the-data))

```bash
dist/bin/bigdl.sh -- java -cp dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar \
com.intel.analytics.bigdl.models.utils.ImageNetSeqFileGenerator -f imagenet_folder \
-o output_folder -p cores_number -r -v
```

For Torch Resnet model, the command to transform the sequence file is (validation data is the same with the [BigDL Inception Prepare the data](https://github.com/intel-analytics/BigDL/tree/master/dl/src/main/scala/com/intel/analytics/bigdl/models/inception#prepare-the-data))

```bash
dist/bin/bigdl.sh -- java -cp dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar \
com.intel.analytics.bigdl.models.utils.ImageNetSeqFileGenerator -f imagenet_folder \
-o output_folder -p cores_number -v
```

Then put the transformed sequence files to HDFS.

```
hdfs dfs -put output_folder hdfs_folder
```

Having prepared the dataset, you can submit your spark job by 

```
modelType=caffe
folder=hdfs_folder
modelName=inception
pathToCaffePrototxt=data/model/googlenet/deploy.prototxt
pathToModel=data/model/googlenet/bvlc_googlenet.caffemodel
batchSize=448
dist/bin/bigdl.sh -- spark-submit \
 --driver-class-path dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
 --class com.intel.analytics.bigdl.example.loadmodel.ModelValidator \
 dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar  \
 -t $modelType -f $folder -m $modelName --caffeDefPath $pathToCaffePrototxt \
 --modelPath $pathToModel \
 -b $batchSize --env spark --node 8
```

where 

```--node``` is the number of nodes to test the model

other parameters have the same meaning as local mode.


## Expected Results

You should get similar top1 or top5 accuracy as the original model.
