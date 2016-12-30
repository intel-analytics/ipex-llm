# Load Pretrained Model

Bigdl supports loading pretrained models from other popular deep learning projects.

Currently, two sources are supported:

* Torch model
* Caffe model

**ModelValidator** provides an integrated example to load models from the above sources, 
test over imagenet validation dataset on both local mode and spark cluster mode.

## Preparation

To start with this example, you need prepare your model, dataset.

The caffe model used in this example can be found in 
[Inference Caffe Model](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet)
and [Alexnet Caffe Model](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet).

The torch model used in this example can be found in
[Resnet Torch Model](https://github.com/facebook/fb.resnet.torch/tree/master/pretrained).

The imagenet validation dataset preparation can be found
[here](https://github.com/intel-analytics/BigDL/tree/master/dl/src/main/scala/com/intel/analytics/bigdl/models/inception)

## Run in Local Mode

In the local mode example, we use original imagenet image folder as input.

Command to run the example in local mode:

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

## Run in spark Mode

In the spark mode example, we use transformed imagenet sequence file as input.

For caffe inception model and alexnet model, the command to transform the sequence file is

```bash
dist/bin/bigdl.sh -- java -cp dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar \
com.intel.analytics.bigdl.models.utils.ImageNetSeqFileGenerator -f imagenet_folder \
-o output_folder -p cores_number -r
```

For torch resnet model, the command to transform the sequence file is

```bash
dist/bin/bigdl.sh -- java -cp dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar \
com.intel.analytics.bigdl.models.utils.ImageNetSeqFileGenerator -f imagenet_folder \
-o output_folder -p cores_number
```

Then put the transformed sequence files to HDFS.

Having prepared the dataset, you can submit your spark job by 

```
modelType=caffe
folder=imagenet
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











