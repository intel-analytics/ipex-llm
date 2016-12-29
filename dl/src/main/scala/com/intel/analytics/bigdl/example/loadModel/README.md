# Load Pretrained Model

Bigdl supports loading pretrained models from serialized models 
for inference, finetune or training.

Currently, three sources are supported:

* Bigdl model
* Torch model
* Caffe model

**ImageClassifier** provides an integrated example to load models from the above sources, 
test over imagenet validation dataset on both local mode and spark cluster mode.

## Preparation

To start with this example, you need prepare your model, dataset.

You can find some pretrained caffe model in [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)

## Run in Local Mode

Command to run the example in local mode:

```
java -cp bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
com.intel.analytics.bigdl.example.loadModel.ImageClassifier \
-t $modelType -f $folder -m $modelName --caffeDefPath $pathToCaffePrototxt \
--modelPath $pathToModel -b $batchSize --env local
```

where 

* ```-t``` is the type of model to load, it can be bigdl, torch, caffe.
* ```-f``` is the folder holding validation data,
* ```-m``` is the name of model to use, it can be inception, alexnet, or resenet in this example.
* ```--caffeDefPath``` is the path of caffe prototxt file, this is only needed for caffe model
* ```--modelPath``` is the path to serialized model
* ```-b``` is the batch size to use when do the validation.
* ```--env``` is the execution environment

Some other parameters

* ```-n```: node number to do the validation
* ```--meanFile```: mean values that is need in alexnet caffe model preprocess part

## Run in spark Mode

```
spark-submit \
 --driver-class-path bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
 --class com.intel.analytics.bigdl.example.loadModel.ImageClassifier \
 bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar  \
 -t $modelType -f $folder -m $modelName --caffeDefPath $pathToCaffePrototxt \
 --modelPath $pathToModel \
 -b $batchSize --env spark --node 8
```

where 

```--node``` is the number of nodes to test the model


## Expected Results

You should get similar top1 or top5 accuracy as the original model.











