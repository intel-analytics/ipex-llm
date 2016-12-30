## Summary
 This example shows how to predict images on trained model.
 
1. Only support load bigdl model and torch model, not support caffe model
2. Only support some models of imagenet, such as resnet and inception.
3. Only support loading imagenet images or similar from local, not hdfs.

## Preparation

To start with this example, you need prepare your model and dataset.

1. Prepare model.

    The torch resnet model used in this example can be found in [Resnet Torch Model](https://github.com/facebook/fb.resnet.torch/tree/master/pretrained).
    The bigdl inception model used in this example can be trained with [bigdl Inception](https://github.com/intel-analytics/BigDL/tree/master/dl/src/main/scala/com/intel/analytics/bigdl/models/inception).
    You can choose one of them, and then put the trained model in $modelPath, and set corresponding $modelType（torch or bigdl）.
   
2. Prepare predict dataset

    You can use imagenet-2012 validation dataset to run the example, the data can be found from <http://image-net.org/download-images>.
    After you download the files(ILSVRC2012_img_val.tar), run the follow commands to prepare the data.
    
     ```bash
    mkdir predict
    tar -xvf ILSVRC2012_img_val.tar -C ./predict/
    ```
    
    If your want to use your own data similar to imagenet, <code>Note</code> : You should put all predict images in the same $folder.



## Run this example

Command to run the example in spark local mode:

```
    ./bigdl.sh 
    spark-submit --master local[*] --driver-memory 10g --executor-memory 20g --class com.intel.analytics.bigdl.example.imageclassification.ImagePredictor bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar 
        --modelPath ./resnet-18.t7 --folder ./predict --modelType torch -c 4 -n 1 --batchSize 32
```


Command to run the example in spark cluster mode:

```
    MASTER=xxx.xxx.xxx.xxx:xxxx
    ./bigdl.sh 
    spark-submit --master ${MASTER} --driver-memory 10g --executor-memory 20g --class com.intel.analytics.bigdl.example.imageclassification.ImagePredictor bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar 
    --modelPath ./resnet-18.t7 --folder ./predict --modelType torch -c 8 -n 4 --batchSize 32
```

where 

* ```--modelPath``` is model snapshot location.
* ```--folder``` is the folder of predict images.
* ```--modelType``` is the type of model to load, it can be bigdl or torch.
* ```-n``` is nodes number to use the model.
* ```-c``` is cores number on each node.
* ```--showNum``` is the result number to show, default 100.
* ```--batchSize``` is the batch size to use when do the prediction, default 32.