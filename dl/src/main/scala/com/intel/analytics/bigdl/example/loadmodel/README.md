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
[Resnet Torch Model](https://github.com/facebook/fb.resnet.torch/tree/master/pretrained#trained-resnet-torch-models).

The imagenet validation dataset preparation can be found from
[BigDL inception Prepare the data](https://github.com/intel-analytics/BigDL/tree/master/dl/src/main/scala/com/intel/analytics/bigdl/models/inception#prepare-the-data).

## Run in Local Mode

In the local mode example, we use original imagenet image folder as input.

Command to run the example in local mode:

```shell
dist/bin/bigdl.sh --                                                       \
java -cp dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar \
         com.intel.analytics.bigdl.example.loadmodel.ModelValidator        \
     -t <caffe | torch | bigdl> -f <imagenet path> -m <model name>         \
     --caffeDefPath <caffe model prototxt path> --modelPath <model path>   \
     -b <batch size> --env <local | spark> --node <node number>
```

where 

- ```-t``` is the type of model to load, it can be torch, caffe.
- ```-f``` is the folder holding validation data,
- ```-m``` is the name of model to use, it can be inception, alexnet, or resenet in this example.
- ```--caffeDefPath``` is the path of caffe prototxt file, this is only needed for caffe model
- ```--modelPath``` is the path to serialized model
- ```-b``` is the batch size to use when do the validation.
- ```--env``` is the execution environment

Some other parameters

- ```-n```: node number to do the validation
- ```--meanFile```: mean values that is needed in alexnet caffe model preprocess part

For example, following the steps below will load BVLC GoogLeNet. 

+ Workspace  structure. 

  ```
  [last: 0s][~/loadmodel]$ tree . -L 2
  .
  ├── data
  │   └── model
  ├── dist
  │   ├── bin
  │   └── lib
  └── imagenet
      └── val

  7 directories, 0 files
  [last: 0s][~/loadmodel]$ tree data/
  data/
  └── model
      └── bvlc_googlenet
          ├── bvlc_googlenet.caffemodel
          └── deploy.prototxt

  2 directories, 2 files
  [last: 0s][~/loadmodel]$ tree dist/
  dist/
  ├── bin
  │   ├── bigdl.sh
  │   ├── classes.lst
  │   └── img_class.lst
  └── lib
      ├── bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar
      └── bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar
  ```

+ Execute command.

  ```shell
  modelType=caffe
  folder=imagenet
  modelName=inception
  pathToCaffePrototxt=data/model/bvlc_googlenet/deploy.prototxt
  pathToModel=data/model/bvlc_googlenet/bvlc_googlenet.caffemodel
  batchSize=64
  nodeNum=1
  dist/bin/bigdl.sh --                                                            \
  java -cp dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar      \
           com.intel.analytics.bigdl.example.loadmodel.ModelValidator             \
       -t $modelType -f $folder -m $modelName --caffeDefPath $pathToCaffePrototxt \
       --modelPath $pathToModel -b $batchSize --env local --node $nodeNum
  ```

## Run in spark Mode

In the spark mode example, we use transformed imagenet sequence file as input.

For caffe inception model and alexnet model, the command to transform the sequence file is (validation data is the different with the [BigDL inception Prepare the data](https://github.com/intel-analytics/BigDL/tree/master/dl/src/main/scala/com/intel/analytics/bigdl/models/inception#prepare-the-data))

```bash
dist/bin/bigdl.sh --
java -cp dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar \
         com.intel.analytics.bigdl.models.utils.ImageNetSeqFileGenerator   \
    -f <imagenet_folder> -o <output_folder> -p <cores_number> -r -v
```

For torch resnet model, the command to transform the sequence file is (validation data is the same with the [BigDL inception Prepare the data](https://github.com/intel-analytics/BigDL/tree/master/dl/src/main/scala/com/intel/analytics/bigdl/models/inception#prepare-the-data))

```bash
dist/bin/bigdl.sh --
java -cp dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar \
         com.intel.analytics.bigdl.models.utils.ImageNetSeqFileGenerator   \
     -f imagenet_folder -o output_folder -p cores_number -v
```

Then put the transformed sequence files to HDFS.

```
hdfs dfs -put <output_folder> <hdfs_folder>
```

For example, following the steps below will load BVLC GoogLeNet. 

+ Workspace  structure.

  ```
  [last: 0s][~/loadmodel]$ tree . -L 2
  .
  ├── data
  │   └── model
  ├── dist
  │   ├── bin
  │   └── lib
  └── imagenet
      └── val

  7 directories, 0 files
  [last: s][~/loadmodel]$ tree dist/
  dist/
  ├── bin
  │   ├── bigdl.sh
  │   ├── classes.lst
  │   └── img_class.lst
  └── lib
      ├── bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar
      └── bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar

  2 directories, 5 files
  [last: s][~/loadmodel]$ tree data/
  data/
  └── model
      └── bvlc_googlenet
          ├── bvlc_googlenet.caffemodel
          └── deploy.prototxt

  2 directories, 2 files
  ```


- Transform the validation data to sequence file.

  ```shell
  imagenet_folder=imagenet
  output_folder=seq
  cores_number=28
  dist/bin/bigdl.sh --                                                       \
  java -cp dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar \
           com.intel.analytics.bigdl.models.utils.ImageNetSeqFileGenerator   \
       -f ${imagenet_folder} -o ${output_folder} -p ${cores_number} -r -v
  hdfs dfs -put seq/val/ /
  ```


- Workspace on hdfs

  ```
  [last: s][~/loadmodel]$ hdfs dfs -ls /val
  Found 28 items
  -rw-r--r--   3 bigdl supergroup  273292894 2016-12-30 17:42 /val/imagenet-seq-0_0.seq
  -rw-r--r--   3 bigdl supergroup  276180279 2016-12-30 17:42 /val/imagenet-seq-10_0.seq
  -rw-r--r--   3 bigdl supergroup  252203135 2016-12-30 17:42 /val/imagenet-seq-11_0.seq
  -rw-r--r--   3 bigdl supergroup  276909503 2016-12-30 17:42 /val/imagenet-seq-12_0.seq
  -rw-r--r--   3 bigdl supergroup  283597155 2016-12-30 17:42 /val/imagenet-seq-13_0.seq
  -rw-r--r--   3 bigdl supergroup  271930407 2016-12-30 17:42 /val/imagenet-seq-14_0.seq
  -rw-r--r--   3 bigdl supergroup  277978520 2016-12-30 17:42 /val/imagenet-seq-15_0.seq
  -rw-r--r--   3 bigdl supergroup  274984408 2016-12-30 17:42 /val/imagenet-seq-16_0.seq
  -rw-r--r--   3 bigdl supergroup  265484517 2016-12-30 17:42 /val/imagenet-seq-17_0.seq
  -rw-r--r--   3 bigdl supergroup  266317348 2016-12-30 17:42 /val/imagenet-seq-18_0.seq
  -rw-r--r--   3 bigdl supergroup  265998818 2016-12-30 17:42 /val/imagenet-seq-19_0.seq
  ....
  ```

- Execute command.

  ```shell
  master=spark://xxx.xxx.xxx.xxx:xxxx # please set your own spark master
  modelType=caffe
  folder=/
  modelName=inception
  pathToCaffePrototxt=data/model/bvlc_googlenet/deploy.prototxt
  pathToModel=data/model/bvlc_googlenet/bvlc_googlenet.caffemodel
  batchSize=448
  nodeNum=8
  dist/bin/bigdl.sh --                                                                     \
  spark-submit --driver-memory 20g --master $master --executor-memory 100g                 \
               --driver-class-path dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
               --class com.intel.analytics.bigdl.example.loadmodel.ModelValidator          \
                       dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar             \
              -t $modelType -f $folder -m $modelName --caffeDefPath $pathToCaffePrototxt   \
              --modelPath $pathToModel -b $batchSize --env spark --node $nodeNum
  ```

where 

```--node``` is the number of nodes to test the model

other parameters have the same meaning as local mode.


## Expected Results

You should get similar top1 or top5 accuracy as the original model.
