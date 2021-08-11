# Load Pre-trained Model

This example demonstrates how to use BigDL to load pre-trained [Torch](http://torch.ch/) or [Caffe](http://caffe.berkeleyvision.org/) model into Spark program for prediction.

**ModelValidator** provides an integrated example to load models, and test over imagenet validation dataset on Spark.

For most CNN models, it's recommended to enable MKL-DNN acceleration by specifying `bigdl.engineType` as `mkldnn` for model validation.

## Preparation

To start with this example, you need prepare your model, dataset.

The caffe model used in this example can be found in 
[GoogleNet Caffe Model](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet)
and [Alexnet Caffe Model](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet).

The torch model used in this example can be found in
[Resnet Torch Model](https://github.com/facebook/fb.resnet.torch/tree/master/pretrained#trained-resnet-torch-models).

The imagenet validation dataset preparation can be found from
[BigDL inception Prepare the data](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/inception#prepare-the-data).

## Run as a Spark program

When running as a Spark program, we use transformed imagenet sequence file as input.

For Caffe Inception model and Alexnet model, the command to transform the sequence file is (validation data is the different with the [BigDL inception Prepare the data](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/inception#prepare-the-data))

```bash
java -cp dist/lib/bigdl-VERSION-jar-with-dependencies-and-spark.jar \
         com.intel.analytics.bigdl.models.utils.ImageNetSeqFileGenerator   \
    -f <imagenet_folder> -o <output_folder> -p <cores_number> -r -v
```

For Torch Resnet model, the command to transform the sequence file is (validation data is the same with the [BigDL Inception Prepare the data](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/inception#prepare-the-data))

```bash
java -cp dist/lib/bigdl-VERSION-jar-with-dependencies-and-spark.jar \
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
  │   ├── classes.lst
  │   └── img_class.lst
  └── lib
      ├── bigdl-VERSION-jar-with-dependencies-and-spark.jar
      └── bigdl-VERSION-jar-with-dependencies.jar

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
  java -cp dist/lib/bigdl-VERSION-jar-with-dependencies-and-spark.jar \
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

- Execute command for Spark standalone mode.
```shell
  master=spark://xxx.xxx.xxx.xxx:xxxx # please set your own spark master
  engine=... # mklblas/mkldnn. For most cnn models, you can set bigdl.engineType as mkldnn to get better performance.
  modelType=caffe
  folder=hdfs://...
  modelName=inception
  pathToCaffePrototxt=data/model/bvlc_googlenet/deploy.prototxt
  pathToModel=data/model/bvlc_googlenet/bvlc_googlenet.caffemodel
  batchSize=448
  spark-submit --driver-memory 20g --master $master --executor-memory 100g                 \
               --executor-cores 28                                                         \
               --total-executor-cores 112                                                  \
               --conf "spark.driver.extraJavaOptions=-Dbigdl.engineType=$engine"  \
               --conf "spark.executor.extraJavaOptions=-Dbigdl.engineType=$engine"  \
               --driver-class-path dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
               --class com.intel.analytics.bigdl.example.loadmodel.ModelValidator          \
                       dist/lib/bigdl-VERSION-jar-with-dependencies.jar             \
              -t $modelType -f $folder -m $modelName --caffeDefPath $pathToCaffePrototxt   \
              --modelPath $pathToModel -b $batchSize
```

- Execute command for Spark yarn mode.
```shell
  modelType=caffe
  folder=hdfs://...
  engine=... # mklblas/mkldnn. For most cnn models, you can set bigdl.engineType as mkldnn to get better performance.
  modelName=inception
  pathToCaffePrototxt=data/model/bvlc_googlenet/deploy.prototxt
  pathToModel=data/model/bvlc_googlenet/bvlc_googlenet.caffemodel
  batchSize=448
  spark-submit --driver-memory 20g --master yarn --executor-memory 100g                    \
               --deploy-mode client
               --conf "spark.yarn.am.extraJavaOptions=-Dbigdl.engineType=$engine" \
               --conf "spark.executor.extraJavaOptions=-Dbigdl.engineType=$engine" \
               --executor-cores 28                                                         \
               --num-executors 4                                                  \
               --driver-class-path dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
               --class com.intel.analytics.bigdl.example.loadmodel.ModelValidator          \
                       dist/lib/bigdl-VERSION-jar-with-dependencies.jar             \
              -t $modelType -f $folder -m $modelName --caffeDefPath $pathToCaffePrototxt   \
              --modelPath $pathToModel -b $batchSize
```

## Expected Results

You should get similar top1 or top5 accuracy as the original model.
