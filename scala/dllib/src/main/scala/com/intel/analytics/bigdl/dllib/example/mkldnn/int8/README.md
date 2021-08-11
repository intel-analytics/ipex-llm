# Inference acceleration with MKL-DNN Low Numerical Precision (Int8) Computing

You can use the mkldnn version of low numerical precision inference by the
API `quantize()` which will give you better performance on Intel Xeon Scalable
processors. There're only two steps, scale generation and quantize the model.
Often you can combine them into one, but for demonstrate the process of how to
use it.

## Generate the Scales of Pretrained Model

If you use a BigDL model which is trained by yourself or converted from other
frameworks. You should generate the scales first. It needs some sample images
to do the `forward` which can be the test or validation dataset. And because
it's the sample images, you need no to pass the whole validate dataset. And of
cause you can use spark local mode to generate scales.

After that, you can call `GenerateInt8Scales`, it will generate a model with
a `quantized` in the name. It's the original model combined with scales information.

```bash
#!/bin/bash

MASTER="local[1]"

EXECUTOR_CORES=32
DRIVER_MEMORY=50G
EXECUTOR_MEMORY=100G

EXECUTOR_CORES=32
TOTAL_EXECUTOR_CORES=${EXECUTOR_CORES}
BATCH_SIZE=128

BIGDL_VERSION=0.8.0

VAL_FOLDER=hdfs://xxx.xxx.xxx.xxx:xxxx/imagenet-noresize/val
MODEL=./resnet-50.bigdl

spark-submit \
  --master ${MASTER} \
  --driver-memory ${DRIVER_MEMORY} \
  --executor-memory ${EXECUTOR_MEMORY} \
  --executor-cores ${EXECUTOR_CORES} \
  --total-executor-cores ${TOTAL_EXECUTOR_CORES} \
  --class com.intel.analytics.bigdl.example.mkldnn.int8.GenerateInt8Scales \
  ./dist/lib/bigdl-${BIGDL_VERSION}-jar-with-dependencies.jar \
  -f  ${VAL_FOLDER} \
  --batchSize ${BATCH_SIZE}  \
  --model ${MODEL}
```

## Do the Evaluation on the Quantized Model

When you prepared the relative quantized model, it's very simple to use int8 based
on mkldnn. On your loaded model, to call `quantize()`, it will return a new
quantized model. Now, you can do inference like other models. You could enable the
model fusion by java property, `-Dbigdl.mkldnn.fusion=true`, which works for most
CNN models and you can normally get performance benifit.

## Use different engine to quantize the model

You can use `bigdl.engineType` to set different engine to do the quantize. If you
set the engine to `mklblas`, it will use bigquant to quantize the model, otherwise
will use the mkldnn int8

## Command to startup

```bash
#!/bin/bash

MASTER=spark://xxx.xxx.xxx.xxx:xxxx

EXECUTOR_CORES=32
DRIVER_MEMORY=50G
EXECUTOR_MEMORY=100G

EXECUTOR_CORES=32
EXECUTOR_NUMBER=4 # executor number you want
TOTAL_EXECUTOR_CORES=$((EXECUTOR_CORES * EXECUTOR_NUMBER))
BATCH_SIZE=$((TOTAL_EXECUTOR_CORES * 4))

BIGDL_VERSION=0.8.0

VAL_FOLDER=hdfs://xxx.xxx.xxx.xxx:xxxx/imagenet-noresize/val
MODEL=./resnet-50.bigdl.quantized

spark-submit \
  --master ${MASTER} \
  --driver-memory ${DRIVER_MEMORY} \
  --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" \
  --conf "spark.network.timeout=1000000" \
  --conf "spark.driver.extraJavaOptions=-Dbigdl.engineType=mkldnn -Dbigdl.mkldnn.fusion=true" \
  --conf "spark.executor.extraJavaOptions=-Dbigdl.engineType=mkldnn -Dbigdl.mkldnn.fusion=true" \
  --executor-memory ${EXECUTOR_MEMORY} \
  --executor-cores ${EXECUTOR_CORES} \
  --num-executors ${EXECUTOR_NUMBER} \
  --total-executor-cores ${TOTAL_EXECUTOR_CORES} \
  --class com.intel.analytics.bigdl.example.mkldnn.int8.TestImageNet \
  ./dist/lib/bigdl-${BIGDL_VERSION}-jar-with-dependencies.jar \
  -f  ${VAL_FOLDER} \
  --batchSize ${BATCH_SIZE}  \
  --model ${MODEL}
```
