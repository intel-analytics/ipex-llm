# Inference acceleration with MKL-DNN Low Numerical Precision (Int8) Computating


You can use the mkldnn version of low numerical precision inference by the
API `quantize()` which will give you better performance on Intel Xeon Scalable
processors. There're only two steps, scale generation and quantize the model.

## Generate the Scales of Pretrained Model

If you use a BigDL model which is trained by yourself or converted from other
frameworks. You should generate the scales first. It needs some sample images
to do the `forward` which can be the test or validation dataset.

After that, you can call `GenerateInt8Scales`, it will generate a model with
a `quantized` as the suffix. It's the original model with scales information.

## Do the Evaluation on the Quantized Model

When you prepared the relative quantized model, it's very simple to use int8 based
on mkldnn. On your loaded model, to call `quantize()`, it will return a new
quantized model. Now, you can do inference like other models. You should enable the
model fusion by java property, `-Dbigdl.mkldnn.fusion=true`, which will have the
best performance.

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
  --class com.intel.analytics.bigdl.example.mkldnnint8.TestImageNet \
  ./dist/lib/bigdl-${BIGDL_VERSION}-jar-with-dependencies.jar \
  -f  ${VAL_FOLDER} \
  --batchSize ${BATCH_SIZE}  \
  --model ${MODEL}
```
