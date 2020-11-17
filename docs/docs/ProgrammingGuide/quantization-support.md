## Introduction

Quantization is a method that will use low-precision caculations to substitute float caculations. It will improve the inference performance and reduce the size of model by up to 4x.

## Quantize the pretrained model

BigDL provide command line tools for converting the pretrained
(BigDL, Caffe, Torch and Tensorflow) model to quantized model with
parameter `--quantize true`. You can modify the script below to convert a model to
quantized model.

By default, you should set some shell variables below. The `BIGDL_HOME` is the 
`dist` directory. If the spark version you use is after 2.0, you should add
spark jars directory to `SPARK_JAR`.

```bash
#!/bin/bash

set -x

VERSION=0.13.0-SNAPSHOT
BIGDL_HOME=${WORKSPACE}/dist
JAR_HOME=${BIGDL_HOME}/lib/target
SPARK_JAR=/opt/spark/jars/*
JAR=${JAR_HOME}/bigdl-${VERSION}-jar-with-dependencies.jar:${SPARK_JAR}
```

For example, we want to convert a caffe model to a bigdl model.

```bash
FROM=caffe
TO=bigdl
MODEL=bvlc_alexnet.caffemodel
```

And the last commands are as follows.

```bash
CLASS=com.intel.analytics.bigdl.utils.ConvertModel


java -cp ${JAR} ${CLASS} --from ${FROM} --to ${TO} \
    --input ${MODEL} --output ${MODEL%%.caffemodel}.bigdlmodel \
    --prototxt ${PWD}/deploy.prototxt --quantize true
```

` ConvertModel` supports converting different types of pretrained models to bigdlmodel.
It also supports converting bigdlmodel to other types. The help is

```
Usage: Convert models between different dl frameworks [options]

  --from <value>
        What's the type origin model bigdl,caffe,torch,tensorflow?
  --to <value>
        What's the type of model you want bigdl,caffe,torch?
  --input <value>
        Where's the origin model file?
  --output <value>
        Where's the bigdl model file to save?
  --prototxt <value>
        Where's the caffe deploy prototxt?
  --quantize <value>
        Do you want to quantize the model? Only works when "--to" is bigdl;you can only perform inference using the new quantized model.
  --tf_inputs <value>
        Inputs for Tensorflow
  --tf_outputs <value>
        Outputs for Tensorflow

```

## Quantize model in code

You can call `quantize()` method to quantize the model. It will deep copy original model and generate new one. You can only perform inference using the new quantized model.

```scala
val model = xxx
val quantizedModel = model.quantize()
quantizeModel.forward(inputTensor)
```

There's also a Python API which is same as scala version.

```python
model = xxx
quantizedModel = model.quantize()
```
