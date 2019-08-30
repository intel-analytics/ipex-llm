# OpenVINO ResNet_v1_50 example
This example illustrates how to use a pre-trained OpenVINO optimized model to make inferences with OpenVINO toolkit as backend using Analytics Zoo. We hereby illustrate the support of [VNNI](https://en.wikichip.org/wiki/x86/avx512vnni) using [OpenVINO](https://software.intel.com/en-us/openvino-toolkit) as backend in Analytics Zoo, which aims at accelerating inference by utilizing low numerical precision (Int8) computing. Int8 quantized models can generally give you better performance on Intel Xeon scalable processors.
 
## Environment
* Apache Spark (This version needs to be same with the version you use to build Analytics Zoo)
* [Analytics Zoo](https://analytics-zoo.github.io/master/#PythonUserGuide/install/)

- Set `ZOO_NUM_MKLTHREADS` to determine cores used by OpenVINO, e.g, `export ZOO_NUM_MKLTHREADS=10`. If it is set to `all`, e.g., `export ZOO_NUM_MKLTHREADS=all`, then OpenVINO will utilize all physical cores for Prediction.
- Set `KMP_BLOCKTIME=200`, i.e., `export KMP_BLOCKTIME=200`

## PrepareOpenVINOResNet
TensorFlow models cannot be directly loaded by OpenVINO. It should be converted to OpenVINO optimized model and int8 optimized model first. You can use [PrepareOpenVINOResNet](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/vnni/openvino) or [OpenVINO toolkit](https://docs.openvinotoolkit.org/2018_R5/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html) to finish this job.

__Sample Result files in MODEL_PATH__:
```
resnet_v1_50.ckpt
resnet_v1_50_inference_graph.bin
resnet_v1_50_inference_graph-calibrated.bin
resnet_v1_50_inference_graph-calibrated.xml
resnet_v1_50_inference_graph.mapping
resnet_v1_50_inference_graph.xml
```

Among them, `resnet_v1_50_inference_graph.xml` and `resnet_v1_50_inference_graph.bin` are OpenVINO optimized ResNet_v1_50 model and weight, `resnet_v1_50_inference_graph-calibrated.xml` and `resnet_v1_50_inference_graph-calibrated.bin` are OpenVINO int8 optimized ResNet_v1_50 model and weight. Both of them can be loaded by OpenVINO or Zoo.

## Image Classification with ResNet_v1_50

```
python predict.py --image ${image} --model ${model}
```

### Options
* `--image` The path where the images are stored. It can be either a folder or an image path. Local file system, HDFS and Amazon S3 are supported.
* `--model` The path to the TensorFlow object detection model.

### Results
We print the inference result of each batch.
```
[ INFO ] Start inference (1 iterations)

batch_0
* Predict result {'Top-1': '67'}
* Predict result {'Top-1': '65'}
* Predict result {'Top-1': '334'}
* Predict result {'Top-1': '795'}
```

