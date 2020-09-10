# OpenVINO ResNet_v1_50 example
This example illustrates how to use a pre-trained OpenVINO optimized model to make inferences with OpenVINO toolkit as backend using Analytics Zoo. We hereby illustrate the support of [VNNI](https://en.wikichip.org/wiki/x86/avx512vnni) using [OpenVINO](https://software.intel.com/en-us/openvino-toolkit) as backend in Analytics Zoo, which aims at accelerating inference by utilizing low numerical precision (Int8) computing. Int8 quantized models can generally give you better performance on Intel Xeon scalable processors.

## Environment
* Apache Spark (This version needs to be same with the version you use to build Analytics Zoo)
* [Analytics Zoo](https://analytics-zoo.github.io/master/#PythonUserGuide/install/)

- Set `ZOO_NUM_MKLTHREADS` to determine cores used by OpenVINO, e.g, `export ZOO_NUM_MKLTHREADS=10`. If it is set to `all`, e.g., `export ZOO_NUM_MKLTHREADS=all`, then OpenVINO will utilize all physical cores for Prediction.
- Set `KMP_BLOCKTIME=200`, i.e., `export KMP_BLOCKTIME=200`

## Prepare OpenVINO Model
TensorFlow models cannot be directly loaded by OpenVINO. It should be converted to OpenVINO optimized model and int8 optimized model first. Please install [OpenVINO](https://software.intel.com/en-us/openvino-toolkit), and covert models with `downloader` tool in [OpenVINO Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo). This tool is located in OpenVINO installation directory (`deployment_tools/open_model_zoo`).

The model we used is named [resnet-50-tf](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/resnet-50-tf/resnet-50-tf.md).

If you prefer GUI workflow, you can use [OpenVINO DL Workbench](https://docs.openvinotoolkit.org/latest/_docs_Workbench_DG_Install_Workbench.html) with Docker container.

__Sample Result files in MODEL_PATH__:
```
resnet_v1_50.xml
resnet_v1_50.bin
resnet_v1_50_i8.xml
resnet_v1_50_i8.bin
```

Among them, `resnet_v1_50.xml` and `resnet_v1_50.bin` are OpenVINO optimized ResNet_v1_50 model and weight, `resnet_v1_50_i8.xml` and `resnet_v1_50_i8.bin` are OpenVINO int8 optimized ResNet_v1_50 model and weight. Both of them can be loaded by OpenVINO or Zoo.

__Note that int8 optimized model promises better performance (~2X) with slightly lower accuracy.__

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
