# Serve a forecaster on TorchServe

Model serving is for model deployment in a production environment for wide accessibility.

This example shows how to serve Chronos forecaster and predict through TorchServe. We will take `TCNForecaster` and nyc_taxi dataset in this example.

## Setup environment
We recommend to use Anaconda to setup the environment:
```bash
conda create -n my_env python=3.7 setuptools=58.0.4
conda activate my_env
pip install --pre --upgrade bigdl-chronos[pytorch]
```

Please refer to [Chronos Install Guide](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/install.html) for more information.

After installing Chronos, please refer to [Quick start with TorchServe](https://github.com/pytorch/serve/blob/master/README.md#serve-a-model) to install TorchServe.

> **Note**:
> Chronos only supports Python 3.7.2 ~ latest 3.7.x currently. To install dependencies of TorchServe, we need to run install_dependencies.py with tag [v0.6.0](https://github.com/pytorch/serve/tree/v0.6.0).


## Prepare forecaster
Create and train a `TCNForecaster` based on nyc_taxi dataset, then save the pth file with path "./checkpoint/ckpt.pth".
```bash
python ./generate_torchscript_pt.py
```

## Archive forecasting model
To serve a model with TorchServe, archive the model as a mar file.
```bash
torch-model-archiver --model-name tcn_nyctaxi --version 1.1 --serialized-file ./checkpoint/ckpt.pth --handler ./model_handler:entry_point_function_name --export-path ./model_store/
```
The parameter `serialized-file` is the pth file name and `export-path` is the output mar file path. Please refer to [Torch Model archiver for TorchServe](https://github.com/pytorch/serve/blob/master/model-archiver/README.md) for more information.

## Serve forecaster
When the mar file is ready, use the following command to serve the forecaster:
```bash
torchserve --start --ncs --model-store model_store --models tcn_nyctaxi.mar
```
Then TorchServe runs and listens for inference requests.

## Predict
To test the forecaster server, send an inference request through HTTP.
```bash
curl http://localhost:8080/predictions/tcn_nyctaxi -T data
```
Then return the prediction results.

## Stop serving
To stop serving, just run the command:
```bash
torchserve --stop
```
