# BigDL-Nano ONNX Runtime Example - Pytorch/MNIST

This example is adapted from https://github.com/pytorch/examples/blob/main/mnist/main.py which uses a self-defined CNN to train and test the classification task of MNIST.

In this example you may learn how to use onnxruntime easily in bigdl-nano to...

- Accerlate your inferencing
- Validate the accuracy of onnxruntime backend
- Export onnx file for future deployment

## Prepare the environment
We recommend you to use Anaconda to prepare the environment, especially if you want to run on a yarn cluster:
```bash
conda create -n my_env python=3.7 # "my_env" is conda environment name, you can use any name you like.
conda activate my_env
pip install --pre --upgrade bigdl-nano[pytorch]
```

## Prepare data
We are using the MNIST dataset. There is no need to prepare the dataset since the example script will download it for you.

## Run the example
```bash
python mnist_ort.py --save-model
```

## Sample output
## 
```bash
Train Epoch: 1 [0/60000 (0%)]           Loss: 2.305400
Train Epoch: 1 [640/60000 (1%)]         Loss: 1.359781
Train Epoch: 1 [1280/60000 (2%)]        Loss: 0.830733
Train Epoch: 1 [1920/60000 (3%)]        Loss: 0.613777
# ...
Test set: Average loss: 0.0335, Accuracy: 9880/10000 (99%)
test{'onnx': False} took 2.0284812450408936 seconds.
Test set: Average loss: 0.0335, Accuracy: 9880/10000 (99%)
test{'onnx': True} took 1.8070614337921143 seconds.
# ...
```