# BigDL-Nano Transfer Learning Example with Tensorflow

This example describes how to use BigDL Nano to optimize a TensorFlow transfer learning workload.

This example is migrated from the TensorFlow tutorial notebook at 
https://github.com/tensorflow/docs/blob/r2.4/site/en/tutorials/images/transfer_learning.ipynb

## Quick Start
1. Prepare Environment

    You can install the necessary packages with the following command
    
    ```
    pip install bigdl-nano[tensorflow]
    ```

    Then setup the environment with the script `bigdl-nano-init`:

    ```
    source bigdl-nano-init
    ```

2. Run the Example

    You can run this example in your conda environment with the following command:
    ```
    python  transfer_learning.py
    ```


## Workflow and Results
We use the MobileNetV2 network as our base model combined with "Preprocessing" and "Prediction" Layers. The initial loss and accuracy will be printed before training steps like this:

```
Number of trainable variables: 2
26/26 [==============================] - 5s 87ms/step - loss: 0.9020 - accuracy: 0.4468
initial loss: 0.90
initial accuracy: 0.45
```

Then we freeze the base model and train other layers in the model. After 10 epoches of training, we unfreeze the model except the first 100 layers of the base model and continue to train. We will get the final accuracy after the evaluation of the model at the end of the workflow.

```
Number of trainable variables now: 56
Epoch 10/20
63/63 [==============================] - 14s 172ms/step - loss: 0.1472 - accuracy: 0.9390 - val_loss: 0.0623 - val_accuracy: 0.9790
Epoch 11/20
63/63 [==============================] - 10s 149ms/step - loss: 0.1188 - accuracy: 0.9480 - val_loss: 0.0571 - val_accuracy: 0.9802
...
```

After the training, we evaluate the model. You can check the accuracy at the end of workflow. 

```
6/6 [==============================] - 1s 66ms/step - loss: 0.0351 - accuracy: 0.9844
Test accuracy : 0.984375
```

