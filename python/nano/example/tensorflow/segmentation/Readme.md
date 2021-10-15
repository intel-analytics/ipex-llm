# BigDL-Nano Image Segmentation Example with Tensorflow

This example describes how to use BigDL Nano to optimize a TensorFlow image segmentation example. 

## Quick Start 

1. Setup Environment

    Install the BigDL-Nano with Tensorflow support with `pip` in a conda environment. 

    ```
    pip install bigdl-nano[tensorflow]
    ```

    Then install Tensorflow Examples for `pix2pix` model and Tensorflow Datasets for `oxford_iiit_pet` dataset support. 

    ```
    pip install git+https://github.com/tensorflow/examples.git
    pip install tensorflow_datasets
    ```

    You may need to install the `pydot` and `graphviz` for `plot_model/model_to_dot` if you want to plot the model with `tf.keras.utils.plot_model`. You can install with the following command in Ubuntu 18.04+

    ```
    pip install pydot
    sudo apt install graphviz
    ```

    Then setup the environment with the script `bigdl-nano-init`:

    ```
    source bigdl-nano-init
    ```

2. Run the Example

    You can run this example in your conda environment with the following command:
    
    ```
    python segmentation.py
    ```

## Result
You can check the evaluation result at the end of the running output.
```
58/58 [==============================] - 12s 211ms/step - loss: 0.3430 - accuracy: 0.8843
```