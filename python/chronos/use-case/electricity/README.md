# Use Chronos Forecasters on Electricity dataset.



In this use-case, we will use Chronos Forecasters(TCN, Autoformer) on electricity dataset. Electricity dataset is a widely used public dataset for Time series forecasting in engineering validation and researching.

For API docs of TCNForecaster and AutoformerForecaster, please refer to
[Chronos Forecasters API Document](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/forecasters.html)



## Prepare the environment



We recommend you to use conda to prepare the environment.



```bash
conda create -n my_env python=3.7 setuptools=58.0.4 # "my_env" is conda environment name, you can use any name you like.
conda activate my_env
pip install --pre --upgrade bigdl-chronos[pytorch,inference]
source bigdl-nano-init # accelerate the environment
```



For more detailed information, please refer to [Chronos Install Guide](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/install.html)



## Prepare data



We are using the **Electricity** data with the preprocessing aligned with [this paper](https://arxiv.org/abs/2106.13008).



Download link: [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/e1ccfff39ad541908bae/) or [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing).

You only need to download `electricity.csv`.



## Run the example



```bash
python tcn.py # for tcn forecaster
python autoformer.py # for autoformer forecaster
```



## Output

### TCNForecaster
After you run the code, the training process is shown like this:
```bash
Epoch 0:  28%|██████████████████████████                                                                  | 156/550 [00:09<00:25, 15.62it/s, loss=1.13]
```

After training 30 epochs, MSE is shown like this,
```bash
MSE is: 0.5876001
```

and the inference latency is shown like this:
```bash
Inference latency is: 0.00480920190349875
Inference latency with onnx is: 0.002660592173564555
```

### AutoformerForecaster

After you run the code, the training process is shown like this:
```bash
Epoch 0:   4%|████                                                                                         | 24/550 [00:09<03:34,  2.46it/s, loss=1.01]
```

After training 3 epochs, MSE is shown like this,
```bash
MSE on test dataset: [{'val_loss': 0.30125972628593445}]
```
*Note: Autoformer suffers overfitting problem on electricity dataset.*

and the inference latency is shown like this:
```bash
latency(8 cores): 0.010854396792372106
latency(1 cores): 0.019527499927325033
```



