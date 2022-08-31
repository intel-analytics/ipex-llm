# Use Chronos Forecasters on Electricity dataset.



In this use-case, we will use Chronos Forecasters(TCN, Autoformer) on electricity dataset. Electricity dataset is a widely used public dataset for Time series forecasting in engineering validation and researching.

For API docs of TCNForecaster and AutoformerForecaster, please refer to
[Chronos Forecasters API Document](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/forecasters.html)



## Prepare the environment



We recommend you to use conda to prepare the environment.



```bash
conda create -n my_env python=3.7 setuptools=58.0.4 # "my_env" is conda environment name, you can use any name you like.
conda activate my_env
pip install --pre --upgrade bigdl-chronos[all]
pip install torch==1.11.0 # for better performance
source bigdl-nano-init # accelerate the environment
```



For more detailed information, please refer to [Chronos Install Guide](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/chronos.html#install)



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
Epoch 0:  17%|████████▏                                       | 94/550 [00:08<00:39, 11.61it/s, loss=0.95]
```

After training 30 epochs, MSE is shown like this,
```bash
MSE is: [array(0.48502076, dtype=float32)]
```

and the inference latency is shown like this:
```bash
Inference latency is: 0.0060901641845703125
Inference latency with onnx is: 0.0030126571655273438
```

### AutoformerForecaster

After you run the code, the training process is shown like this:
```bash
Epoch 0:   2%|██▏                                                                                                                   | 10/550 [00:02<02:08,  4.19it/s, loss=1.18]
```

After training 3 epochs, MSE is shown like this,
```bash
MSE on test dataset: [{'val_loss': 0.2887305021286011}]
```
*Note: Autoformer suffers overfitting problem on electricity dataset.*

and the inference latency is shown like this:
```bash
latency(8 cores): 0.013892467462774168
latency(1 cores): 0.022499848716141295
```



