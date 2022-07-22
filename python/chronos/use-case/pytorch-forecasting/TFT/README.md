# Use Chronos to help pytorch-forecasting improve the training/inference speed of TFT model
Chronos can help a 3rd party time series lib to improve the performance (both training and inferencing) and accuracy. This use-case shows nano can easily help pytorch-forecasting speed up the training/inference of TFT (Temporal Fusion Transformers) model.

More detailed information please refer to: https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/speed_up.html

## Prepare the environment
We recommend you to use conda to prepare the environment, especially if you want to run on a yarn cluster:
```bash
conda create -n my_env python=3.7 # "my_env" is conda environment name, you can use any name you like.
conda activate my_env
pip install --pre --upgrade bigdl-chronos[all]
pip install pytorch_forecasting
pip install torch==1.11.0  # if your pytorch_forecasting version is 0.10.0 or above, you need to reinstall torch, otherwise you don't need this.
pip install intel_extension_for_pytorch
```
Please refer to [Chronos Install Guide](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/chronos.html#install)

## Prepare data
We are using the [Stallion dataset from Kaggle](https://www.kaggle.com/datasets/utathya/future-volume-prediction). The data will be loaded as pandas dataframe automatically.

## Run the example
```bash
bigdl-nano-init python tft.py
```
## Training
### Changes to use Chronos TSTrainer
- Change `from pytorch_lightning import Trainer` to `from bigdl.chronos.pytorch import TSTrainer`
- Set `gpus=0` in TSTrainer
- Set `num_processes=8` in TSTrainer and set `batch_size = 128 // num_processes`

### Results
In an experimental platform, the training speed of TFT model using nano Trainer is 3.5 times the speed of the training without nano Trainer. We can see that the training speed is significantly improved.

## Inferencing
**Currently inference speed-up is still on activate developing. Some definition in pytorch-forecasting's TFT needs to be changed.**

### Changes to use Chronos TSTrainer
Inside pytorch-forecasting's definition, we need to remove some code in `models/temporal_fusion_transformer/__init__.py`
```python
# comment all the `lengths` and `enforce_sorted` in code(2 in all)

# run local encoder
encoder_output, (hidden, cell) = self.lstm_encoder(
    embeddings_varying_encoder, (input_hidden, input_cell)  #, lengths=encoder_lengths, enforce_sorted=False
)

# run local decoder
decoder_output, _ = self.lstm_decoder(
    embeddings_varying_decoder,
    (hidden, cell),
    #  lengths=decoder_lengths,
    #  enforce_sorted=False,
)
```
Then you may enjoy the accleration provided by `TSTrainer`
```python
best_tft = TSTrainer.optimize(best_tft)
best_tft(x)
```
### Results
For an input of (128, 24, 7) and output of (128, 6, 7), the acceleration is ~1.5X on a single core.
