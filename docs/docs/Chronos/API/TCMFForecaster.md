## **Introduction**

Analytics Zoo Chronos TCMFForecaster provides an efficient way to forecast high dimensional time series. 

TCMFForecaster is based on DeepGLO algorithm, which is a deep forecasting model which thinks globally and acts locally.
You can refer to [the deepglo paper](https://arxiv.org/abs/1905.03806) for more details. 

TCMFForecaster supports distributed training and inference. It is based on Orca PyTorch Estimator, which is an estimator to do PyTorch training/evaluation/prediction on Spark in a distributed fashion. Also you can choose to enable distributed training and inference or not.

__Remarks__:

- You can refer to [TCMFForecaster installation](../tutorials/TCMFForecaster/#step-0-prepare-environment) to install required packages.
- Your operating system (OS) is required to be one of the following 64-bit systems:
__Ubuntu 16.04 or later__ and __macOS 10.12.6 or later__.
---

### TCMFForecaster


### Create TCMFForecaster


```python
from zoo.chronos.forecaster.tcmf_forecaster import TCMFForecaster
model = TCMFForecaster(
         vbsize=128,
         hbsize=256,
         num_channels_X=[32, 32, 32, 32, 32, 1],
         num_channels_Y=[16, 16, 16, 16, 16, 1],
         kernel_size=7,
         dropout=0.1,
         rank=64,
         kernel_size_Y=7,
         learning_rate=0.0005,
         normalize=False,
         use_time=True,
         svd=True,)
```
* `vbsize`: int, default is 128.
            Vertical batch size, which is the number of cells per batch.
* `hbsize`: int, default is 256.
            Horizontal batch size, which is the number of time series per batch.
* `num_channels_X`: list, default=[32, 32, 32, 32, 32, 1].
            List containing channel progression of temporal convolution network for local model
* `num_channels_Y`: list, default=[16, 16, 16, 16, 16, 1]
            List containing channel progression of temporal convolution network for hybrid model.
* `kernel_size`: int, default is 7.
            Kernel size for local models
* `dropout`: float, default is 0.1.
            Dropout rate during training
* `rank`: int, default is 64.
            The rank in matrix factorization of global model.
* `kernel_size_Y`: int, default is 7.
            Kernel size of hybrid model
* `learning_rate`:  float, default is 0.0005
* `normalize`: boolean, false by default.
            Whether to normalize input data for training.
* `use_time`: boolean, default is True.
            Whether to use time coveriates.
* `svd`: boolean, default is False.
            Whether factor matrices are initialized by NMF

### Use TCMFForecaster
#### **Train model**
After an TCMFForecaster is created, you can call forecaster API to train a tcmf model:
```python
model.fit(x,
          val_len=24,
          start_date="2020-4-1",
          freq="1H",
          covariates=None,
          dti=None,
          period=24,
          y_iters=10,
          init_FX_epoch=100,
          max_FX_epoch=300,
          max_TCN_epoch=300,
          alt_iters=10,
          num_workers=None)
```
* `x`: the input for fit. Only dict of ndarray and SparkXShards of dict of ndarray
       are supported. Example: {'id': id_arr, 'y': data_ndarray}. If input is SparkXShards, each partition will use one model to fit.
* `val_len`:int, default is 24.
            Validation length. We will use the last val_len time points as validation data.
* `start_date`: str or datetime-like.
            Start date time for the time-series. e.g. "2020-01-01"
* `freq`: str or DateOffset, default is 'H'
            Frequency of data
* `covariates`: 2-D ndarray or None. The shape of ndarray should be (r, T), where r is
            the number of covariates and T is the number of time points.
            Global covariates for all time series. If None, only default time coveriates will be
            used while use_time is True. If not, the time coveriates used is the stack of input
            covariates and default time coveriates.
* `dti`: DatetimeIndex or None.
            If None, use default fixed frequency DatetimeIndex generated with start_date and freq.
* `period`: int, default is 24.
            Periodicity of input time series, leave it out if not known
* `y_iters`: int, default is 10.
            Number of iterations while training the hybrid model.
* `init_FX_epoch`: int, default is 100.
            Number of iterations while initializing factors
* `max_FX_epoch`: int, default is 300.
            Max number of iterations while training factors.
* `max_TCN_epoch`: int, default is 300.
            Max number of iterations while training the local model.
* `alt_iters`: int, default is 10.
            Number of iterations while alternate training.
* `num_workers`: the number of workers you want to use for fit. It is only effective while input x is dict of ndarray. If None, it defaults to
        num_ray_nodes in the created RayContext or 1 if there is no active RayContext.

#### **Get prediction results of model**
After Training, you can call forecaster API to get the prediction result of tcmf model. `model.predict` will output the prediction results of future `horizon` steps after `x` in `fit`.
```
model.predict(horizon=24,
              future_covariates=None,
              future_dti=None,
              num_workers=None,
              )
```
* `future_covariates`: covariates corresponding to future horizon steps data to predict.
        2-D ndarray or None.
        The shape of ndarray should be (r, horizon), where r is the number of covariates.
        Global covariates for all time series. If None, only default time coveriates will be
        used while use_time is True. If not, the time coveriates used is the stack of input
        covariates and default time coveriates.
* `future_dti`: dti corresponding to future horizon steps data to predict.
        DatetimeIndex or None.
        If None, use default fixed frequency DatetimeIndex generated with the last date of x in
        fit and freq.
* `num_workers`: the number of workers to use in predict. It is only effective while input `x` in `fit` is dict of ndarray. If None, it defaults to
        num_ray_nodes in the created RayContext or 1 if there is no active RayContext.

#### **Evaluate model**
After Training, you can call forecaster API to evaluate the tcmf model. `model.evaluate` will output the evaluation results for future `horizon` steps after `x` in `fit`.
```
model.evaluate(target_value,
               metric=['mae'],
               target_covariates=None,
               target_dti=None,
               num_workers=None,
               )
```
* `target_value`: target value for evaluation. It should be of the same format as input x in fit, which is a dict of ndarray or SparkXShards of dict of ndarray.
                  We interpret the second dimension of y in target value as the horizon length for evaluation.
* `metric`: the metrics. A list of metric names.
* `target_covariates`: covariates corresponding to target_value.
        2-D ndarray or None.
        The shape of ndarray should be (r, horizon), where r is the number of covariates.
        Global covariates for all time series. If None, only default time coveriates will be
        used while use_time is True. If not, the time coveriates used is the stack of input
        covariates and default time coveriates.
* `target_dti`: dti corresponding to target_value.
        DatetimeIndex or None.
        If None, use default fixed frequency DatetimeIndex generated with the last date of x in
        fit and freq.
* `num_workers`: the number of workers to use in evaluate. It is only effective while input target value is dict of ndarray. If None, it defaults to
        num_ray_nodes in the created RayContext or 1 if there is no active RayContext.

#### **Incrementally fit the model with additional data**
Incrementally fit the model. Note that we only incrementally fit X_seq (TCN in global model). We haven't enable fit_incremental for input SparkXshards yet.
```python
model.fit_incremental(x_incr,
                      covariates_incr=None,
                      dti_incr=None
                     )
```
* `x_incr`: incremental data to be fitted. It should be of the same format as input x in fit, which is a dict of ndarray or SparkXShards of dict of ndarray.
Example: {'id': id_arr, 'y': incr_ndarray}, and incr_ndarray is of shape (n, T_incr), where
n is the number of target time series, T_incr is the number of time steps incremented. You
can choose not to input 'id' in x_incr, but if you do, the elements of id in x_incr should
be the same as id in x of fit.
* `covariates_incr`: covariates corresponding to x_incr. 2-D ndarray or None.
    The shape of ndarray should be (r, T_incr), where r is the number of covariates.
    Global covariates for all time series. If None, only default time coveriates will be
    used while use_time is True. If not, the time coveriates used is the stack of input
    covariates and default time coveriates.
* `dti_incr`: dti corresponding to the x_incr. DatetimeIndex or None.
    If None, use default fixed frequency DatetimeIndex generated with the last date of x in
    fit and freq.

#### **Save model**
You can save model after fit for future deployment.
```
model.save(path)
```
* `path`: (str) Path to target saved file.

#### **Load model**
You can load saved model with 
```
TCMFForecaster.load(path, 
                    distributed=False, 
                    minPartitions=None)
```
* `path`: (str) Path to target saved file.
* `distributed`: Whether the model is distributed trained with input of dict of SparkXshards.
* `minPartitions`: The minimum partitions for the XShards.

#### **Check whether model is distributed with input xshards**
You can check whether model is distributed by input xshards with `model.is_xshards_distributed()`.
