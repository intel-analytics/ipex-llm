
In this guide, we will show you how to use Chronos TCMFForecaster for high dimension time series forecasting.

Refer to [TCMFForecaster example](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/chronos/examples/tcmf/run_electricity.py) for demonstration of using TCMFForecaster for distributed training and inference. 

Refer to [TCMFForecaster API Guide](../API/TCMFForecaster.md) for more details of AutoTS APIs.

---
### **Step 0: Prepare environment**
a. We recommend conda to set up your environment. Note that conda environment is required to run onq
yarn, but not strictly necessary for running on local. 
```
conda create -n zoo python=3.7
conda activate zoo
```

b. If you want to enable TCMFForecaster distributed training, it requires pre-install pytorch and horovod. You can follow the [horovod document](https://github.com/horovod/horovod/blob/master/docs/install.rst) to install the horovod and pytorch with Gloo support.
And here are the commands that work on for us on ubuntu 16.04. The exact steps may vary from different machines.

```
conda install -y pytorch==1.4.0 torchvision==0.5.0 cpuonly -c pytorch
conda install -y cmake==3.16.0 -c conda-forge
conda install cxx-compiler==1.0 -c conda-forge
conda install openmpi
HOROVOD_WITH_PYTORCH=1; HOROVOD_WITH_GLOO=1; pip install --no-cache-dir horovod==0.19.1
pip install analytics_zoo-${VERSION}-${TIMESTAMP}-py2.py3-none-${OS}_x86_64.whl[ray]
```

If you don't need distributed training. You only need to install pytorch in your environment.

```
pip install torch==1.4.0 torchvision==0.5.0
```

c. Download and install nightly build analytics zoo whl by following instructions ([here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/#install-the-latest-nightly-build-wheels-for-pip)).

```
pip install analytics_zoo-${VERSION}-${TIMESTAMP}-py2.py3-none-${OS}_x86_64.whl[ray]
```

d. Install other packages
```
pip install scikit-learn==0.22
pip install pandas==1.0
pip install requests
```

### **Step 1: Init Orca Context**
You need to init an orca context with `init_ray_on_spark=True` before distributed training, and stop it after training is completed. Note orca context is not needed if you don't want to enable distributed training.
```python
from zoo.orca import init_orca_context, stop_orca_context

# run in local mode
init_orca_context(cluster_mode="local", cores=4, memory='2g', num_nodes=1, init_ray_on_spark=True)

# run in yarn client mode
init_orca_context(cluster_mode="yarn-client", 
                  num_nodes=2, cores=2, 
                  driver_memory="6g", driver_cores=4, 
                  conda_name='zoo', 
                  extra_memory_for_ray="10g", 
                  object_store_memory='5g')
```
* Reference: [Orca Context](https://analytics-zoo.github.io/master/#Orca/context/)

### **Step 2: Create a TCMFForecaster**

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
### **Step 3: Use TCMFForecaster**

#### **Fit with TCMFForecaster**

```
model.fit(
        x,
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
        num_workers=num_workers_for_fit)
```

#### **Evaluate with TCMFForecaster**
You can either directly call `model.evaluate` as
```
model.evaluate(target_value,
               metric=['mae'],
               target_covariates=None,
               target_dti=None,
               num_workers=num_workers_for_predict,
               )
```

Or you could predict first and then evaluate with metric name.

```
yhat = model.predict(horizon,
                     future_covariates=None,
                     future_dti=None,
                     num_workers=num_workers_for_predict)

from zoo.orca.automl.metrics import Evaluator
evaluate_mse = Evaluator.evaluate("mse", target_data, yhat)
```

#### **Incremental fit TCMFForecaster**
```python
model.fit_incremental(x_incr, covariates_incr=None, dti_incr=None)
```

#### **Save and Load**

```python
model.save(dirname)
loaded_model = TCMFForecaster.load(dirname)
```
