# Tune a Forecasting Task Automatically

---

![](../../../../image/colab_logo_32px.png)[Run in Google Colab](https://colab.research.google.com/github/intel-analytics/BigDL/blob/branch-2.0/python/chronos/colab-notebook/chronos_autots_nyc_taxi.ipynb) &nbsp;![](../../../../image/GitHub-Mark-32px.png)[View source on GitHub](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/chronos/colab-notebook/chronos_autots_nyc_taxi.ipynb)

---

**In this guide we will demonstrate how to use _Chronos AutoTSEstimator_ and _Chronos TSPipeline_ to auto tune a time seires forecasting task and handle the whole model development process easily.**

### **Introduction**

Chronos provides `AutoTSEstimator` as a highly integrated solution for time series forecasting task with hyperparameter autotuning, auto feature selection and auto preprocessing. Users can prepare a `TSDataset`(recommended, used in this notebook) or their own data creator as input data. By constructing a `AutoTSEstimator` and calling `fit` on the data, a `TSPipeline` contains the best model and pre/post data processing will be returned for further development of deployment.

`AutoTSEstimator` only support LSTM, TCN, and Seq2seq built-in models and 3rd party models for now.

### **Step 0: Prepare Environment**

We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to prepare the environment. Please refer to the [install guide](../Overview/chronos.html#install) for more details.

```bash
conda create -n my_env python=3.7
conda activate my_env
pip install --pre --upgrade bigdl-chronos[all]
```

### **Step 1: Init Orca Context**
```python
if args.cluster_mode == "local":
    init_orca_context(cluster_mode="local", cores=4) # run in local mode
elif args.cluster_mode == "k8s":
    init_orca_context(cluster_mode="k8s", num_nodes=2, cores=2) # run on K8s cluster
elif args.cluster_mode == "yarn":
    init_orca_context(cluster_mode="yarn-client", num_nodes=2, cores=2) # run on Hadoop YARN cluster
```
This is the only place where you need to specify local or distributed mode. View [Orca Context](../../Orca/Overview/orca-context.md) for more details.

**Note:** You should `export HADOOP_CONF_DIR=/path/to/hadoop/conf/dir` when running on Hadoop YARN cluster. View [Hadoop User Guide](../../UserGuide/hadoop.md) for more details.

### **Step 2: Prepare a TSDataset**
Prepare a `TSDataset` and call necessary operations on it.
```python
from bigdl.chronos.data import TSDataset
from sklearn.preprocessing import StandardScaler

tsdata_train, tsdata_val, tsdata_test\
    = TSDataset.from_pandas(df, dt_col="timestamp", target_col="value", with_split=True, val_ratio=0.1, test_ratio=0.1)

standard_scaler = StandardScaler()
for tsdata in [tsdata_train, tsdata_val, tsdata_test]:
    tsdata.gen_dt_feature()\
          .impute(mode="last")\
          .scale(standard_scaler, fit=(tsdata is tsdata_train))
```
There is no need to call `.roll()` or `.to_torch_data_loader()` in this step, which is the largest difference between the usage of `AutoTSEstimator` and _Chronos Forecaster_. `AutoTSEstimator` will do that automatically and tune the parameters as well.

Please call `.gen_dt_feature()`(recommended), `.gen_rolling_feature()`, and `gen_global_feature()` to generate all candidate features to be selected by `AutoTSEstimator` as well as your input extra feature.

Detailed information please refer to [TSDataset API doc](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/tsdataset.html) and [Time series data basic concepts](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/data_processing_feature_engineering.html).

### **Step 3: Create an AutoTSEstimator**

```python
import bigdl.orca.automl.hp as hp
from bigdl.chronos.autots import AutoTSEstimator
auto_estimator = AutoTSEstimator(model='lstm', # the model name used for training
                                 search_space='normal', # a default hyper parameter search space
                                 past_seq_len=hp.randint(1, 10), # hp sampling function of past_seq_len for auto-tuning
) 
```
We prebuild three defualt search space for each build-in model, which you can use the by setting `search_space` to "minimal"，"normal", or "large" or define your own search space in a dictionary. The larger the search space, the better accuracy you will get and the more time will be cost.

`past_seq_len` can be set as a hp sample function, the proper range is highly related to your data. A range between 0.5 cycle and 3 cycle is reasonable.

Detailed information please refer to [AutoTSEstimator API doc](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/autotsestimator.html#autotsestimator) and basic concepts [here](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/forecasting.html#use-autots-pipeline).

### **Step 4: Fit with AutoTSEstimator**
```python
# fit with AutoTSEstimator for a returned TSPipeline
ts_pipeline = auto_estimator.fit(data=tsdata_train, # train dataset
                                 validation_data=tsdata_val, # validation dataset
                                 epochs=5) # number of epochs to train in each trial
```
Detailed information please refer to [AutoTSEstimator API doc](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/autotsestimator.html#autotsestimator).
### **Step 5: Further deployment with TSPipeline**
The `TSPipeline` will reply the same preprcessing and corresponding postprocessing operations on the test data. You may carry out predict, evaluate or save/load for further development.
```python
# predict with the best trial
y_pred = ts_pipeline.predict(tsdata_test)
```

```python
# evaluate the result pipeline
mse, smape = ts_pipeline.evaluate(tsdata_test, metrics=["mse", "smape"])
print("Evaluate: the mean square error is", mse)
print("Evaluate: the smape value is", smape)
```

```python
# save the pipeline
my_ppl_file_path = "/tmp/saved_pipeline"
ts_pipeline.save(my_ppl_file_path)
# restore the pipeline for further deployment
from bigdl.chronos.autots import TSPipeline
loaded_ppl = TSPipeline.load(my_ppl_file_path)
```
Detailed information please refer to [TSPipeline API doc](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/tsdataset.html).

### **Optional: Examine the leaderboard visualization**
To view the evaluation result of "not chosen" trails and find some insight or even possibly improve you search space for a new autotuning task. We provide a leaderboard through tensorboard.
```python
# show a tensorboard view
%load_ext tensorboard
%tensorboard --logdir /tmp/autots_estimator/autots_estimator_leaderboard/
```
Detailed information please refer to [Visualization](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/useful_functionalities.html#automl-visualization).
