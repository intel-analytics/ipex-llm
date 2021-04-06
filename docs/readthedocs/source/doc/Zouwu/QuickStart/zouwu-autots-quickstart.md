# Use AutoML for Time-Series Forecasting

---

![](../../../../image/colab_logo_32px.png)[Run in Google Colab](https://colab.research.google.com/github/intel-analytics/analytics-zoo/blob/master/docs/docs/colab-notebook/zouwu/zouwu_autots_nyc_taxi.ipynb) &nbsp;![](../../../../image/GitHub-Mark-32px.png)[View source on GitHub](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/colab-notebook/zouwu/zouwu_autots_nyc_taxi.ipynb)

---

**In this guide we will demonstrate how to use _Zouwu AutoTS_ for automated time seires forecasting in 4 simple steps.**

### **Step 0: Prepare Environment**

We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to prepare the environment. Please refer to the [install guide](../../UserGuide/python.md) for more details.

```bash
conda create -n zoo python=3.7 # "zoo" is conda environment name, you can use any name you like.
conda activate zoo
pip install analytics-zoo[automl] # install either version 0.9 or latest nightly build
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

### **Step 2: Create an AutoTSTrainer**

You can then Create an `AutoTSTrainer`.

```python
from zoo.zouwu.autots.forecast import AutoTSTrainer

trainer = AutoTSTrainer(dt_col="timestamp",  
                        target_col="value",  
                        horizon=1,           
                        extra_features_col=None
                        )
```
### **Step 3: Fit with AutoTSTrainer**

You can then train on the input data using `AutoTSTrainer.fit` with a recipe to specify search space.

```python
from zoo.zouwu.config.recipe import LSTMGridRandomRecipe

ts_pipeline = trainer.fit(train_df, val_df,
                          recipe=LSTMGridRandomRecipe(
                              num_rand_samples=1,
                              epochs=1,
                              look_back=6,
                              batch_size=[64]),
                          metric="mse")
```

### **Step 4: Further deployment with TSPipeline**

You can use the result `ts_pipeline` for prediction, evaluation or (incremental) fitting.
```python
# predict with the best trial
pred_df = ts_pipeline.predict(test_df)

# evaluate the result pipeline
mse, smape = ts_pipeline.evaluate(test_df, metrics=["mse", "smape"])
print("Evaluate: the mean square error is", mse)
print("Evaluate: the smape value is", smape)
```

You can also save and restore the pipeline for further deployment.
```python
# save the pipeline
my_ppl_file_path = ts_pipeline.save("/tmp/saved_pipeline/nyc_taxi.ppl")

# restore the pipeline for further deployment
from zoo.zouwu.autots.forecast import TSPipeline
loaded_ppl = TSPipeline.load(my_ppl_file_path)
```
That's it, the same code can run seamlessly in your local laptop and the distribute K8s or Hadoop cluster.

**Note:** An `OrcaContext` is only necessary for `AutoTSTrainer` and is not needed if you only use `TSPipeline`.

