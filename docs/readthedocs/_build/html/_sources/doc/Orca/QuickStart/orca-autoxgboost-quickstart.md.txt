# Use AutoXGBoost to auto-tune XGBoost parameters

---

![](../../../../image/colab_logo_32px.png)[Run in Google Colab](https://colab.research.google.com/github/intel-analytics/BigDL/blob/branch-2.0/python/orca/colab-notebook/quickstart/autoxgboost_regressor_sklearn_boston.ipynb) &nbsp;![](../../../../image/GitHub-Mark-32px.png)[View source on GitHub](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/orca/colab-notebook/quickstart/autoxgboost_regressor_sklearn_boston.ipynb)

---

**In this guide we will describe how to use Orca AutoXGBoost for automated xgboost tuning**

Orca AutoXGBoost enables distributed automated hyper-parameter tuning for XGBoost, which includes `AutoXGBRegressor` and `AutoXGBClassifier` for sklearn`XGBRegressor` and `XGBClassifier` respectively. See more about [xgboost scikit-learn API](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn).
### **Step 0: Prepare Environment**

[Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) is needed to prepare the Python environment for running this example. Please refer to the [install guide](https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/distributed-tuning.html#install) for more details.


### **Step 1: Init Orca Context**
```python
from bigdl.orca import init_orca_context, stop_orca_context

if cluster_mode == "local":
    init_orca_context(cores=6, memory="2g", init_ray_on_spark=True) # run in local mode
elif cluster_mode == "k8s":
    init_orca_context(cluster_mode="k8s", num_nodes=2, cores=4, init_ray_on_spark=True) # run on K8s cluster
elif cluster_mode == "yarn":
    init_orca_context(
      cluster_mode="yarn-client", cores=4, num_nodes=2, memory="2g", init_ray_on_spark=True, 
      driver_memory="10g", driver_cores=1) # run on Hadoop YARN cluster
```

This is the only place where you need to specify local or distributed mode. View [Orca Context](./../Overview/orca-context.md) for more details.

**Note:** You should `export HADOOP_CONF_DIR=/path/to/hadoop/conf/dir` when running on Hadoop YARN cluster. View [Hadoop User Guide](./../../UserGuide/hadoop.md) for more details.

### **Step 2: Define Search space**

You should define a dictionary as your hyper-parameter search space.

The keys are hyper-parameter names you want to search for `XGBRegressor`, and you can specify how you want to sample each hyper-parameter in the values of the search space. See [automl.hp](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/AutoML/automl.html#orca-automl-hp) for more details.

```python
from bigdl.orca.automl import hp

search_space = {
    "n_estimators": hp.grid_search([50, 100, 200]),
    "max_depth": hp.choice([2, 4, 6]),
}
```

### **Step 3: Automatically fit and search with Orca AutoXGBoost**

First create an `AutoXGBRegressor`.

```python
from bigdl.orca.automl.xgboost import AutoXGBRegressor

auto_xgb_reg = AutoXGBRegressor(cpus_per_trial=2, 
                                name="auto_xgb_classifier",
                                min_child_weight=3,
                                random_state=2)
```

Next, use the `AutoXGBRegressor` to fit and search for the best hyper-parameter set.

```python
auto_xgb_reg.fit(data=(X_train, y_train),
                 validation_data=(X_test, y_test),
                 search_space=search_space,
                 n_sampling=2,
                 metric="rmse")
```

### **Step 4: Get best model and hyper parameters**

You can get the best learned model and the best hyper-parameter set for further deployment. The best model is an sklearn `XGBRegressor` instance.

```python
best_model = auto_xgb_reg.get_best_model()
best_config = auto_xgb_reg.get_best_config()
```

**Note:** You should call `stop_orca_context()` when your application finishes.
