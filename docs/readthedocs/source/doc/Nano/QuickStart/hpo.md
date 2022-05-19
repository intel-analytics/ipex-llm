# AutoML Overview

Nano provides built-in AutoML support through a hyperparameter optimization module (Nano HPO).

By simply changing imports, you are able to specify search spaces directly in layer/activation/function parameters when defining the model to search the model architecture, or in `learning_rate` or `batch_size` to control the training process. To trigger the search, just call `model.search`(tensorflow)  or `Trainer.search`(pytorch)  to run a number of trials, or `model.search_summary`(tensorflow) or `Trainer.search_summary`(pytorch) to review the search results.

Under the hood, Nano HPO collects the search spaces and pass them to the underlying HPO engine (i.e. Optuna), which generate hyperparameter combinations accordingly for each trial. The objects and parameters used in model and training are automatically wrapped into searchable objects at creation, and their instantiations are delayed until the actual parameter values are available in each trial.

### Install

If you did not install BigDL-Nano yet, follow the guide [here](../Overview/nano.md#2-install) to install BigDL-Nano according to the system and framework (i.e. tensorflow or pytorch) you use.

Then you need to install a few extra dependencies required for Nano HPO, using below commands.

```bash
pip install ConfigSpace
pip install optuna
```

### Global HPO Configuration

#### Enable/Disable HPO

For Tensorflow programs, you can use below command to enable HPO. This command add searchable layers, activations, functions, optimizers, etc into the `bigdl.nano.tf` module which you can import in your program. Such searchable objects allows you to specify search spaces in the parameters.

```python
import bigdl.nano.automl as nano_automl
nano_automl.hpo_config.enable_hpo_tf()
```
To disable, use `disable_hpo_tf`
```python
import bigdl.nano.automl as nano_automl
nano_automl.hpo_config.disable_hpo_tf()
```
---

Similarly, for PyTorch programs, you can use below command to enable HPO and disable HPO.
```python
import bigdl.nano.automl as nano_automl
nano_automl.hpo_config.enable_hpo_pytorch()
```
```python
import bigdl.nano.automl as nano_automl
nano_automl.hpo_config.disable_hpo_pytorch()
```

### Tensorflow


#### APIs to Run search and get results

The major API to drive the hyperparameter search is `search` and `search_summary`, which is called after `model.compile` and before `model.fit`. The usage is like below.

```python
model = ... # define the model
model.compile(...)
model.search(n_trials=2,
            target_metric='accuracy',
            direction="maximize",
            x=x_train,
            y=y_train,
            batch_size=32,
            epochs=2,
            validation_split=0.2)
study = model.search_summary()
model.fit(...)
```

- `search` runs the `n_trials` number of trials (which means there're `n_trials` of hyperparameter combinations are tried), and optimizes the `target_metric` in the specified `direction`. `search` does not return or save any tuned model. `search` also allows specifying the type of sampler and pruner. Refer to (API docs)[] for details.

- Use `model.search_summary` to retrieve the statistics of all the trials, and you can even do deeper analysis if needed. Examples of results analysis and visualization can be found [here]().


####  Search the Model Architecture


##### model defined using Sequential API

##### model defined using Functional API

##### model defined by Subclassing tf.keras.Model

#### Search the learning rate and batch size

---

### PyTorch

####  Search the Model Architecture

#### Search the learning rate and batch size

### Visualization


