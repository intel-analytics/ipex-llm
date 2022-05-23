# AutoML Overview

Nano provides built-in AutoML support through hyperparameter optimization (referred to as  _Nano-HPO_ below).

By simply changing imports, you are able to search the model architecture (e.g. by specifying search spaces in layer/activation/function arguments when defining the model), or the training procedure (by specifying search spaces in `learning_rate` or `batch_size`). Use `Model/Trainer.search` to launch trials, and `Model/Trainer.search_summary` to review the search results.

Under the hood, The objects used in model and training are implicitly turned into searchable objects at creation, which allows search spaces to be specified in their init arguments. Nano-HPO collects the search spaces and pass them to the underlying HPO engine (i.e. Optuna), which generate hyperparameter suggestions accordingly. The instantiation and execution of the corresponding objects is delayed until the hyperparameter values are available in each trial.

### Install

If you did not install BigDL-Nano yet, follow the guide [here](../Overview/nano.md#2-install) to install BigDL-Nano according to the system and framework (i.e. tensorflow or pytorch) you use.

Then you need to install a few extra dependencies required for Nano HPO, using below commands.

```bash
pip install ConfigSpace
pip install optuna
```

### Global HPO Configuration

#### Enable/Disable HPO

For Tensorflow programs, you can use below command to enable HPO. This command add searchable layers, activations, functions, optimizers, etc into the `bigdl.nano.tf` module so that you can import them in your program. These objects allows search spaces to be specified in the init arguments.

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

- `search` runs the `n_trials` number of trials (meaning `n_trials` set of hyperparameter combinations are searched), and optimizes the `target_metric` in the specified `direction`. `search` does not return or save any tuned model. `search` also allows specifying the type of sampler and pruner. Refer to (API docs)[] for details.

- Use `model.search_summary` to retrieve the statistics of all the trials, and you can even do deeper analysis if needed. Examples of results analysis and visualization can be found [here]().


####  Search the Model Architecture

You can specify search spaces when defining the model, so that the model architecture can be searched using different hyperparamters

##### model defined using Sequential API

##### model defined using Functional API

##### model defined by Subclassing tf.keras.Model

#### Search the learning rate and batch size

---

### PyTorch

####  Search the Model Architecture

#### Search the learning rate and batch size

### Visualization


