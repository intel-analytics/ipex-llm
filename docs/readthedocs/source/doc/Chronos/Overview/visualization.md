# AutoML Visualization

AutoML visualization provides two kinds of visualization. You may use them while fitting on auto models or AutoTS pipeline.
* During the searching process, the visualizations of each trail are shown and updated every 30 seconds. (Monitor view)
* After the searching process, a leaderboard of each trail's configs and metrics is shown. (Leaderboard view)

**Note**: AutoML visualization is based on tensorboard and tensorboardx. They should be installed properly before the training starts.

<span id="monitor_view">**Monitor view**</span>

Before training, start the tensorboard server through

```python
tensorboard --logdir=<logs_dir>/<name>
```

`logs_dir` is the log directory you set for your predictor(e.g. `AutoTSEstimator`, `AutoTCN`, etc.). `name ` is the name parameter you set for your predictor.

The data in SCALARS tag will be updated every 30 seconds for users to see the training progress.

![](../Image/automl_monitor.png)

After training, start the tensorboard server through

```python
tensorboard --logdir=<logs_dir>/<name>_leaderboard/
```

where `logs_dir` and `name` are the same as stated in [Monitor view](#monitor_view).

A dashboard of each trail's configs and metrics is shown in the SCALARS tag.

![](../Image/automl_scalars.png)

A leaderboard of each trail's configs and metrics is shown in the HPARAMS tag.

![](../Image/automl_hparams.png)

**Use visualization in Jupyter Notebook**

You can enable a tensorboard view in jupyter notebook by the following code.

```python
%load_ext tensorboard
# for scalar view
%tensorboard --logdir <logs_dir>/<name>/
# for leaderboard view
%tensorboard --logdir <logs_dir>/<name>_leaderboard/
```