Chronos How-to Guides
=========================
How-to guides are bite-sized, executable examples where users could check when meeting with some specific topic during the usage.

Forecasting
-------------------------
* `Train forecaster on single node <how_to_train_forecaster_on_one_node.html>`__

    In this guidance, **we demonstrate how to train forecasters on one node**. In the training process, forecaster will learn the pattern (like the period, scale...) in history data. Although Chronos supports training on a cluster, it's highly recommeneded to try one node first before allocating a cluster to make life easier.

* `Tune forecaster on single node <how_to_tune_forecaster_model.html>`__

    In this guidance, we demonstrate **how to tune forecaster on single node**. In tuning process, forecaster will find the best hyperparameter combination among user-defined search space, which is a common process if users pursue a forecaster with higher accuracy.


.. toctree::
    :maxdepth: 1
    :hidden:

    how_to_train_forecaster_on_one_node
    how_to_tune_forecaster_model

