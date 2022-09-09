Chronos How-to Guides
=========================
How-to guides are bite-sized, executable examples where users could check when meeting with some specific topic during the usage.

Install
-------------------------

* `Install Chronos on Windows <windows_guide.html>`__
* `Use Chronos in container(docker) <docker_guide_single_node.html>`__
=======
* `Create a forecaster <how-to-create-forecaster.html>`__

    In this guidance, we demonstrate **how to create a Forecaster**. Including two ways of creating a forecaster and an explanation of some important parameters.

* `Train forecaster on single node <how_to_train_forecaster_on_one_node.html>`__

    In this guidance, **we demonstrate how to train forecasters on one node**. In the training process, forecaster will learn the pattern (like the period, scale...) in history data. Although Chronos supports training on a cluster, it's highly recommeneded to try one node first before allocating a cluster to make life easier.

* `Tune forecaster on single node <how_to_tune_forecaster_model.html>`__

    In this guidance, we demonstrate **how to tune forecaster on single node**. In tuning process, forecaster will find the best hyperparameter combination among user-defined search space, which is a common process if users pursue a forecaster with higher accuracy.

* `Speed up inference of forecaster through ONNXRuntime <how_to_speedup_inference_of_forecaster_through_ONNXRuntime.html>`__

    In this guidance, **we demonstrate how to speed up inference of forecaster through ONNXRuntime**. In inferencing process, Chronos supports ONNXRuntime to accelerate inferencing which is helpful to users.

* `Speed up inference of forecaster through OpenVINO <how_to_speedup_inference_of_forecaster_through_OpenVINO.html>`__

    In this guidance, **we demonstrate how to speed up inference of forecaster through OpenVINO**. In inferencing process, Chronos supports OpenVINO to accelerate inferencing which is helpful to users.

* `Evaluate a forecaster <how_to_evaluate_using_forecaster.html>`__

    In this guidance, **we demonstrate how to evaluate a forecaster** in detail. The evaluate result is calculated by actual value and predicted value.



.. toctree::
    :maxdepth: 1
    :hidden:

    windows_guide
    docker_guide_single_node

Forecasting
-------------------------
* `Create a forecaster <how_to_create_forecaster.html>`__
* `Train forecaster on single node <how_to_train_forecaster_on_one_node.html>`__
* `Tune forecaster on single node <how_to_tune_forecaster_model.html>`__
* `Speed up inference of forecaster through ONNXRuntime <how_to_speedup_inference_of_forecaster_through_ONNXRuntime.html>`__
* `Speed up inference of forecaster through OpenVINO <how_to_speedup_inference_of_forecaster_through_OpenVINO.html>`__

.. toctree::
    :maxdepth: 1
    :hidden:

    how_to_create_forecaster
    how_to_train_forecaster_on_one_node
    how_to_tune_forecaster_model
    how_to_speedup_inference_of_forecaster_through_ONNXRuntime
    how_to_speedup_inference_of_forecaster_through_OpenVINO
    how_to_evaluate_using_forecaster

