Chronos Quick Tour
======================
Welcome to Chronos for building a fast, accurate and scalable time series analysis application🎉! Start with our quick tour to understand some critical concepts and how to use them to tackle your tasks.

.. grid:: 1 1 1 1

    .. grid-item-card::
        :text-align: center
        :shadow: none
        :class-header: sd-bg-light
        :class-footer: sd-bg-light
        :class-card: sd-mb-2

        **Data processing**
        ^^^
        Time series data processing includes imputing, deduplicating, resampling, scale/unscale, roll sampling, etc to process raw time series data(typically in a table) to a format that is understandable to the models. ``TSDataset`` and ``XShardsTSDataset`` are provided for an abstraction.
        +++
        .. button-ref:: TSDataset/XShardsTSDataset
            :color: primary
            :expand:
            :outline:

            Get Started

.. grid:: 1 1 3 3

    .. grid-item-card::
        :text-align: center
        :shadow: none
        :class-header: sd-bg-light
        :class-footer: sd-bg-light
        :class-card: sd-mb-2

        **Forecasting**
        ^^^
        Time series forecasting uses history data to predict future data. ``Forecaster`` and ``AutoTSEstimator`` are provided for built-in algorithms and distributed hyperparameter tunning.
        +++
        .. button-ref:: Forecaster
            :color: primary
            :expand:
            :outline:

            Get Started

    .. grid-item-card:: 
        :text-align: center
        :shadow: none
        :class-header: sd-bg-light
        :class-footer: sd-bg-light
        :class-card: sd-mb-2

        **Anomaly Detection**
        ^^^
        Time series anomaly detection finds the anomaly point in time series. ``Detector`` is provided for many built-in algorithms.
        +++
        .. button-ref:: Detector
            :color: primary
            :expand:
            :outline:

            Get Started

    .. grid-item-card:: 
        :text-align: center
        :shadow: none
        :class-header: sd-bg-light
        :class-footer: sd-bg-light
        :class-card: sd-mb-2

        **Simulation**
        ^^^
        Time series simulation generates synthetic time series data. ``Simulator`` is provided for many built-in algorithms.
        +++
        .. button-ref:: Simulator(experimental)
            :color: primary
            :expand:
            :outline:

            Get Started


TSDataset/XShardsTSDataset
---------------------

In Chronos, we provide a ``TSDataset`` (and a ``XShardsTSDataset`` to handle large data input in distributed fashion) abstraction to represent a time series dataset. It is responsible for preprocessing raw time series data(typically in a table) to a format that is understandable to the models. Many typical transformation, preprocessing and feature engineering method can be called cascadely on ``TSDataset`` or ``XShardsTSDataset``.

.. code-block:: python

    # !wget https://raw.githubusercontent.com/numenta/NAB/v1.0/data/realKnownCause/nyc_taxi.csv
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from bigdl.chronos.data import TSDataset

    df = pd.read_csv("nyc_taxi.csv", parse_dates=["timestamp"])
    tsdata = TSDataset.from_pandas(df,
                                dt_col="timestamp",
                                target_col="value")
    scaler = StandardScaler()
    tsdata.deduplicate()\
        .impute()\
        .gen_dt_feature()\
        .scale(scaler)\
        .roll(lookback=100, horizon=1)


.. grid:: 2
    :gutter: 1

    .. grid-item-card::

        .. button-ref:: ./data_processing_feature_engineering
            :color: primary
            :expand:
            :outline:

            Tutorial

    .. grid-item-card::

        .. button-ref:: ../../PythonAPI/Chronos/tsdataset
            :color: primary
            :expand:
            :outline:

            API Document

Forecaster
-----------------------
We have implemented quite a few algorithms among traditional statistics to deep learning for time series forecasting in ``bigdl.chronos.forecaster`` package. Users may train these forecasters on history time series and use them to predict future time series.

To import a specific forecaster, you may use {algorithm name} + "Forecaster", and call ``fit`` to train the forecaster and ``predict`` to predict future data.

.. code-block:: python

    from bigdl.chronos.forecaster import TCNForecaster  # TCN is algorithm name
    from bigdl.chronos.data.repo_dataset import get_public_dataset

    if __name__ == "__main__":
        # use nyc_taxi public dataset
        train_data, _, test_data = get_public_dataset("nyc_taxi")
        for data in [train_data, test_data]:
            # use 100 data point in history to predict 1 data point in future
            data.roll(lookback=100, horizon=1)

        # create a forecaster
        forecaster = TCNForecaster.from_tsdataset(train_data)

        # train the forecaster
        forecaster.fit(train_data)

        # predict with the trained forecaster
        pred = forecaster.predict(test_data)


AutoTSEstimator
---------------------------
For time series forecasting, we also provide an ``AutoTSEstimator`` for distributed hyperparameter tunning as an extention to ``Forecaster``. Users only need to create a ``AutoTSEstimator`` and call ``fit`` to train the estimator. A ``TSPipeline`` will be returned for users to predict future data.

.. code-block:: python

    from bigdl.orca.automl import hp
    from bigdl.chronos.data.repo_dataset import get_public_dataset
    from bigdl.chronos.autots import AutoTSEstimator
    from bigdl.orca import init_orca_context, stop_orca_context
    from sklearn.preprocessing import StandardScaler

    if __name__ == "__main__":
        # initial orca context
        init_orca_context(cluster_mode="local", cores=4, memory="8g", init_ray_on_spark=True)

        # load dataset
        tsdata_train, tsdata_val, tsdata_test = get_public_dataset(name='nyc_taxi')

        # dataset preprocessing
        stand = StandardScaler()
        for tsdata in [tsdata_train, tsdata_val, tsdata_test]:
            tsdata.gen_dt_feature().impute()\
                .scale(stand, fit=tsdata is tsdata_train)

        # AutoTSEstimator initalization
        autotsest = AutoTSEstimator(model="tcn",
                                    future_seq_len=10)

        # AutoTSEstimator fitting
        tsppl = autotsest.fit(data=tsdata_train,
                            validation_data=tsdata_val)

        # Prediction
        pred = tsppl.predict(tsdata_test)

        # stop orca context
        stop_orca_context()

.. grid:: 3
    :gutter: 1

    .. grid-item-card::

        .. button-ref:: ../QuickStart/chronos-tsdataset-forecaster-quickstart
            :color: primary
            :expand:
            :outline:

            Quick Start

    .. grid-item-card::

        .. button-ref:: ./forecasting
            :color: primary
            :expand:
            :outline:

            Tutorial

    .. grid-item-card::

        .. button-ref:: ../../PythonAPI/Chronos/forecasters
            :color: primary
            :expand:
            :outline:

            API Document

Detector
--------------------
We have implemented quite a few algorithms among traditional statistics to deep learning for time series anomaly detection in ``bigdl.chronos.detector.anomaly`` package.

To import a specific detector, you may use {algorithm name} + "Detector", and call ``fit`` to train the detector and ``anomaly_indexes`` to get anomaly data points' indexs.

.. code-block:: python

    from bigdl.chronos.detector.anomaly import DBScanDetector  # DBScan is algorithm name
    from bigdl.chronos.data.repo_dataset import get_public_dataset

    if __name__ == "__main__":
        # use nyc_taxi public dataset
        train_data = get_public_dataset("nyc_taxi", with_split=False)

        # create a detector
        detector = DBScanDetector()

        # fit a detector
        detector.fit(train_data.to_pandas()['value'].to_numpy())

        # find the anomaly points
        anomaly_indexes = detector.anomaly_indexes()

.. grid:: 3
    :gutter: 1

    .. grid-item-card::

        .. button-ref:: ../QuickStart/chronos-anomaly-detector
            :color: primary
            :expand:
            :outline:

            Quick Start

    .. grid-item-card::

        .. button-ref:: ./anomaly_detection
            :color: primary
            :expand:
            :outline:

            Tutorial

    .. grid-item-card::

        .. button-ref:: ../../PythonAPI/Chronos/anomaly_detectors
            :color: primary
            :expand:
            :outline:

            API Document

Simulator(experimental)
---------------------
Simulator is still under activate development with unstable API.

.. grid:: 2
    :gutter: 1

    .. grid-item-card::

        .. button-ref:: ./simulation
            :color: primary
            :expand:
            :outline:

            Tutorial

    .. grid-item-card::

        .. button-ref:: ../../PythonAPI/Chronos/simulator
            :color: primary
            :expand:
            :outline:

            API Document
