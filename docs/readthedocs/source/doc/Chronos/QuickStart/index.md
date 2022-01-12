# Chronos Tutorial

- [**Predict Number of Taxi Passengers with Chronos Forecaster**](./chronos-tsdataset-forecaster-quickstart.html)

    > ![](../../../../image/colab_logo_32px.png)[Run in Google Colab](https://colab.research.google.com/github/intel-analytics/BigDL/blob/branch-2.0/python/chronos/colab-notebook/chronos_nyc_taxi_tsdataset_forecaster.ipynb) &nbsp;![](../../../../image/GitHub-Mark-32px.png)[View source on GitHub](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/chronos/colab-notebook/chronos_nyc_taxi_tsdataset_forecaster.ipynb)

    In this guide we will demonstrate how to use _Chronos TSDataset_ and _Chronos Forecaster_ for time series processing and predict number of taxi passengers.

---------------------------

- [**Tune a Forecasting Task Automatically**](./chronos-autotsest-quickstart.html)

    > ![](../../../../image/colab_logo_32px.png)[Run in Google Colab](https://colab.research.google.com/github/intel-analytics/BigDL/blob/branch-2.0/python/chronos/colab-notebook/chronos_autots_nyc_taxi.ipynb) &nbsp;![](../../../../image/GitHub-Mark-32px.png)[View source on GitHub](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/chronos/colab-notebook/chronos_autots_nyc_taxi.ipynb)

    In this guide we will demonstrate how to use _Chronos AutoTSEstimator_ and _Chronos TSPipeline_ to auto tune a time seires forecasting task and handle the whole model development process easily.

---------------------------

- [**Detect Anomaly Point in Real Time Traffic Data**](./chronos-anomaly-detector.html)

    > ![](../../../../image/colab_logo_32px.png)[Run in Google Colab](https://colab.research.google.com/github/intel-analytics/BigDL/blob/branch-2.0/python/chronos/colab-notebook/chronos_minn_traffic_anomaly_detector.ipynb) &nbsp;![](../../../../image/GitHub-Mark-32px.png)[View source on GitHub](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/chronos/colab-notebook/chronos_minn_traffic_anomaly_detector.ipynb)

    In this guide we will demonstrate how to use _Chronos Anomaly Detector_ for real time traffic data from the Twin Cities Metro area in Minnesota anomaly detection.

---------------------------

- [**AutoTSEstimator with customized model**](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/forecasting.html)

    > ![](../../../../image/GitHub-Mark-32px.png)[View source on GitHub](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/chronos/use-case/network_traffic/network_traffic_autots_customized_model.ipynb)

    In this notebook, we demonstrate a reference use case where we use the network traffic KPI(s) in the past to predict traffic KPI(s) in the future. We demonstrate how to use _AutoTSEstimator_ to adjust the parameters of a customized model.

---------------------------

- [**Network Traffic Forecasting with AutoTSEstimator**](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/forecasting.html#)

    > ![](../../../../image/GitHub-Mark-32px.png)[View source on GitHub](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/chronos/use-case/network_traffic/network_traffic_autots_forecasting.ipynb)

    In this notebook, We demostrate how to use _AutoTS_ in project [Chronos][4] to do time series forecasting in an automated and distributed way.

---------------------------

- [**Univariate and Multivariate Forecasting**](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/forecasting.html#time-series-forecasting-concepts)

    > ![](../../../../image/GitHub-Mark-32px.png)[View source on GitHub](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/chronos/use-case/network_traffic/network_traffic_model_forecasting.ipynb)

    In this notebook, We demostrate how to do univariate forecasting (predict only 1 series), and multivariate forecasting (predicts more than 1 series at the same time) using Project Chronos.

---------------------------

- [**Multivariate Multistep Forecasting)**](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/forecasting.html#time-series-forecasting-concepts)

    > ![](../../../../image/GitHub-Mark-32px.png)[View source on GitHub](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/chronos/use-case/network_traffic/network_traffic_multivariate_multistep_tcnforecaster.ipynb)

    In this notebook, We demostrate how to do multivariate multistep forecasting using Project Chronos.

---------------------------

- [**Stock Price Prediction**](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/forecasting.html)

    > ![](../../../../image/GitHub-Mark-32px.png)[View source on GitHub](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/chronos/use-case/fsi/stock_prediction.ipynb)

    In this notebook, we demonstrate a reference use case where we use historical stock price data to predict the future price. The dataset we use is the daily stock price of S&P500 stocks during 2013-2018 (data source). We demostrate how to do univariate forecasting using the past 80% of the total days' MMM price to predict the future 20% days' daily price. Reference: *<https://github.com/jwkanggist/tf-keras-stock-pred>*

---------------------------

- [**Stock Price Prediction with ProphetForecaster and AutoProphet**](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/forecasting.html)

    > ![](../../../../image/GitHub-Mark-32px.png)[View source on GitHub](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/chronos/use-case/fsi/stock_prediction_prophet.ipynb)

    In this notebook, we demonstrate a reference use case where we use historical stock price data to predict the future price using the ProphetForecaster and AutoProphet. The dataset we use is the daily stock price of S&P500 stocks during 2013-2018 [data source](https://www.kaggle.com/camnugent/sandp500/).

    Reference: *<https://facebook.github.io/prophet>*, *<https://github.com/jwkanggist/tf-keras-stock-pred>*

---------------------------

- [**Unsupervised Anomaly Detection**]https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/anomaly_detection.html)

    > ![](../../../../image/GitHub-Mark-32px.png)[View source on GitHub](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised.ipynb)

    We demonstrates how to perform anomaly detection based on Chronos's built-in [DBScanDetector][DBScan], [AEDetector][AE] and [ThresholdDetector][Threshold].

---------------------------

- [**Unsupervised Anomaly Detection based on Forecasts**](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/anomaly_detection.html)

    > ![](../../../../image/GitHub-Mark-32px.png)[View source on GitHub](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised_forecast_based.ipynb)

    We demonstrates how to leverage Chronos's built-in models ie. MTNet, to do time series forecasting. Then perform anomaly detection on predicted value with [ThresholdDetector][Threshold].


[DBScan]: <https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/anomaly_detectors.html#dbscandetector>
[AE]: <https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/anomaly_detectors.html#aedetector>
[Threshold]: <https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/anomaly_detectors.html#thresholddetector>
[4]: <https://github.com/intel-analytics/bigdl/tree/branch-2.0/python/chronos/src/bigdl/chronos>

