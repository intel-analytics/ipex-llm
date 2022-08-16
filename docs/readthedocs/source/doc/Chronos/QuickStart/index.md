# Chronos Tutorial

</br>

```eval_rst
.. raw:: html

    <link rel="stylesheet" type="text/css" href="../../../_static/css/chronos_tutorial.css" />

    <div id="tutorial">
        <div>
            <div><input type="checkbox" class="checkboxes" name="choice" value="A">A</div>
            <div><input type="checkbox" class="checkboxes" name="choice" value="B">B</div>
            <div><input type="checkbox" class="checkboxes" name="choice" value="C">C</div>
        </div>
        </br>

        <details id="ChronosForecaster">
            <summary><a href="./chronos-tsdataset-forecaster-quickstart.html">Predict Number of Taxi Passengers with Chronos Forecaster</a></summary>
            <img src="../../../_images/colab_logo_32px.png"><a href="https://colab.research.google.com/github/intel-analytics/BigDL/blob/main/python/chronos/colab-notebook/chronos_nyc_taxi_tsdataset_forecaster.ipynb">Run in Google Colab</a>
            &nbsp;
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/colab-notebook/chronos_nyc_taxi_tsdataset_forecaster.ipynb">View source on GitHub</a>
            <p>In this guide we will demonstrate how to use _Chronos TSDataset_ and _Chronos Forecaster_ for time series processing and predict number of taxi passengers.</p>
        </details>
        <hr>

        <details id="TuneaForecasting">
            <summary><a href="./chronos-autotsest-quickstart.html">Tune a Forecasting Task Automatically</a></summary>
            <img src="../../../_images/colab_logo_32px.png"><a href="https://colab.research.google.com/github/intel-analytics/BigDL/blob/main/python/chronos/colab-notebook/chronos_autots_nyc_taxi.ipynb">Run in Google Colab</a>
            &nbsp;
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/colab-notebook/chronos_autots_nyc_taxi.ipynb">View source on GitHub</a>
            <p>In this guide we will demonstrate how to use _Chronos AutoTSEstimator_ and _Chronos TSPipeline_ to auto tune a time seires forecasting task and handle the whole model development process easily.</p>
        </details>
        <hr>

        <details id="DetectAnomaly">
            <summary><a href="./chronos-anomaly-detector.html">Detect Anomaly Point in Real Time Traffic Data</a></summary>
            <img src="../../../_images/colab_logo_32px.png"><a href="https://colab.research.google.com/github/intel-analytics/BigDL/blob/main/python/chronos/colab-notebook/chronos_minn_traffic_anomaly_detector.ipynb">Run in Google Colab</a>
            &nbsp;
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/colab-notebook/chronos_minn_traffic_anomaly_detector.ipynb">View source on GitHub</a>
            <p>In this guide we will demonstrate how to use _Chronos Anomaly Detector_ for real time traffic data from the Twin Cities Metro area in Minnesota anomaly detection.</p>
        </details>
        <hr>

        <details id="AutoTSEstimator">
            <summary><a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/network_traffic/network_traffic_autots_customized_model.ipynb">Tune a Customized Time Series Forecasting Model with AutoTSEstimator.</a></summary>
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/network_traffic/network_traffic_autots_customized_model.ipynb">View source on GitHub</a>
            <p>In this notebook, we demonstrate a reference use case where we use the network traffic KPI(s) in the past to predict traffic KPI(s) in the future. We demonstrate how to use _AutoTSEstimator_ to adjust the parameters of a customized model.</p>
        </details>
        <hr>

        <details id="AutoWIDE">
            <summary><a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/network_traffic/network_traffic_autots_forecasting.ipynb">Auto Tune the Prediction of Network Traffic at the Transit Link of WIDE</a></summary>
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/network_traffic/network_traffic_autots_forecasting.ipynb">View source on GitHub</a>
            <p>In this notebook, we demostrate a reference use case where we use the network traffic KPI(s) in the past to predict traffic KPI(s) in the future. We demostrate how to use _AutoTS_ in project <a href="https://github.com/intel-analytics/bigdl/tree/main/python/chronos/src/bigdl/chronos">Chronos</a> to do time series forecasting in an automated and distributed way.</p>
        </details>
        <hr>

        <details id="MultvarWIDE">
            <summary><a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/network_traffic/network_traffic_model_forecasting.ipynb">Multivariate Forecasting of Network Traffic at the Transit Link of WIDE</a></summary>
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/network_traffic/network_traffic_model_forecasting.ipynb">View source on GitHub</a>
            <p>In this notebook, we demonstrate a reference use case where we use the network traffic KPI(s) in the past to predict traffic KPI(s) in the future. We demostrate how to do univariate forecasting (predict only 1 series), and multivariate forecasting (predicts more than 1 series at the same time) using Project <a href="https://github.com/intel-analytics/bigdl/tree/main/python/chronos/src/bigdl/chronos">Chronos</a>.</p>
        </details>
        <hr>

        <details id="MultstepWIDE">
            <summary><a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/network_traffic/network_traffic_multivariate_multistep_tcnforecaster.ipynb">Multistep Forecasting of Network Traffic at the Transit Link of WIDE</a></summary>
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/network_traffic/network_traffic_multivariate_multistep_tcnforecaster.ipynb">View source on GitHub</a>
            <p>In this notebook, we demonstrate a reference use case where we use the network traffic KPI(s) in the past to predict traffic KPI(s) in the future. We demostrate how to do multivariate multistep forecasting using Project <a href="https://github.com/intel-analytics/bigdl/tree/main/python/chronos/src/bigdl/chronos">Chronos</a>.</p>
        </details>
        <hr>

        <details id="LSTMForecaster">
            <summary><a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/fsi/stock_prediction.ipynb">Stock Price Prediction with LSTMForecaster</a></summary>
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/fsi/stock_prediction.ipynb">View source on GitHub</a>
            <p>In this notebook, we demonstrate a reference use case where we use historical stock price data to predict the future price. The dataset we use is the daily stock price of S&P500 stocks during 2013-2018 (data source). We demostrate how to do univariate forecasting using the past 80% of the total days' MMM price to predict the future 20% days' daily price.</p>
            <p>Reference: <a href="https://github.com/jwkanggist/tf-keras-stock-pred">https://github.com/jwkanggist/tf-keras-stock-pred</a></p>
        </details>
        <hr>

        <details id="AutoProphet">
            <summary><a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/fsi/stock_prediction_prophet.ipynb">Stock Price Prediction with ProphetForecaster and AutoProphet</a></summary>
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/fsi/stock_prediction_prophet.ipynb">View source on GitHub</a>
            <p>In this notebook, we demonstrate a reference use case where we use historical stock price data to predict the future price using the ProphetForecaster and AutoProphet. The dataset we use is the daily stock price of S&P500 stocks during 2013-2018 <a href="https://www.kaggle.com/camnugent/sandp500/">data source</a>.</p>
            <p>Reference: <a href="https://facebook.github.io/prophet">https://facebook.github.io/prophet</a>, <a href="https://github.com/jwkanggist/tf-keras-stock-pred">https://github.com/jwkanggist/tf-keras-stock-pred</a></p>
        </details>
        <hr>

        <details id="Unsupervised">
            <summary><a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised.ipynb">Unsupervised Anomaly Detection for CPU Usage</a></summary>
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised.ipynb">View source on GitHub</a>
            <p>We demonstrates how to perform anomaly detection based on Chronos's built-in <a href="../../PythonAPI/Chronos/anomaly_detectors.html#dbscandetector">DBScanDetector</a>, <a href="../../PythonAPI/Chronos/anomaly_detectors.html#aedetector">AEDetector</a> and <a href="../../PythonAPI/Chronos/anomaly_detectors.html#thresholddetector">ThresholdDetector</a>.</p>
        </details>
        <hr>

        <details id="Anomaly Detection">
            <summary><a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised_forecast_based.ipynb">Anomaly Detection for CPU Usage Based on Forecasters</a></summary>
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised_forecast_based.ipynb">View source on GitHub</a>
            <p>We demonstrates how to leverage Chronos's built-in models ie. MTNet, to do time series forecasting. Then perform anomaly detection on predicted value with <a href="../../PythonAPI/Chronos/anomaly_detectors.html#thresholddetector">ThresholdDetector</a>.</p>
        </details>
        <hr>

        <details id="DeepARmodel">
            <summary><a href="https://github.com/intel-analytics/BigDL/tree/main/python/chronos/use-case/pytorch-forecasting/DeepAR">Help pytorch-forecasting improve the training speed of DeepAR model</a></summary>
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/tree/main/python/chronos/use-case/pytorch-forecasting/DeepAR">View source on GitHub</a>
            <p>Chronos can help a 3rd party time series lib to improve the performance (both training and inferencing) and accuracy. This use-case shows Chronos can easily help pytorch-forecasting speed up the training of DeepAR model.</p>
        </details>
        <hr>

        <details id="TFTmodel">
            <summary><a href="https://github.com/intel-analytics/BigDL/tree/main/python/chronos/use-case/pytorch-forecasting/TFT">Help pytorch-forecasting improve the training speed of TFT model</a></summary>
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/tree/main/python/chronos/use-case/pytorch-forecasting/TFT">View source on GitHub</a>
            <p>Chronos can help a 3rd party time series lib to improve the performance (both training and inferencing) and accuracy. This use-case shows Chronos can easily help pytorch-forecasting speed up the training of TFT model.</p>
        </details>
        <hr>

        <details id="hyperparameter">
            <summary><a href="https://github.com/intel-analytics/BigDL/tree/main/python/chronos/example/hpo/muti_objective_hpo_with_builtin_latency_tutorial.ipynb">Tune a Time Series Forecasting Model with multi-objective hyperparameter optimization.</a></summary>
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/tree/main/python/chronos/example/hpo/muti_objective_hpo_with_builtin_latency_tutorial.ipynb">View source on GitHub</a>
            <p>In this notebook, we demostrate how to use _multi-objective hyperparameter optimization with built-in latency metric_ in project <a href="https://github.com/intel-analytics/bigdl/tree/main/python/chronos/src/bigdl/chronos">Chronos</a> to do time series forecasting and achieve good tradeoff between performance and latency.</p>
        </details>
        <hr>

    </div>

    <script src="../../../_static/js/chronos_tutorial.js"></script> 
```