# Chronos Examples

```eval_rst
.. raw:: html

    <link rel="stylesheet" type="text/css" href="../../../_static/css/chronos_tutorial.css" />

    <div id="tutorial">
        <h3 style="text-align:left">Filter:</h3>
        <p>Please <span style="font-weight:bold;">check</span> the checkboxes or <span style="font-weight:bold;">click</span> tag buttons to show the related examples. Reclick or uncheck will hide corresponding examples. If nothing is checked or clicked, all the examples will be displayed. </p>
        <div class="border">
            <div class="choiceline">
                <div class="choicebox"><input type="checkbox" class="checkboxes" name="choice" value="forecast" id="forecast">forecast </div>
                <div class="choicebox"><input type="checkbox" class="checkboxes" name="choice" value="anomaly_detection" id="anomaly_detection">anomaly detection</div>
                <div class="choicebox"><input type="checkbox" class="checkboxes" name="choice" value="simulation" id="simulation">simulation</div>
                <div class="choicebox"><input type="checkbox" class="checkboxes" name="choice" value="hyperparameter_tuning" id="hyperparameter_tuning">AutoML</div>
            </div>
            <div class="choiceline">
                <div class="choicebox"><input type="checkbox" class="checkboxes" name="choice" value="onnxruntime" id="onnxruntime">onnxruntime </div>
                <div class="choicebox"><input type="checkbox" class="checkboxes" name="choice" value="quantization" id="quantization">quantization</div>
                <div class="choicebox"><input type="checkbox" class="checkboxes" name="choice" value="distributed" id="distributed">distributed</div>
                <div class="choicebox"><input type="checkbox" class="checkboxes" name="choice" value="customized_model" id="customized_model">customized model</div>
            </div>
            <div class="hiddenline">
                <div class="choicebox"><input type="checkbox" class="forecasters" name="forecasters" value="TCNForecaster" id="TCNForecaster">TCNForecaster</div>
                <div class="choicebox"><input type="checkbox" class="forecasters" name="forecasters" value="AutoTSEstimator" id="AutoTSEstimator">AutoTSEstimator</div>
                <div class="choicebox"><input type="checkbox" class="forecasters" name="forecasters" value="DBScanDetector" id="DBScanDetector">DBScanDetector</div>
                <div class="choicebox"><input type="checkbox" class="forecasters" name="forecasters" value="LSTMForecaster" id="LSTMForecaster">LSTMForecaster</div>
                <div class="choicebox"><input type="checkbox" class="forecasters" name="forecasters" value="AutoProphet" id="AutoProphet">AutoProphet</div>
                <div class="choicebox"><input type="checkbox" class="forecasters" name="forecasters" value="MTNetForecaster" id="MTNetForecaster">MTNetForecaster</div>
                <div class="choicebox"><input type="checkbox" class="forecasters" name="forecasters" value="DeepAR" id="DeepAR">DeepAR</div>
                <div class="choicebox"><input type="checkbox" class="forecasters" name="forecasters" value="AutoLSTM" id="AutoLSTM">AutoLSTM</div>
                <div class="choicebox"><input type="checkbox" class="forecasters" name="forecasters" value="Seq2SeqForecaster" id="Seq2SeqForecaster">Seq2SeqForecaster</div>
                <div class="choicebox"><input type="checkbox" class="forecasters" name="forecasters" value="DPGANSimulator" id="DPGANSimulator">DPGANSimulator</div>
                <div class="choicebox"><input type="checkbox" class="forecasters" name="forecasters" value="TCMFForecaster" id="TCMFForecaster">TCMFForecaster</div>
                <div class="choicebox"><input type="checkbox" class="forecasters" name="forecasters" value="TFT_model" id="TFT_model">TFT_model</div>
            </div>
        </div>
        </br>
        <div class="showingForecaster">Currently showing forcaster:&nbsp;<i>All Forecasters</i>&nbsp;&nbsp;(<span style="font-weight:bold;">Reclick</span> the tag of these forecasters to undo.)</div>
        </br>

        <details id="ChronosForecaster">
            <summary>
                <a href="./chronos-tsdataset-forecaster-quickstart.html">Predict Number of Taxi Passengers with Chronos Forecaster</a>
                <p>Tag: 
                    <button value="forecast">forecast</button>&nbsp;
                    <button value="TCNForecaster" class="roundbutton">TCNForecaster</button>
                </p>
            </summary>
            <img src="../../../_images/colab_logo_32px.png"><a href="https://colab.research.google.com/github/intel-analytics/BigDL/blob/main/python/chronos/colab-notebook/chronos_nyc_taxi_tsdataset_forecaster.ipynb">Run in Google Colab</a>
            &nbsp;
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/colab-notebook/chronos_nyc_taxi_tsdataset_forecaster.ipynb">View source on GitHub</a>
            <p>In this guide we will demonstrate how to use <span>Chronos TSDataset</span> and <span>Chronos Forecaster</span> for time series processing and predict number of taxi passengers.</p>
        </details>
        <hr>

        <details id="TuneaForecasting">
            <summary>
                <a href="./chronos-autotsest-quickstart.html">Tune a Forecasting Task Automatically</a>
                <p>Tag: 
                    <button value="forecast">forecast</button>&nbsp;
                    <button value="hyperparameter_tuning">AutoML</button>&nbsp;
                    <button value="AutoTSEstimator" class="roundbutton">AutoTSEstimator</button>
                </p>
            </summary>
            <img src="../../../_images/colab_logo_32px.png"><a href="https://colab.research.google.com/github/intel-analytics/BigDL/blob/main/python/chronos/colab-notebook/chronos_autots_nyc_taxi.ipynb">Run in Google Colab</a>
            &nbsp;
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/colab-notebook/chronos_autots_nyc_taxi.ipynb">View source on GitHub</a>
            <p>In this guide we will demonstrate how to use <span>Chronos AutoTSEstimator</span> and <span>Chronos TSPipeline</span> to auto tune a time seires forecasting task and handle the whole model development process easily.</p>
        </details>
        <hr>

        <details id="DetectAnomaly">
            <summary>
                <a href="./chronos-anomaly-detector.html">Detect Anomaly Point in Real Time Traffic Data</a>
                <p>Tag: 
                    <button value="anomaly_detection">anomaly detection</button>&nbsp;
                    <button value="DBScanDetector" class="roundbutton">DBScanDetector</button>
                </p>
            </summary>
            <img src="../../../_images/colab_logo_32px.png"><a href="https://colab.research.google.com/github/intel-analytics/BigDL/blob/main/python/chronos/colab-notebook/chronos_minn_traffic_anomaly_detector.ipynb">Run in Google Colab</a>
            &nbsp;
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/colab-notebook/chronos_minn_traffic_anomaly_detector.ipynb">View source on GitHub</a>
            <p>In this guide we will demonstrate how to use <span>Chronos Anomaly Detector</span> for real time traffic data from the Twin Cities Metro area in Minnesota anomaly detection.</p>
        </details>
        <hr>

        <details id="AutoTS">
            <summary>
                <a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/network_traffic/network_traffic_autots_customized_model.ipynb">Tune a Customized Time Series Forecasting Model with AutoTSEstimator.</a>
                <p>Tag: 
                    <button value="forecast">forecast</button>&nbsp;
                    <button value="hyperparameter_tuning">AutoML</button>&nbsp;
                    <button value="customized_model">customized model</button>&nbsp;
                    <button value="AutoTSEstimator" class="roundbutton">AutoTSEstimator</button>
                </p>
            </summary>
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/network_traffic/network_traffic_autots_customized_model.ipynb">View source on GitHub</a>
            <p>In this notebook, we demonstrate a reference use case where we use the network traffic KPI(s) in the past to predict traffic KPI(s) in the future. We demonstrate how to use <span>AutoTSEstimator</span> to adjust the parameters of a customized model.</p>
        </details>
        <hr>

        <details id="AutoWIDE">
            <summary>
                <a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/network_traffic/network_traffic_autots_forecasting.ipynb">Auto Tune the Prediction of Network Traffic at the Transit Link of WIDE</a>
                <p>Tag: 
                    <button value="forecast">forecast</button>&nbsp;
                    <button value="hyperparameter_tuning">AutoML</button>&nbsp;
                    <button value="AutoTSEstimator" class="roundbutton">AutoTSEstimator</button>
                </p>
            </summary>
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/network_traffic/network_traffic_autots_forecasting.ipynb">View source on GitHub</a>
            <p>In this notebook, we demostrate a reference use case where we use the network traffic KPI(s) in the past to predict traffic KPI(s) in the future. We demostrate how to use <span>AutoTS</span> in project <span><a href="https://github.com/intel-analytics/bigdl/tree/main/python/chronos/src/bigdl/chronos">Chronos</a></span> to do time series forecasting in an automated and distributed way.</p>
        </details>
        <hr>

        <details id="MultvarWIDE">
            <summary>
                <a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/network_traffic/network_traffic_model_forecasting.ipynb">Multivariate Forecasting of Network Traffic at the Transit Link of WIDE</a>
                <p>Tag: 
                    <button value="forecast">forecast</button>&nbsp;
                    <button value="LSTMForecaster" class="roundbutton">LSTMForecaster</button>
                </p>
            </summary>
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/network_traffic/network_traffic_model_forecasting.ipynb">View source on GitHub</a>
            <p>In this notebook, we demonstrate a reference use case where we use the network traffic KPI(s) in the past to predict traffic KPI(s) in the future. We demostrate how to do univariate forecasting (predict only 1 series), and multivariate forecasting (predicts more than 1 series at the same time) using Project <span><a href="https://github.com/intel-analytics/bigdl/tree/main/python/chronos/src/bigdl/chronos">Chronos</a></span>.</p>
        </details>
        <hr>

        <details id="MultstepWIDE">
            <summary>
                <a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/network_traffic/network_traffic_multivariate_multistep_tcnforecaster.ipynb">Multistep Forecasting of Network Traffic at the Transit Link of WIDE</a>
                <p>Tag: 
                    <button value="forecast">forecast</button>&nbsp;
                    <button value="TCNForecaster" class="roundbutton">TCNForecaster</button>
                </p>
            </summary>
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/network_traffic/network_traffic_multivariate_multistep_tcnforecaster.ipynb">View source on GitHub</a>
            <p>In this notebook, we demonstrate a reference use case where we use the network traffic KPI(s) in the past to predict traffic KPI(s) in the future. We demostrate how to do multivariate multistep forecasting using Project <span><a href="https://github.com/intel-analytics/bigdl/tree/main/python/chronos/src/bigdl/chronos">Chronos</a></span>.</p>
        </details>
        <hr>

        <details id="LSTMF">
            <summary>
            <a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/fsi/stock_prediction.ipynb">Stock Price Prediction with LSTMForecaster</a>
            <p>Tag: 
                <button value="forecast">forecast</button>&nbsp;
                <button value="LSTMForecaster" class="roundbutton">LSTMForecaster</button>
            </p>
            </summary>          
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/fsi/stock_prediction.ipynb">View source on GitHub</a>
            <p>In this notebook, we demonstrate a reference use case where we use historical stock price data to predict the future price. The dataset we use is the daily stock price of S&P500 stocks during 2013-2018 (data source). We demostrate how to do univariate forecasting using the past 80% of the total days' MMM price to predict the future 20% days' daily price.</p>
            <p>Reference: <span><a href="https://github.com/jwkanggist/tf-keras-stock-pred">https://github.com/jwkanggist/tf-keras-stock-pred</a></span></p>
        </details>
        <hr>

        <details id="AutoPr">
            <summary>
                <a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/fsi/stock_prediction_prophet.ipynb">Stock Price Prediction with ProphetForecaster and AutoProphet</a>
                <p>Tag: 
                    <button value="forecast">forecast</button>&nbsp;
                    <button value="hyperparameter_tuning">AutoML</button>&nbsp;
                    <button value="AutoProphet" class="roundbutton">AutoProphet</button>
                </p>
            </summary>
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/fsi/stock_prediction_prophet.ipynb">View source on GitHub</a>
            <p>In this notebook, we demonstrate a reference use case where we use historical stock price data to predict the future price using the ProphetForecaster and AutoProphet. The dataset we use is the daily stock price of S&P500 stocks during 2013-2018 <span><a href="https://www.kaggle.com/camnugent/sandp500/">data source</a></span>.</p>
            <p>Reference: <span><a href="https://facebook.github.io/prophet">https://facebook.github.io/prophet</a></span>, <span><a href="https://github.com/jwkanggist/tf-keras-stock-pred">https://github.com/jwkanggist/tf-keras-stock-pred</a></span></p>
        </details>
        <hr>

        <details id="Unsupervised">
            <summary>
                <a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised.ipynb">Unsupervised Anomaly Detection for CPU Usage</a>
                <p>Tag: 
                    <button value="anomaly_detection">anomaly detection</button>&nbsp;
                    <button value="DBScanDetector" class="roundbutton">DBScanDetector</button>
                </p>
            </summary>
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised.ipynb">View source on GitHub</a>
            <p>We demonstrates how to perform anomaly detection based on Chronos's built-in <span><a href="../../PythonAPI/Chronos/anomaly_detectors.html#dbscandetector">DBScanDetector</a></span>, <span><a href="../../PythonAPI/Chronos/anomaly_detectors.html#aedetector">AEDetector</a></span> and <span><a href="../../PythonAPI/Chronos/anomaly_detectors.html#thresholddetector">ThresholdDetector</a></span>.</p>
        </details>
        <hr>

        <details id="AnomalyDetection">
            <summary>
                <a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised_forecast_based.ipynb">Anomaly Detection for CPU Usage Based on Forecasters</a>
                <p>Tag: 
                    <button value="forecast">forecast</button>&nbsp;
                    <button value="anomaly_detection">anomaly detection</button>&nbsp;
                    <button value="MTNetForecaster" class="roundbutton">MTNetForecaster</button>
                </p>
            </summary>
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised_forecast_based.ipynb">View source on GitHub</a>
            <p>We demonstrates how to leverage Chronos's built-in models ie. MTNet, to do time series forecasting. Then perform anomaly detection on predicted value with <span><a href="../../PythonAPI/Chronos/anomaly_detectors.html#thresholddetector">ThresholdDetector</a></span>.</p>
        </details>
        <hr>

        <details id="DeepARmodel">
            <summary>
                <a href="https://github.com/intel-analytics/BigDL/tree/main/python/chronos/use-case/pytorch-forecasting/DeepAR">Help pytorch-forecasting improve the training speed of DeepAR model</a>
                <p>Tag: 
                    <button value="forecast">forecast</button>&nbsp;
                    <button value="customized_model">customized model</button>&nbsp;
                    <button value="DeepAR" class="roundbutton">DeepAR</button>
                </p>
            </summary>
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/tree/main/python/chronos/use-case/pytorch-forecasting/DeepAR">View source on GitHub</a>
            <p>Chronos can help a 3rd party time series lib to improve the performance (both training and inferencing) and accuracy. This use-case shows Chronos can easily help pytorch-forecasting speed up the training of DeepAR model.</p>
        </details>
        <hr>

        <details id="TFTmodel">
            <summary>
                <a href="https://github.com/intel-analytics/BigDL/tree/main/python/chronos/use-case/pytorch-forecasting/TFT">Help pytorch-forecasting improve the training speed of TFT model</a>
                <p>Tag: 
                    <button value="forecast">forecast</button>&nbsp;
                    <button value="customized_model">customized model</button>&nbsp;
                    <button value="TFT_model" class="roundbutton">TFT Model</button>
                </p>
            </summary>
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/tree/main/python/chronos/use-case/pytorch-forecasting/TFT">View source on GitHub</a>
            <p>Chronos can help a 3rd party time series lib to improve the performance (both training and inferencing) and accuracy. This use-case shows Chronos can easily help pytorch-forecasting speed up the training of TFT model.</p>
        </details>
        <hr>

        <details id="hyperparameter">
            <summary>
                <a href="https://github.com/intel-analytics/BigDL/tree/main/python/chronos/example/hpo/muti_objective_hpo_with_builtin_latency_tutorial.ipynb">Tune a Time Series Forecasting Model with multi-objective hyperparameter optimization.</a>
                <p>Tag: 
                    <button value="forecast">forecast</button>&nbsp;
                    <button value="hyperparameter_tuning">AutoML</button>&nbsp;
                    <button value="TCNForecaster" class="roundbutton">TCNForecaster</button>
                </p>
            </summary>
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/tree/main/python/chronos/example/hpo/muti_objective_hpo_with_builtin_latency_tutorial.ipynb">View source on GitHub</a>
            <p>In this notebook, we demostrate how to use <span>multi-objective hyperparameter optimization with built-in latency metric</span> in project <span><a href="https://github.com/intel-analytics/bigdl/tree/main/python/chronos/src/bigdl/chronos">Chronos</a></span> to do time series forecasting and achieve good tradeoff between performance and latency.</p>
        </details>
        <hr>

        <details id="taxiDataset">
            <summary>
                <a href="https://github.com/intel-analytics/BigDL/tree/main/python/chronos/example/auto_model">Auto tuning prophet on nyc taxi dataset</a>
                <p>Tag: 
                    <button value="forecast">forecast</button>&nbsp;
                    <button value="hyperparameter_tuning">AutoML</button>&nbsp;
                    <button value="AutoLSTM" class="roundbutton">AutoLSTM</button>&nbsp;
                    <button value="AutoProphet" class="roundbutton">AutoProphet</button>
                </p>
            </summary>
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/tree/main/python/chronos/example/auto_model">View source on GitHub</a>
            <p>This example collection will demonstrate Chronos auto models (i.e. autolstm & autoprophet) perform automatic time series forecasting on nyc_taxi dataset. The auto model will search the best hyperparameters automatically.</p>
        </details>
        <hr>

        <details id="distributedFashion">
            <summary>
                <a href="https://github.com/intel-analytics/BigDL/tree/main/python/chronos/example/distributed">Use Chronos forecasters in a distributed fashion</a>
                <p>Tag: 
                    <button value="forecast">forecast</button>&nbsp;
                    <button value="distributed">distributed</button>&nbsp;
                    <button value="Seq2SeqForecaster" class="roundbutton">Seq2SeqForecaster</button>&nbsp;
                    <button value="TCNForecaster" class="roundbutton">TCNForecaster</button>
                </p>
            </summary>
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/tree/main/python/chronos/example/distributed">View source on GitHub</a>
            <p>Users can easily train their forecasters in a distributed fashion to handle extra large dataset and speed up the process (training and data processing) by utilizing a cluster or pseudo-distribution on a single node. The functionality is powered by Project Orca.</p>
        </details>
        <hr>

        <details id="ONNX">
            <summary>
                <a href="https://github.com/intel-analytics/BigDL/tree/main/python/chronos/example/onnx">Use ONNXRuntime to speed-up forecasters' inferecing</a>
                <p>Tag: 
                    <button value="forecast">forecast</button>&nbsp;
                    <button value="onnxruntime">onnxruntime</button>&nbsp;
                    <button value="hyperparameter_tuning">AutoML</button>&nbsp;
                    <button value="AutoTSEstimator" class="roundbutton">AutoTSEstimator</button>&nbsp;
                    <button value="Seq2SeqForecaster" class="roundbutton">Seq2SeqForecaster</button>
                </p>
            </summary>
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/tree/main/python/chronos/example/onnx">View source on GitHub</a>
            <p>This example will demonstrate how to use ONNX to speed up the inferencing(prediction/evaluation) on forecasters and AutoTSEstimator. In this example, onnx speed up the inferencing for ~4X.</p>
        </details>
        <hr>

        <details id="Quantize">
            <summary>
                <a href="https://github.com/intel-analytics/BigDL/tree/main/python/chronos/example/quantization">Quantize Chronos forecasters method to speed-up inference</a>
                <p>Tag: 
                    <button value="forecast">forecast</button>&nbsp;
                    <button value="quantization">quantization</button>&nbsp;
                    <button value="TCNForecaster" class="roundbutton">TCNForecaster</button>
                </p>
            </summary>
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/tree/main/python/chronos/example/quantization">View source on GitHub</a>
            <p>Users can easily quantize their forecasters to low precision and speed up the inference process (both throughput and latency) by on a single node. The functionality is powered by Project Nano.</p>
        </details>
        <hr>

        <details id="SimualateTimeSeriesData">
            <summary>
                <a href="https://github.com/intel-analytics/BigDL/tree/main/python/chronos/example/simulator">Simualate time series data with similar pattern as example data</a>
                <p>Tag: 
                    <button value="simulation">simulation</button>&nbsp;
                    <button value="DPGANSimulator" class="roundbutton">DPGANSimulator</button>
                </p>
            </summary>
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/tree/main/python/chronos/example/simulator">View source on GitHub</a>
            <p>This example shows how to generate synthetic data with similar distribution as training data with the fast and easy DPGANSimulator API provided by Chronos.</p>
        </details>
        <hr>

        <details id="TCMF">
            <summary>
                <a href="https://github.com/intel-analytics/BigDL/tree/main/python/chronos/example/tcmf">High dimension time series forecasting with Chronos TCMFForecaster</a>
                <p>Tag: 
                    <button value="forecast">forecast</button>&nbsp;
                    <button value="distributed">distributed</button>&nbsp;
                    <button value="TCMFForecaster" class="roundbutton">TCMFForecaster</button>
                </p>
            </summary>
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/tree/main/python/chronos/example/tcmf">View source on GitHub</a>
            <p>This example demonstrates how to use BigDL Chronos TCMFForecaster to run distributed training and inference for high dimension time series forecasting task.</p>
        </details>
        <hr>

        <details id="PenalizeUnderestimation">
            <summary>
                <a href="https://github.com/intel-analytics/BigDL/tree/main/python/chronos/example/loss/penalize_underestimation.ipynb">Penalize underestimation with LinexLoss</a>
                <p>Tag: 
                    <button value="forecast">forecast</button>&nbsp;
                    <button value="TCNForecaster" class="roundbutton">TCNForecaster</button>
                </p>
            </summary>
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/tree/main/python/chronos/example/loss/penalize_underestimation.ipynb">View source on GitHub</a>
            <p>This example demonstrates how to use TCNForecaster to penalize underestimation based on a built-in loss function LinexLoss.</p>
        </details>
        <hr>

        <details id="GPUtrainingCPUacceleration">
            <summary>
                <a href="https://github.com/intel-analytics/BigDL/tree/main/python/chronos/example/inference-acceleration">Accelerate the inference speed of model trained on other platform</a>
                <p>Tag: 
                    <button value="forecast">forecast</button>&nbsp;
                    <button value="customized_model">customized model</button>&nbsp;
                    <button value="TCNForecaster" class="roundbutton">TCNForecaster</button>
                </p>
            </summary>
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/tree/main/python/chronos/example/inference-acceleration">View source on GitHub</a>
            <p>In this example, we show an example to train the model on GPU and accelerate the model by using onnxruntime on CPU.</p>
        </details>
        <hr>

        <details id="ServeForecaster">
            <summary>
                <a href="https://github.com/intel-analytics/BigDL/tree/main/python/chronos/example/serving">Serve Chronos forecaster and predict through TorchServe</a>
                <p>Tag: 
                    <button value="forecast">forecast</button>&nbsp;
                    <button value="TCNForecaster" class="roundbutton">TCNForecaster</button>
                </p>
            </summary>
            <img src="../../../_images/GitHub-Mark-32px.png"><a href="https://github.com/intel-analytics/BigDL/tree/main/python/chronos/example/serving">View source on GitHub</a>
            <p>In this example, we show how to serve Chronos forecaster and predict through TorchServe.</p>
        </details>
        <hr>

    </div>

    <script src="../../../_static/js/chronos_tutorial.js"></script> 
```