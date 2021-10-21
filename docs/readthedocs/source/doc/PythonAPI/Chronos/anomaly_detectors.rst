Anomaly Detectors
=====================

chronos.anomaly.ae_detector
----------------------------------------

AEDetector is unsupervised anomaly detector. It builds an autoencoder network, tries to fit the model to the input data, and calcuates the reconstruction error. The samples with larger reconstruction errors are more likely the anomalies.

.. automodule:: bigdl.chronos.detector.anomaly.ae_detector
    :members:
    :show-inheritance:


chronos.anomaly.dbscan_detector
----------------------------------------

DBScanDetector uses DBSCAN clustering for anomaly detection. The DBSCAN algorithm tries to cluster the points and label the points that do not belong to any clusters as -1. It thus detects outliers in the input time series.

.. automodule:: bigdl.chronos.detector.anomaly.dbscan_detector
    :members:
    :show-inheritance:


chronos.anomaly.th_detector
----------------------------------------

ThresholdDetector is a simple anomaly detector that detectes anomalies based on threshold. The target value for anomaly testing can be either 1) the sample value itself or 2) the difference between the forecasted value and the actual value, if the forecasted values are provied. The thresold can be set by user or esitmated from the train data accoring to anomaly ratio and statistical distributions.

.. automodule:: bigdl.chronos.detector.anomaly.th_detector
    :members: ThresholdDetector
    :show-inheritance:
