---
## **Introduction**

Analytics Zoo Orca OpenVINO Estimator provides a set APIs for running OpenVINO model on Spark in a distributed fashion.

---

### Orca OpenVINO Estimator

Orca OpenVINO Estimator is an estimator to do OpenVINO prediction on Spark in a distributed fashion.

It can support various data types, like XShards, ndarray, list of ndarray, etc.

### Create Estimator from OpenVINO Model

You can create Orca OpenVINO Estimator with OpenVINO IR xml file and bin file.

```
from zoo.orca.learn.openvino.estimator import Estimator

Estimator.from_openvino(*, model_path, batch_size=0)
```

* `model_path`: (string) The file path to the OpenVINO IR xml file. Please put the OpenVINO IR bin file in the same folder with the xml file.
* `batch_size`: (int) Set batch Size, default is 0 (use default batch size).

### Inference with Orca OpenVINO Estimator

After an Estimator is created, you can call estimator API to predict data:

```
predict(self, data)
```

* `data`:  Inference data. Ndarray, list of ndarrays and SparkXShards are supported.

### Load OpenVINO model

You can load an OpenVINO model using `load(self, model_path, batch_size=0)`

* `model_path`: (string) The file path to the OpenVINO IR xml file. Please put the OpenVINO IR bin file in the same folder with the xml file.
* `batch_size`: (int) Set batch Size, default is 0 (use default batch size).
