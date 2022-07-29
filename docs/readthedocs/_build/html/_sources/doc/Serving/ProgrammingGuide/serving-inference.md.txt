# Inference by Cluster Serving

## Model Inference
Once you finish the installation and service launch, you could do inference using Cluster Serving client API.

We support Python API and HTTP RESTful API for conducting inference with Data Pipeline in Cluster Serving. 

### Python API
For Python API, the requirements of python packages are `opencv-python`(for raw image only), `pyyaml`, `redis`. You can use `InputQueue` and `OutputQueue` to connect to data pipeline by providing the pipeline url, e.g. `my_input_queue = InputQueue(host, port)` and `my_output_queue = OutputQueue(host, port)`. If parameters are not provided, default url `localhost:6379` would be used.

We provide some basic usages here, for more details, please see [API Guide](APIGuide.md).

To input data to queue, you need a `InputQueue` instance, and using `enqueue` method, for each input, give a key correspond to your model or give arbitrary key if your model does not care about it.

To enqueue an image
```
from bigdl.serving.client import InputQueue
input_api = InputQueue()
input_api.enqueue('my-image1', user_define_key={"path: 'path/to/image1'})
```
To enqueue an instance containing 1 image and 2 ndarray
```
from bigdl.serving.client import InputQueue
import numpy as np
input_api = InputQueue()
t1 = np.array([1,2])
t2 = np.array([[1,2], [3,4]])
input_api.enqueue('my-instance', img={"path": 'path/to/image'}, tensor1=t1, tensor2=t2)
```
There are 4 types of inputs in total, string, image, tensor, sparse tensor, which could represents nearly all types of models. For more details of usage, go to [API Guide](APIGuide.md)

To get data from queue, you need a `OutputQueue` instance, and using `query` or `dequeue` method. The `query` method takes image uri as parameter and returns the corresponding result. The `dequeue` method takes no parameter and just returns all results and also delete them in data queue. See following example.
```
from bigdl.serving.client import OutputQueue
output_api = OutputQueue()
img1_result = output_api.query('img1')
all_result = output_api.dequeue() # the output queue is empty after this code
```
Consider the code above,
```
img1_result = output_api.query('img1')
```
##### Sync API
Python API is a pub-sub schema async API. Specifically, thread would not block once you call `enqueue` method. If you want the thread to block, see this section.

To use sync API, create a `InputQueue` instance with `sync=True` and `frontend_url=frontend_server_url` argument.
```
from bigdl.serving.client import InputQueue
input_api = InputQueue(sync=True, frontend_url=frontend_server_url)
response = input_api.predict(request_json_string)
print(response.text)
```
example of `request_json_string` is
```
'{
  "instances" : [ {
    "ids" : [ 100.0, 88.0 ]
  }]
}'
```
This API is also a python support of [Restful API](#restful-api) section, so for more details of input format, refer to it.
### RESTful API
RESTful API uses serving HTTP server.
This part describes API endpoints and end-to-end examples on usage. 
The requests and responses are in JSON format. The composition of them depends on the requests type or verb. See the APIs for details.
In case of error, all APIs will return a JSON object in the response body with error as key and the error message as the value:
```
{
  "error": <error message string>
}
```
#### Predict API
URL
```
POST http://host:port/predict
```
Request Example for images as inputs:
```
curl -d \
'{
  "instances": [
    {
      "image": "/9j/4AAQSkZJRgABAQEASABIAAD/7RcEUGhvdG9za..."
    },   
    {
      "image": "/9j/4AAQSkZJRgABAQEASABIAAD/7RcEUGhvdG9za..."
    }
  ]
}' \
-X POST http://host:port/predict
```
Response Example
```
{
  "predictions": [
    "{value=[[903,0.1306194]]}",    
    "{value=[[903,0.1306194]]}"
  ]
}
```
Request Example for tensor as inputs:
```
curl -d \
'{
  "instances" : [ {
    "ids" : [ 100.0, 88.0 ]
  }, {
    "ids" : [ 100.0, 88.0 ]
  } ]
}' \
-X POST http://host:port/predict
```
Response Example
```
{
  "predictions": [
    "{value=[[1,0.6427843]]}",
    "{value=[[1,0.6427842]]}"
  ]
}
```
Another request example for composition of scalars and tensors.
```
curl -d \
 '{
  "instances" : [ {
    "intScalar" : 12345,
    "floatScalar" : 3.14159,
    "stringScalar" : "hello, world. hello, arrow.",
    "intTensor" : [ 7756, 9549, 1094, 9808, 4959, 3831, 3926, 6578, 1870, 1741 ],
    "floatTensor" : [ 0.6804766, 0.30136853, 0.17394465, 0.44770062, 0.20275897, 0.32762378, 0.45966738, 0.30405098, 0.62053126, 0.7037923 ],
    "stringTensor" : [ "come", "on", "united" ],
    "intTensor2" : [ [ 1, 2 ], [ 3, 4 ], [ 5, 6 ] ],
    "floatTensor2" : [ [ [ 0.2, 0.3 ], [ 0.5, 0.6 ] ], [ [ 0.2, 0.3 ], [ 0.5, 0.6 ] ] ],
    "stringTensor2" : [ [ [ [ "come", "on", "united" ], [ "come", "on", "united" ], [ "come", "on", "united" ], [ "come", "on", "united" ] ], [ [ "come", "on", "united" ], [ "come", "on", "united" ], [ "come", "on", "united" ], [ "come", "on", "united" ] ] ], [ [ [ "come", "on", "united" ], [ "come", "on", "united" ], [ "come", "on", "united" ], [ "come", "on", "united" ] ], [ [ "come", "on", "united" ], [ "come", "on", "united" ], [ "come", "on", "united" ], [ "come", "on", "united" ] ] ] ]
  }]
}' \
-X POST http://host:port/predict
```
Another request example for composition of sparse and dense tensors.
```
curl -d \
'{
  "instances" : [ {
    "sparseTensor" : {
      "shape" : [ 100, 10000, 10 ],
      "data" : [ 0.2, 0.5, 3.45, 6.78 ],
      "indices" : [ [ 1, 1, 1 ], [ 2, 2, 2 ], [ 3, 3, 3 ], [ 4, 4, 4 ] ]
    },
    "intTensor2" : [ [ 1, 2 ], [ 3, 4 ], [ 5, 6 ] ]
  }]
}' \
-X POST http://host:port/predict
```


#### Metrics API
URL
```
GET http://host:port/metrics
```
Response example:
```
[
  {
    name: "bigdl.serving.redis.get",
    count: 810,
    meanRate: 12.627772820651845,
    min: 0,
    max: 25,
    mean: 0.9687099303718213,
    median: 0.928579,
    stdDev: 0.8150031623593447,
    _75thPercentile: 1.000047,
    _95thPercentile: 1.141443,
    _98thPercentile: 1.268665,
    _99thPercentile: 1.608387,
    _999thPercentile: 25.874584
  }
]
```
## Logs and Visualization
To see outputs/logs, go to FLink UI -> job -> taskmanager, (`localhost:8081` by default), or go to `${FLINK_HOME}/logs`

To visualize the statistics, e.g. performance, go to Flink UI -> job -> metrics, and select the statistic to monitor
