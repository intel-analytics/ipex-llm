# API Guide

## Python 

### class InputQueue
The class `Input` defines methods allowing you to input data into Cluster Serving [Input Pipeline]().

#### __init__

[view source]()

```
__init__(host=None, port=None, sync=False, frontend_url=None)
```
sets up a connection with configuration in your Cluster Serving [configuration file]() `config.yaml`.

_return_: None

#### enqueue
[view source]()

```
enqueue(uri, data*)
```
puts key-value pair into data pipeline, if your model has key regulation, pass corresponded key. Otherwise, give the key any name you want.

_return_: None

`uri`: a string, unique identification of your image

`data`: key-value pair of your data.

There are 4 types of inputs in total, string, image, tensor, sparse tensor. See following example to enqueue specific type of data.
_Example_
Import the dependency and create an instance of `InputQueue`
```
from zoo.serving.client import InputQueue
input_api = InputQueue()
```
To enqueue a list of string, pass a list of str objects, list of str type input is usually used in Tensorflow models.
```
input_api.enqueue('my-string-input', user_define_key=['hello', 'world'])
```
To enqueue an image, you could pass either image path or base64 encoded image bytes, the type of your parameter is identified by key of dict, see example below. If you pass image path, `cv2` package is required. (Could be installed by `pip install opencv-python`)

To pass image path, use key `path`
```
image_path = "path/to/image"
input_api.enqueue('my-image1', user_define_key={"path": image_path})
```
To pass base64 encoded image bytes, use key `b64`
```
image_bytes = "base64_encoded_bytes"
input_api.enqueue('my-image1', user_define_key={"b64": image_bytes})
```

To enqueue a tensor or sparse tensor, `numpy` package is required. (Would be installed while you installed Analytics Zoo, if not, could be installed by `pip install numpy`)

To enqueue a tensor, pass a ndarray object.
```
import numpy as np
input_api.enqueue('my-tensor1', a=np.array([1,2]))
```
To enqueue a sparse tensor pass a list of ndarray object, normally sparse tensor is only used if your model specifies the input as sparse tensor. The list should have size of 3, where the 1st element is a 2-D ndarray, representing the indices of values, the 2nd element is a 1-D ndarray, representing the values corresponded with the indices, the 3rd element is a 1-D ndarray representing the shape of the sparse tensor.

A sparse tensor of shape (5,10), with 3 elements at position (0, 0), (1, 2), (4, 5), having value 101, 102, 103, visualized as following,
```
101. 0.   0.   0.   0.   0.   0.   0.   0.   0
0.   0.   102. 0.   0.   0.   0.   0.   0.   0
0.   0.   0.   0.   0.   0.   0.   0.   0.   0
0.   0.   0.   0.   0.   0.   0.   0.   0.   0
0.   0.   0.   0.   0.   103. 0.   0.   0.   0
```

could be represented as following list.
```
indices = np.array([[0, 1, 4], [0, 2, 5]])
values = np.array([101, 102, 103])
shape = np.array([5, 10])
tensor = [indices, values, shape]
```
and enqueue it by
```
input_api.enqueue("my-sparse-tensor", input=tensor)
```

To enqueue an instance containing several images, tensors and sparse tensors.
```
import numpy as np
input_api.enqueue('my-instance', 
    img1={"path: 'path/to/image1'},
    img2={"path: 'path/to/image2'},
    tensor1=np.array([1,2]), 
    tensor2=np.array([[1,3],[2,3]])
    sparse_tensor=[np.array([[0, 1, 4], [0, 2, 5]]),
                   np.array([101, 102, 103])
                   np.array([5, 10])]
)
```
#### __predict__
[view source]()
```
predict(request_str)
```
_return_: Numpy ndarray
 
_Example_
```
from zoo.serving.client import InputQueue
input_api = InputQueue(sync=True, frontend_url=frontend_server_url)
request_json_string='''{
  "instances" : [ {
    "ids" : [ 100.0, 88.0 ]
  }]
}'''
response = input_api.predict(request_json_string)
print(response.text)
```
### class OutputQueue
The class `Output` defines methods allowing you to get result from Cluster Serving [Output Pipeline]().
#### __init__
[view source]()

```
__init__()
```
sets up a connection with configuration in your Cluster Serving [configuration file]() `config.yaml`.
#### query
[view source]()

```
query(uri)
```
query result in output Pipeline by key `uri`

_return_: Numpy ndarray

Format: 
```
{
    "class_1": "probability_1",
    "class_2": "probability_2",
    ...,
    "class_n": "probability_n"
}
```
where `n` is `top_n` in your serving config, the result is sorted by output probability.

_Example_
```
from zoo.serving.client import OutputQueue
import json
output_api = OutputQueue()
d = output_api.query('my-image') 

tmp_dict = json.loads(d)
for class_idx in tmp_dict.keys():
    output += "class: " + class_idx + "'s prob: " + tmp_dict[class_idx]
print(output)
```

#### dequeue
[view source]()

```
dequeue()
```
gets all result of your model prediction and dequeue them from OutputQueue

_return_: dict(), with keys the `uri` of your [enqueue](), string type, and values the output of your prediction, Numpy ndarray

Format: 
```
{
  "image1": {
      "class_1": "probability_1",
      "class_2": "probability_2",
      ...,
      "class_n": "probability_n"
  }, 
  "image2": {
      "class_1": "probability_1",
      "class_2": "probability_2",
      ...,
      "class_n": "probability_n"
  }
  ...
}
```

where `n` is `top_n` in your serving config, the result is sorted by output probability.

_Example_
```
from zoo.serving.client import OutputQueue
import json
output_api = OutputQueue()
d = output_api.dequeue()

for k in d.keys():
    output = "image: " + k + ", classification-result:"
    tmp_dict = json.loads(result[k])
    for class_idx in tmp_dict.keys():
        output += "class: " + class_idx + "'s prob: " + tmp_dict[class_idx]
    print(output)
```



