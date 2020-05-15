# API Guide

## Python 

### class InputQueue
The class `Input` defines methods allowing you to input data into Cluster Serving [Input Pipeline]().

#### __init__

[view source]()

```
__init__()
```
sets up a connection with configuration in your Cluster Serving [configuration file]() `config.yaml`.

_return_: None

#### enqueue_image
[view source]()

```
enqueue_image(uri, img)
```
puts image `img` with identification `uri` into Pipeline with JPG encoding. `img` can be either a string (which represents the file path of the image), or an ndarray of the image (which should be returned by cv2.imread() of opencv-python package)

_return_: None

`uri`: a string, unique identification of your image

`img`: path or `ndarray` of your image, could be loaded by `cv2.imread()` of opencv-python package.

_Example_
```
from zoo.serving.client import InputQueue
input_api = InputQueue()
input_api.enqueue_image('my-image1', 'path/to/image1')

import cv2
image2 = cv2.imread('path/to/image2')
input_api.enqueue_image('my-image2', image2)
```

#### enqueue_tensor
[view source]()

```
enqueue_tensor(uri, data)
```
puts ndarray or list of ndarray `data` with identification `uri` into Pipeline. 

_return_: None

`uri`: a string, unique identification of your input

`data`: list of ndarray or ndarray

_Example_
```
from zoo.serving.client import InputQueue
import numpy as np
input_api = InputQueue()
sample1 = np.array([1, 2])
input_api.enqueue_tensor("sample1", sample1)

sample2 = [np.array([1, 2]), np.array([[3, 4], [5, 6]])]
input_api.enqueue_tensor("sample2", sample2)
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

_return_: string type, the output of your prediction, which can be parsed to a dict by json. 

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

_return_: dict(), with keys the `uri` of your [enqueue](), string type, and values the output of your prediction, string type, which can be parsed by json. 

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



