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

#### enqueue
[view source]()

```
enqueue(uri, data*)
```
puts key-value pair into data pipeline, if your model has key regulation, pass corresponded key. Otherwise, give the key any name you want.

_return_: None

`uri`: a string, unique identification of your image

`data`: key-value pair of your data.
_Example_
To enqueue an image
```
from zoo.serving.client import InputQueue
input_api = InputQueue()
input_api.enqueue('my-image1', user_define_key='path/to/image1')
```
To enqueue an instance containing 1 image and 2 ndarray
```
from zoo.serving.client import InputQueue
import numpy as np
input_api = InputQueue()
t1 = np.array([1,2])
t2 = np.array([[1,2], [3,4]])
input_api.enqueue_image('my-instance', img='path/to/image', tensor1=t1, tensor2=t2)
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



