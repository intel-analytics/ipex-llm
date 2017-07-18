## Input ##

**Scala:**
```scala
val input = Input()
```
**Python:**
```python
input = Input()
```

Input layer do nothing to the input tensors, just pass them. It should be used as input node
when the first layer of your module accepts multiple tensors as inputs.

Each input node of the graph container should accept one tensor as input. If you want a module
accepting multiple tensors as input, you should add some Input module before it and connect
the outputs of the Input nodes to it. Please see the example of the Graph document.

Please note that the return is not a layer but a Node containing input layer.

**Scala example:**
```scala
val module = Input()
val input = Tensor(3, 2).rand()
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.24058184      0.22737113
0.0028103297    0.18359558
0.80443156      0.07047854
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x2]

module.element.forward(input)
res13: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.24058184      0.22737113
0.0028103297    0.18359558
0.80443156      0.07047854
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x2]
```

**Python example:**
```python
module = Input()
input = np.random.rand(3,2)
[array([
[ 0.87273163,  0.59974301],
[ 0.09416127,  0.135765  ],
[ 0.11577505,  0.46095625]], dtype=float32)]

module.element().forward(input)
com.intel.analytics.bigdl.nn.Echo@535c681 : Activation size is 3x2
[array([
[ 0.87273163,  0.59974301],
[ 0.09416127,  0.135765  ],
[ 0.11577505,  0.46095625]], dtype=float32)]

```
