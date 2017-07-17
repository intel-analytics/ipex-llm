## BinaryTreeLSTM ##

**Scala:**
```scala
val treeLSTM = BinaryTreeLSTM(
  inputSize,
  hiddenSize,
  gateOutput,
  withGraph)
```

**Python:**
```python
tree_lstm = BinaryTreeLSTM(
  input_size,
  hidden_size,
  gate_output,
  with_graph)
```

This class is an implementation of Binary TreeLSTM (Constituency Tree LSTM)
receiving [Constituency-based parse trees](https://en.wikipedia.org/wiki/Parse_tree#Constituency-based_parse_trees).
Tree-LSTM is a kind of recursive neural networks, which describes in the paper 
[Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](https://arxiv.org/abs/1503.00075)
 by Kai Sheng Tai, Richard Socher, and Christopher Manning.


**Parameters:**
* **inputSize** - the size of each input vector
* **hiddenSize** - hidden unit size in GRU
* **gateOutput** - whether gate the output. Default is `true`
* **withGraph** - whether create lstms with `com.intel.analytics.bigdl.nn.Graph`. Default is `true`.

**Scala example:**
```scala
    import com.intel.analytics.bigdl.numeric.NumericDouble
    import com.intel.analytics.bigdl.utils.RandomGenerator.RNG

    RNG.setSeed(100)

    val hiddenSize = 2
    val inputSize = 2

    val inputs =
      Tensor(
        T(T(T(1.0, 2.0),
          T(2.0, 3.0),
          T(4.0, 5.0))))

    val tree =
      Tensor(
        T(T(T(2.0, 5.0, -1.0),
          T(0.0, 0.0, 1.0),
          T(0.0, 0.0, 2.0),
          T(0.0, 0.0, 3.0),
          T(3.0, 4.0, 0.0))))

    val input = T(inputs, tree)

    val gradOutput =
      Tensor(
        T(T(T(2.0, 5.0),
          T(2.0, 3.0),
          T(4.0, 5.0),
          T(2.0, 3.0),
          T(4.0, 5.0),
          T(6.0, 7.0))))

    val model = BinaryTreeLSTM(inputSize, hiddenSize)

    val output = model.forward(input)
    println(output)
    (1,.,.) =
    -0.07799374051859737	-0.14419464399333934	
    -0.2349552348774636	-0.04679071771123799	
    -0.1594515102098235	-0.026039638054106272	
    -0.04540739978946999	-0.0070662412123771254	
    -0.05869603467391258	-0.13559056761784405

    [com.intel.analytics.bigdl.tensor.DenseTensor of size 1x5x2]  
    
    val gradInput = model.backward(input, gradOutput)
    println(gradInput)
     {
    	2: (1,.,.) =
    	   0.0	0.0	0.0	
    	   0.0	0.0	0.0	
    	   0.0	0.0	0.0	
    	   0.0	0.0	0.0	
    	   0.0	0.0	0.0	
    	   
    	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 1x5x3]
    	1: (1,.,.) =
    	   0.5614597104995146	-0.3383652016018004	
    	   0.8172036851171792	-0.46767634057453855	
    	   0.37739630380493044	-0.2335553148048936	
    	   
    	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 1x3x2]
     }
```

**Python example:**
```scala
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
import numpy as np

hidden_size = 2
input_size = 2
inputs = np.array([[
  [1.0, 2.0],
  [2.0, 3.0],
  [4.0, 5.0]
]])

tree = np.array([[
  [2.0, 5.0, -1.0],
  [0.0, 0.0, 1.0],
  [0.0, 0.0, 2.0],
  [0.0, 0.0, 3.0],
  [3.0, 4.0, 0.0]
]])

input = [inputs, tree]

grad_output = np.array([[
  [2.0, 3.0],
  [4.0, 5.0],
  [2.0, 3.0],
  [4.0, 5.0],
  [6.0, 7.0]
]])

model = BinaryTreeLSTM(input_size, hidden_size)
output = model.forward(input)
print output
[[[-0.08113038 -0.0289295 ]
  [ 0.1378704   0.00550814]
  [ 0.33053339 -0.02395477]
  [ 0.26895314 -0.02019646]
  [ 0.34085754 -0.12480961]]]
  
gradient = model.backward(input, grad_output)
print gradient
[array([[[ 0.43623093,  0.97416967],
        [-0.02283204,  0.99245077],
        [-1.11290622,  0.84173977]]], dtype=float32), array([[[ 0.,  0.,  0.],
        [ 0.,  0.,  0.],
        [ 0.,  0.,  0.],
        [ 0.,  0.,  0.],
        [ 0.,  0.,  0.]]], dtype=float32)]
```
 
