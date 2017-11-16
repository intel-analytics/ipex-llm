## TensorTree

TensorTree class is used to decode a tensor to a tree structure.
The given input `content` is a tensor which encodes a constituency parse tree.
The tensor should have the following structure:

Each row of the tensor represents a tree node and the row number is node number.
For each row, except the last column, all other columns represent the children
node number of this node. Assume the value of a certain column of the row is not zero,
the value `p` means this node has a child whose node number is `p` (lies in the `p`-th)
row. Each leaf has a leaf number, in the tensor, the last column represents the leaf number.
Each leaf does not have any children, so all the columns of a leaf except the last should
be zero. If a node is the root, the last column should equal to `-1`.

Note: if any row for padding, the padding rows should be placed at the last rows with all
elements equal to `-1`.

eg. a tensor represents a binary tree:

```
[11, 10, -1;
 0, 0, 1;
 0, 0, 2;
 0, 0, 3;
 0, 0, 4;
 0, 0, 5;
 0, 0, 6;
 4, 5, 0;
 6, 7, 0;
 8, 9, 0;
 2, 3, 0;
 -1, -1, -1;
 -1, -1, -1]
```

Parameters:
* `content` the tensor to be encoded

##TreeLSTM##

TreeLSTM is a base class of all other kinds of tree lstms,
, as described in the paper 
[Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](https://arxiv.org/abs/1503.00075)
 by Kai Sheng Tai, Richard Socher, and Christopher Manning.

---
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
Tree-LSTM is a kind of recursive neural networks, as described in the paper 
[Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](https://arxiv.org/abs/1503.00075)
 by Kai Sheng Tai, Richard Socher, and Christopher Manning.

The input tensor in `forward(input)` is expected to be a table, in which the first element is a 3D tensor (`batch x leaf number x inputSize`) and the second elment is the 3D embedding tensor tree
(`batch x tree node number x (number of branches + 1)`]. output of
`forward(input)` is expected to be a 3D tensor (`batch x tree node number x hiddenSize`).

Parameters:
* `inputSize` the size of each input vector
* `hiddenSize` hidden unit size in GRU
* `gateOutput` whether gate the output. Default is `true`
* `withGraph` whether create lstms with `com.intel.analytics.bigdl.nn.Graph`. Default is `true`.

**Scala example:**
```scala
    import com.intel.analytics.bigdl.numeric.NumericFloat
    import com.intel.analytics.bigdl.utils.RandomGenerator.RNG

    RNG.setSeed(100)

    val hiddenSize = 2
    val inputSize = 2

    val inputs =
      Tensor(
        T(T(T(1f, 2f),
          T(2f, 3f),
          T(4f, 5f))))

    val tree =
      Tensor(
        T(T(T(2f, 5f, -1f),
          T(0f, 0f, 1f),
          T(0f, 0f, 2f),
          T(0f, 0f, 3f),
          T(3f, 4f, 0f))))

    val input = T(inputs, tree)

    val gradOutput =
      Tensor(
        T(T(T(2f, 5f),
          T(2f, 3f),
          T(4f, 5f),
          T(2f, 3f),
          T(4f, 5f),
          T(6f, 7f))))

    val model = BinaryTreeLSTM(inputSize, hiddenSize)

    val output = model.forward(input)
    println(output)
    (1,.,.) =
    -0.07799375	-0.14419462	
    -0.23495524	-0.04679072	
    -0.15945151	-0.026039641	
    -0.0454074	-0.007066241	
    -0.058696028	-0.13559057	
    
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
     	   0.56145966	-0.3383652	
     	   0.81720364	-0.46767634	
     	   0.37739626	-0.23355529	
     	   
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
 

