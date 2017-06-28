## CAdd ##

**Scala:**
```scala
val module = CAdd(size,bRegularizer=null)
```
**Python:**
```python
module = CAdd(size,bRegularizer=None,bigdl_type="float")
```

This layer has a bias tensor with given size. The bias will be added element wise to the input
tensor. If the element number of the bias tensor match the input tensor, a simply element wise
will be done. Or the bias will be expanded to the same size of the input. The expand means
repeat on unmatched singleton dimension(if some unmatched dimension isn't singleton dimension,
it will report an error). If the input is a batch, a singleton dimension will be add to the first
dimension before the expand.

 * @param size the size of the bias 

**Scala example:**
```scala
val module = CAdd[Float](Array(2, 1),bRegularizer=null)
val input = Tensor[Float](2, 3).rand()
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.52146345      0.86262375      0.74210143
0.15882674      0.026310394     0.28394955
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x3]

module.forward(input)
res12: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.97027373      1.311434        1.1909117
-0.047433108    -0.17994945     0.07768971
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
module = CAdd([2, 1],bRegularizer=None,bigdl_type="float")
input = np.random.rand(2, 3)
array([[ 0.71239789,  0.65869477,  0.50425182],
       [ 0.40333312,  0.64843273,  0.07286636]])

module.forward(input)
array([[ 0.89537328,  0.84167016,  0.68722725],
       [ 0.1290929 ,  0.37419251, -0.20137388]], dtype=float32)
```
