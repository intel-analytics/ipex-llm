## BiRecurrent ##

**Scala:**
```scala
val module = BiRecurrent(merge=null)
```
**Python:**
```python
module = BiRecurrent(merge=None,bigdl_type="float")
```

This layer implement a bidirectional recurrent neural network
 * @param merge concat or add the output tensor of the two RNNs. Default is add

**Scala example:**
```scala
val module = BiRecurrent(CAddTable())
.add(RnnCell(6, 4, Sigmoid()))
val input = Tensor(Array(1, 2, 6)).rand()
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
0.55511624      0.44330198      0.9025551       0.26096714      0.3434667       0.20060952
0.24903035      0.24026379      0.89252585      0.23025699      0.8131796       0.4013688

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 1x2x6]

module.forward(input)
res10: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
1.3577285       0.8861933       0.52908427      0.86278
1.2850789       0.82549953      0.5560188       0.81468254

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x4]
```

**Python example:**
```python
module = BiRecurrent(CAddTable()).add(RnnCell(6, 4, Sigmoid()))
input = np.random.rand(1, 2, 6)
array([[[ 0.75637438,  0.2642816 ,  0.61973312,  0.68565282,  0.73571443,
          0.17167681],
        [ 0.16439321,  0.06853251,  0.42257202,  0.42814042,  0.15706152,
          0.57866659]]])

module.forward(input)
array([[[ 0.69091094,  0.97150528,  0.9562254 ,  1.14894259],
        [ 0.83814102,  1.11358368,  0.96752423,  1.00913286]]], dtype=float32)
```
