## HingeEmbeddingCriterion ##

Creates a criterion that measures the loss given an input `x` which is a 1-dimensional vector and a label `y` (`1` or `-1`).
This is usually used for measuring whether two inputs are similar or dissimilar, e.g. using the L1 pairwise distance, and is typically used for learning nonlinear embeddings or semi-supervised learning.


**Scala:**
``` scala
criterion = HingeEmbeddingCriterion([margin: Double = 1,
                                       sizeAverage: Boolean = true])
```
**Python:**
```python
HingeEmbeddingCriterion(margin=1, size_average=True, bigdl_type="float")
```


**Scala example:**
```scala
val loss = HingeEmbeddingCriterion[Double](1)
val input = Tensor[Double](4, 1).randn()
println("input: " + input)
println("Target=1: " + loss.forward(input, Tensor[Double](4, 1).fill(1)))

println("Target=-1: " + loss.forward(input, Tensor[Double](4, 1).fill(-1)))
```

```
input:
    0.8630405188769112	
    1.1516645447482432	
    0.4556065208650613	
    1.1650460282049866	
    [com.intel.analytics.bigdl.tensor.DenseTensor of size 4x1]
Target=1: 0.9088394031738005
Target=-1: 0.17033824006450687

```

**Python example:**
```python
input = np.random.random(4)
target = np.full(4, 1)
print("input: " )
print(input)
print("target: ")
print(target)
HingeEmbeddingCriterion(1.0).forward(input, target)
```
```
input: 
[ 0.49766828  0.65098249  0.71942689  0.34369264]
target: 
[1 1 1 1]
creating: createHingeEmbeddingCriterion
0.5529426
```

