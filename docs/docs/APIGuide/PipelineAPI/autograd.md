## AutoGrad

### mean

 Mean of a `Variable`, alongside the specified axis.
- `axis` axis to compute the mean. 0-based indexed.
- `keepDims` A boolean, whether to keep the dimensions or not.
   If `keepdims` is `False`, the rank of the tensor is reduced
   by 1. If `keep_dims` is `True`,
   the reduced dimensions are retained with length 1.
   
   
**Scala example**
```scala
mean(x: Variable[T], axis: Int = 0, keepDims: Boolean = false)
```


**Python example**
```python
mean(a, axis=0, keepDims=False):

```