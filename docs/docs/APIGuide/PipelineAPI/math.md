
## mean

 Mean of a `Variable`, alongside the specified axis.
- `axis` axis to compute the mean. 0-based indexed.
- `keepDims` A boolean, whether to keep the dimensions or not.
   If `keepDims` is `False`, the rank of the `Variable` is reduced
   by 1. If `keepDims` is `True`,
   the reduced dimensions are retained with length 1.
   
   
**Scala example**
```scala
mean(x: Variable[T], axis: Int = 0, keepDims: Boolean = false)
```


**Python example**
```python
mean(x, axis=0, keepDims=False):
```

## abs

 Element-wise absolute value.
- `x` A `Variable`.
   
   
**Scala example**
```scala
abs(x: Variable[T])
```


**Python example**
```python
abs(x):
```

## sum

 Sum of the values in a `Variable`, alongside the specified axis.
- `axis` axis to compute the mean. 0-based indexed.
- `keepDims` A boolean, whether to keep the dimensions or not.
   If `keepdims` is `False`, the rank of the `Variable` is reduced
   by 1. If `keep_dims` is `True`,
   the reduced dimensions are retained with length 1.
   
   
**Scala example**
```scala
sum(x: Variable[T], axis: Int = 0, keepDims: Boolean = false)
```


**Python example**
```python
sum(x, axis=0, keepDims=False):
```

## clip

 Element-wise value clipping.
- `x` A `Variable`.
- `min` Double
- `max` Double
   
   
**Scala example**
```scala
clip(x: Variable[T], min: Double, max: Double)
```


**Python example**
```python
clip(x, min, max)
```

## square

 Element-wise square.
- `x` A `Variable`.
   
   
**Scala example**
```scala
square(x: Variable[T])
```


**Python example**
```python
square(x):
```

## sqrt

 Element-wise square root.
- `x` A `Variable`.
   
   
**Scala example**
```scala
sqrt(x: Variable[T])
```


**Python example**
```python
sqrt(x):
```

## maximum

 Element-wise maximum of two `Variables`.
- `x` A `Variable`.
- `y` A `Variable` or Double.
   
**Scala example**
```scala
maximum(x: Variable[T], y: Variable[T])
```


**Python example**
```python
maximum(x, y):
```

## log

 Element-wise log.
- `x` A `Variable`.
   
   
**Scala example**
```scala
log(x: Variable[T])
```


**Python example**
```python
log(x):
```

## exp

 Element-wise exponential.
- `x` A `Variable`.
   
   
**Scala example**
```scala
exp(x: Variable[T])
```


**Python example**
```python
exp(x):
```

## pow

 Element-wise exponentiation.
- `x` A `Variable`.
- `a` Double.   
   
**Scala example**
```scala
pow(x: Variable[T])
```


**Python example**
```python
pow(x):
```

## softsign

 Softsign of a `Variable`.
   
   
**Scala example**
```scala
softsign(x: Variable[T])
```


**Python example**
```python
softsign(x):
```

## softplus

 Softplus of a `Variable`.
   
   
**Scala example**
```scala
softplus(x: Variable[T])
```


**Python example**
```python
softplus(x):
```


## stack

   Stacks a list of rank `R` tensors into a rank `R+1` tensor.
   You should start from 1 as dim 0 is for batch.
   - inputs: List of variables (tensors)
   - axis: xis along which to perform stacking.
   
**Scala example**
```scala
def stack[T: ClassTag](inputs: List[Variable[T]], axis: Int = 1)
```


**Python example**
```python
def stack(inputs, axis=1)
```


## expand_dims

   Adds a 1-sized dimension at index "axis".
   
   
**Scala example**
```scala
def expandDims[T: ClassTag](x: Variable[T], axis: Int)
```


**Python example**
```python
expand_dims(x, axis)
```


## contiguous

  Turn the output and grad to be contiguous for the input Variable
   
   
**Scala example**
```scala
def contiguous[T: ClassTag](input: Variable[T])
```


**Python example**
```python
def contiguous(x)
```

