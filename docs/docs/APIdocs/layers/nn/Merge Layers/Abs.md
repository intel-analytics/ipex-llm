## Abs ##

**Scala:**
```scala
abs = Abs[T]()
```
**Python:**
```python
abs = Abs()
```

An element-wise abs operation.


**Scala example:**
```scala
val abs = new Abs[Float]
val input = Tensor[Float](2)
input(1) = 21
input(2) = -29
print(abs.forward(input))
```
`output is:　21.0　29.0`

**Python example:**
```python
abs = Abs()
input = np.array([21, -29, 30])
print(abs.forward(input))
```
`output is: [array([ 21.,  29.,  30.], dtype=float32)]`

