## MultiLabelSoftMarginCriterion ##

**Scala:**
```scala
val module = MultiLabelSoftMarginCriterion()
```
**Python:**
```python
module = MultiLabelSoftMarginCriterion()
```

MultiLabelSoftMarginCriterion is a multiLabel multiclass criterion based on sigmoid:
```
l(x,y) = - sum_i y[i] * log(p[i]) + (1 - y[i]) * log (1 - p[i])
```
 where ```p[i] = exp(x[i]) / (1 + exp(x[i]))```

**Scala example:**
```scala
val criterion = MultiLabelSoftMarginCriterion()
val input = Tensor(3)
input(Array(1)) = 0.4
input(Array(2)) = 0.5
input(Array(3)) = 0.6
val target = Tensor(3)
target(Array(1)) = 0
target(Array(2)) = 1
target(Array(3)) = 1

> criterion.forward(input, target)
res0: Double = 0.608193395686766
```

**Python example:**
```python
criterion = MultiLabelSoftMarginCriterion()
input = np.array([0.4, 0.5, 0.6])
target = np.array([0, 1, 1])

> criterion.forward(input, target)
0.6081934
```
