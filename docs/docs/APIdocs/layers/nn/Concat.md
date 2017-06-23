## Concat ##

**Scala:**
```scala
val module = Concat(dimension: Int)
```
**Python:**
```python
module = Concat(dimension)
```

Concat is a container who concatenates the output of it's submodules along the
provided `dimension`: all submodules take the same inputs, and their output is
concatenated.
```
                 +----Concat----+
            +---->  submodule1  -----+
            |    |              |    |
 input -----+---->  submodule2  -----+----> output
            |    |              |    |
            +---->  submodule3  -----+
                 +--------------+
```

**Scala example:**
```scala
val mlp = Concat(2)
mlp.add(Linear(3,2))
mlp.add(Linear(3,4))

println(mlp.forward(Tensor(2, 3).rand()))
```
Output is
```
-0.17087375	0.12954286	0.15685591	-0.027277306	0.38549712	-0.20375136
-0.9473443	0.030516684	0.23380546	0.625985	-0.031360716	0.40449825
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x6]
```

**Python example:**
```python
mlp = Concat(2)
mlp.add(Linear(3,2))
mlp.add(Linear(3,4))
print(mlp.forward(np.array([[1, 2, 3], [1, 2, 3]])))
```
Output is
```
[array([
[-0.71994132,  2.17439198, -1.46522939,  0.64588934,  2.61534023, -2.39528942],
[-0.89125222,  5.49583197, -2.8865242 ,  1.44914722,  5.26639175, -6.26586771]]
      dtype=float32)]

```
