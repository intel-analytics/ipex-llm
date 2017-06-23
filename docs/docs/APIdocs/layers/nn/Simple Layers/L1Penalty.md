## L1Penalty ##

**Scala:**
```scala
val l1Penalty = L1Penalty(l1weight, sizeAverage, provideOutput)
```
**Python:**
```python
l1Penalty = L1Penalty(l1weight, size_average, provide_output)
```
L1Penalty adds an L1 penalty to an input 
For forward, the output is the same as input and a L1 loss of the latent state will be calculated each time
For backward, gradInput = gradOutput + gradLoss


**Scala example:**
```scala
val l1Penalty = L1Penalty(1, true, true)

> print(l1Penalty.forward(Tensor(3, 3).rand()))
0.782725	0.5012295	0.7882566	
0.76761246	0.9085081	0.7406898	
0.62426275	0.9409664	0.3315808	

```

**Python example:**
```python
l1Penalty = L1Penalty(1, True, True)

>>> l1Penalty.forward(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
[array([[ 1.,  2.,  3.],
       [ 4.,  5.,  6.],
       [ 7.,  8.,  9.]], dtype=float32)]

```

