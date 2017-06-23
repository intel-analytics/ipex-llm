## Reverse ##

 * Reverse the input w.r.t given dimension.
 * The input can be a Tensor or Table.
 * NOTE: __dimension is one-based index__ 

**Scala:**
```scala
Reverse[T](dim: Int = 1, isInplace: Boolean = false)
```
**Python:**
```python
Reverse(dimension=1, bigdl_type="float")
```


**Scala example:**
```scala
import com.intel.analytics.bigdl.utils._

def randomn(): Double = RandomGenerator.RNG.uniform(0, 1)
val input = Tensor[Double](2, 3)
input.apply1(x => randomn())

val layer = new Reverse[Double](1)
print(input)
print(layer.forward(input))
```

```
Output is:

0.05967033724300563	0.9228034485131502	0.13365511689335108	
0.2187003034632653	0.4040174684487283	0.9039117493666708

0.2187003034632653	0.4040174684487283	0.9039117493666708	
0.05967033724300563	0.9228034485131502	0.13365511689335108
```



**Python example:**
```python
input = np.random.random((2,3))
layer = Reverse(1)
print(input)
print(layer.forward(input))
```
```
Output is:
[[ 0.6747489   0.10076833  0.62421962]
 [ 0.39737526  0.25602485  0.99709782]]
 
[ 0.39737526,  0.25602484,  0.99709785],
[ 0.6747489 ,  0.10076834,  0.6242196 ]]
 
```

