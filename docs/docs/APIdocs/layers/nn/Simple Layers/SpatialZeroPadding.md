## SpatialZeroPadding ##

**Scala:**
```scala
val spatialZeroPadding = SpatialZeroPadding(padLeft, padRight, padTop, padBottom)
```
**Python:**
```python
spatialZeroPadding = SpatialZeroPadding(pad_left, pad_right, pad_top, pad_bottom)
```

Each feature map of a given input is padded with specified number of zeros.
 
If padding values are negative, then input will be cropped.

**Scala example:**
```scala
val spatialZeroPadding = SpatialZeroPadding(1, 0, -1, 0)

> print(spatialZeroPadding.forward(Tensor(3, 3, 3).rand()))
(1,.,.) =
0.0	0.3134808731265366	0.5005130991339684	0.7760939800646156	
0.0	0.3250664414372295	0.5973542677238584	0.14889140450395644	

(2,.,.) =
0.0	0.69818788417615	0.4600889780558646	0.534300301456824	
0.0	0.7801548321731389	0.42356246523559093	0.5523960464634001	

(3,.,.) =
0.0	0.3300329972989857	0.06369128986261785	0.17322053853422403	
0.0	0.6164324851706624	0.5436311133671552	0.3554064442869276	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x2x4]


```

**Python example:**
```python
spatialZeroPadding = SpatialZeroPadding(1, 0, -1, 0)
> spatialZeroPadding.forward(np.array([[[1, 2],[3, 4]],[[1, 2],[3, 4]]]))
[array([[[ 0.,  3.,  4.]],
       [[ 0.,  3.,  4.]]], dtype=float32)]
       
```


