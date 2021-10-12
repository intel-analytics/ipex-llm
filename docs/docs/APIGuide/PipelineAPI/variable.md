## Basic operators: `+ - * /`
Those are supported as element-wise operation.

**Scala example**
```scala
x + 1.0
x + y
```


**Python example**
```python
x + 1.0
x + y
```

## squeeze
   Delete the singleton dimension(s).
   The batch dimension needs to be unchanged.
   For example, if input has size (2, 1, 3, 4, 1):
   - squeeze(dim = 1) will give output size (2, 3, 4, 1)
   - squeeze(dims = null) will give output size (2, 3, 4)

**Scala example**
```scala
x.squeeze(1)
```


**Python example**
```python
x.squeeze(1)
```

## slice
Slice the input with the number of dimensions not being reduced.
The batch dimension needs to be unchanged.
- dim The dimension to narrow. 0-based index. Cannot narrow the batch dimension.
     -1 means the last dimension of the input.
- startIndex Non-negative integer. The start index on the given dimension. 0-based index.
- length The length to be sliced. Default is 1.

For example, 
if input is:
1 2 3
4 5 6
- slice(1, 1, 2) will give output
2 3
5 6
- slice(1, 2, -1) will give output
3
6
       

**Scala example**
```scala
x.slice(1, 1, 2)
```


**Python example**
```python
x.slice(1, 1, 2)
```

## index_select
 Select an index of the input in the given dim and return the subset part.
 The batch dimension needs to be unchanged.
 The selected dim would be remove after this operation.
 - dim: The dimension to select. 0-based index. Cannot select the batch dimension.
                 -1 means the last dimension of the input.
 - index: The index of the dimension to be selected. 0-based index.
                -1 means the last dimension of the input.
  
 For example, if input is:
           1 2 3
           4 5 6
 - Select(1, 1) will give output [2 5]
 - Select(1, -1) will give output [3 6]
       

**Scala example**
```scala
x.select(1, 1)
```


**Python example**
```python
x.select(1, 1)
```
