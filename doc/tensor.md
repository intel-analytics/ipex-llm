<a name="sparkdl.Tensor.dok"></a>
# Tensor #

__Tensors of different types__

Actually, several types of `Tensor` exists:

```scala
DenseTensor
SparseTensor
```

The data type of Tensor are implemented _only_ for `Float` and `Double`.

<a name="sparkdl.Tensor"></a>
## Tensor constructors ##

Tensor constructors, create new Tensor object, optionally, allocating new memory. By default the elements of a newly allocated memory are
not initialized, therefore, might contain arbitrary numbers. Here are several ways to construct a new `Tensor`.

Please be attention to that the data type of Tensor are implemented _only_ for `Float` and `Double`. So the following T can only be Double or Float

<a name="sparkdl.Tensor"></a>
### Tensor[T] () ###

Returns an empty tensor.

```scala
import com.intel.analytics.sparkdl.tensor
var y = Tensor[T]()
```

<a name="sparkdl.Tensor"></a>
### Tensor[T] (d1 : Int [,d2 : Int [,d3 : Int [,d4 : Int [,d5 : Int ]]]]]) ###

Create a tensor up to 5 dimensions. The tensor size will be `d1 x d2 x d3 x d4 x d5`.

<a name="sparkdl.Tensor"></a>
### Tensor[T] (dims : Int *) ###

<a name="sparkdl.Tensor"></a>
### Tensor[T] (sizes : Array[Int]) ###

<a name="sparkdl.Tensor"></a>
### Tensor[T] (storage : Storage[T]) ###

<a name="sparkdl.Tensor"></a>
### Tensor[T] (storage : Storage[T], storageOffset : Int, size : Array[Int] = null, stride : Array[Int] = null) ###

Convenience constructor (for the previous constructor) assuming a number of dimensions.

<a name="sparkdl.Tensor"></a>
### Tensor[T] (tensor : Tensor[T]) ###

Returns a new tensor which reference the same
[Storage](#storage) than the given `tensor`. The
[size](#Tensor.size), [stride](#Tensor.stride), and
[storage offset](#storageOffset) are the same than the
given tensor.

The new `Tensor` is now going to "view" the same [storage](storage.md)
as the given `tensor`. As a result, any modification in the elements
of the `Tensor` will have a impact on the elements of the given
`tensor`, and vice-versa. No memory copy!

```scala
x = Tensor[T](2,5)  // DenseTensor of dimension 2x5
y = Tensor[T](x)  // y is same as x
```

<a name="sparkdl.Tensor"></a>
### Tensor[T](vector : DenseVector[T]) ###

<a name="sparkdl.Tensor"></a>
### Tensor(vector : DenseVector) ###

<a name="sparkdl.Tensor"></a>
### Tensor[T](vector : DenseMatrix[T]) ###

<a name="sparkdl.Tensor"></a>
### Tensor(vector : DenseMatrix) ###

<a name="sparkdl.Tensor"></a>
### Tensor[T](indices : Array[Array[Int]], values : Storage[T], shape : Array[Int]) ###

<a name="sparkdl.Tensor"></a>
### Tensor[T](rowIndices : Array[Int], columns: Array[Int],values : Storage[T], shape : Array[Int]) ###

## A note on function calls ##

The rest of this guide will present many functions that can be used to manipulate tensors.

<a name="sparkdl.Tensor.ndimension"></a>
###  nDimension() : Int ###

Return the dimension number of the tensor. For empty tensor, its dimension number is 0.

<a name="sparkdl.Tensor.dim"></a>
###  dim() : Int ###

A shortcut of nDimension().

<a name="sparkdl.Tensor.size"></a>
###  size() : Array[Int] ###

Return the size of tensor. Return an array of which each value represent the size on the dimension(i + 1), i is the index of the corresponding value. It will generate a new array each time you invoke the method.

<a name="sparkdl.Tensor.size"></a>
###  size(dim : Int) : Int ###

Return the size of the tensor on the given dimension.

<a name="sparkdl.Tensor.stride"></a>
###  stride() : Array[Int] ###

Jumps between element on the each dimension in the storage. It will generate a new array each time you invoke the method.

<a name="sparkdl.Tensor.stride"></a>
###  stride(dim : Int) : Int ###

Jumps between element on the given dimension in the storage.

<a name="sparkdl.Tensor.fill"></a>
###  fill(v : T) : Tensor[T] ###

Fill the tensor with a given value. It will change the value of the current tensor and return itself.

<a name="sparkdl.Tensor.zero"></a>
###  zero() : Tensor[T] ###

Fill the tensor with zero. It will change the value of the current tensor and return itself.

<a name="sparkdl.Tensor.randn"></a>
###  randn() : Tensor[T] ###

Fill the tensor with random value(normal gaussian distribution). It will change the value of the current tensor and return itself.

<a name="sparkdl.Tensor.rand"></a>
###  rand() : Tensor[T] ###

Fill the tensor with random value(uniform distribution). It will change the value of the current tensor and return itself.

<a name="sparkdl.Tensor.bernoulli"></a>
###  bernoulli(p : Double) : Tensor[T] ###

Fill with random value(bernoulli distribution). It will change the value of the current tensor and return itself.

<a name="sparkdl.Tensor.transpose"></a>
###  transpose(dim1 : Int, dim2 : Int) : Tensor[T] ###

Create a new tensor which exchanges the given dimensions of the current tensor.

<a name="sparkdl.Tensor.t"></a>
###  t() : Tensor[T] ###

Shortcut of transpose(1, 2) for 2D tensor.

<a name="sparkdl.Tensor.valueAt"></a>
###  valueAt(d1 : Int [,d2 : Int [,d3 : Int [,d4 : Int [,d5 : Int ]]]]) : T ###

Query the value on a given position. Tensor should not be empty.

<a name="sparkdl.Tensor.apply"></a>
###  apply(index: Int) : Tensor[T] ###

Query tensor on a given index. Tensor should not be empty.

<a name="sparkdl.Tensor.apply"></a>
###  apply(indexes: Array[Int])) : Tensor[T] ###

Query the value on a given index. Tensor should not be empty.

<a name="sparkdl.Tensor.apply"></a>
###  apply(t: Table)) : Tensor[T] ###

Subset the tensor by apply the element of the given table to corresponding dimension of the tensor. The element of the given table can be an Int or another Table. An Int means select on current dimension; A table means narrow on current dimension, the table should has two elements, of which the first is start index and the second is the end index. An empty table is equals to Table(1, size_of_current_dimension) If the table length is less than the tensor dimension, the missing dimension is applied by an empty table

<a name="sparkdl.Tensor.setValue"></a>
###  setValue(d1 : Int [,d2 : Int [,d3 : Int [,d4 : Int [,d5 : Int ]]]]): Unit ###

Write the value on a given position.

<a name="sparkdl.Tensor.update"></a>
###  update(index: Int, value: T): Unit ###

For tensor(i) = value. If tensor(i) is another tensor, it will fill the selected subset by the given value.

###  update(index: Int, src: Tensor[T]): Unit ###

Copy the give tensor value to the select subset of the current tensor by the given index. The subset should has the same size of the given tensor.

###  update(indexes: Array[Int], value: T): Unit ###

Write the value to the value indexed by the given index array.

###  update(t: Table, value: T): Unit ###

Fill the select subset of the current tensor with the given value. The element of the given table can be an Int or another Table. An Int means select on current dimension; A tablemeans narrow on current dimension, the table should has two elements, of which the first is start index and the second is the end index. An empty table is equals to Table(1, size_of_current_dimension) If the table length is less than the tensor dimension, the missing dimension is applied by an empty table.

###  update(t: Table, src: Tensor[T]): Unit ###

Copy the given tensor value to the select subset of the current tensor The element of the given table can be an Int or another Table. An Int means select on current dimension; A table means narrow on current dimension, the table should has two elements, of which the first is start index and the second is the end index. An empty table is equals to Table(1, size_of_current_dimension) If the table length is less than the tensor dimension, the missing dimension is applied by an empty table.

###  update(filter: T => Boolean, value: T): Unit ###

Update the value meeting the filter criteria with the give value.

<a name="sparkdl.Tensor.isContiguous"></a>
###  isContiguous(): Boolean ###