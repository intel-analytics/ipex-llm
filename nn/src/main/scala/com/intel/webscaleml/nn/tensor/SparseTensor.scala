package com.intel.webscaleml.nn.tensor

import breeze.linalg.{DenseMatrix, DenseVector}
import com.intel.webscaleml.nn.tensor.TensorType.DataType
import org.apache.spark.mllib.linalg.{Matrix, Vector}

import scala.reflect.ClassTag


//indices count from 1
class SparseTensor [@specialized(Float, Double) T: ClassTag](
   var _indices : Array[Array[Int]],
   var _values : Storage[T],
   var _shape : Array[Int]
  )
  extends Tensor[T] {

  //indices order, count from 0
  var indices_order = Array.range(0, _shape.length)

  require(_shape.length == _indices.length, s"indices' size doesn't match tensor shape")

  require(_values.length == _indices(0).length, s"indices' size doesn't match tensor values")

  var nDimension = _shape.length

  /**
   * A shortcut of nDimension()
 *
   * @see nDimension()
   */
  override def dim(): Int = nDimension

  /**
   * Get a contiguous tensor from current tensor
 *
   * @return the current tensor if it's contiguous; or a new contiguous tensor with separated storage
   */
  override def contiguous(): Tensor[T] = {
    this
  }

  override def getType(): DataType = throw new UnsupportedOperationException(s"Unimplemented")

  override def setValue(d1: Int, value: T): SparseTensor.this.type = {
    require(1 == this.nDimension, "invalid size")
    throw new UnsupportedOperationException(s"Unimplemented")
    this
  }

  override def setValue(d1: Int, d2: Int, value: T): SparseTensor.this.type = {
    require(2 == this.nDimension, "invalid size")
    throw new UnsupportedOperationException(s"Unimplemented")
    this
  }

  override def setValue(d1: Int, d2: Int, d3: Int, value: T): SparseTensor.this.type = {
    require(3 == this.nDimension, "invalid size")
    throw new UnsupportedOperationException(s"Unimplemented")
    this
  }

  override def setValue(d1: Int, d2: Int, d3: Int, d4: Int, value: T): SparseTensor.this.type = {
    require(4 == this.nDimension, "invalid size")
    throw new UnsupportedOperationException(s"Unimplemented")
    this
  }

  override def setValue(d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, value: T): SparseTensor.this.type = {
    require(5 == this.nDimension, "invalid size")
    throw new UnsupportedOperationException(s"Unimplemented")
    this
  }

  override def unfold(dim: Int, size: Int, step: Int): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
    this
  }

  /**
   * Element number
 *
   * @return element number
   */
  override def nElement(): Int = _values.length

  /**
   * Resize the current tensor to the give shape
 *
   * @param sizes Array describe the size
   * @param strides Array describe the jumps
   * @return
   */
  override def resize(sizes: Array[Int], strides: Array[Int]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
    this
  }

  /**
   * Fill with random value(normal gaussian distribution).
   * It will change the value of the current tensor and return itself
 *
   * @return current tensor
   */
  override def randn(): Tensor[T] = {
    throw new UnsupportedOperationException(s"Unimplemented")
    this
  }

  /**
   * For tensor(i) = value. If tensor(i) is another tensor, it will fill the selected subset by the given value
 *
   * @param index index
   * @param value value to write
   */
  override def update(index: Int, value: T): Unit = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  /**
   * Copy the give tensor value to the select subset of the current tensor by the given index. The subset should
   * has the same size of the given tensor
 *
   * @param index index
   * @param src tensor to write
   */
  override def update(index: Int, src: Tensor[T]): Unit = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  /**
   * Write the value to the value indexed by the given index array
 *
   * @param indexes index array. It should has same length with the tensor dimension
   * @param value value to write
   */
  override def update(indexes: Array[Int], value: T): Unit = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  /**
   * Fill the select subset of the current tensor with the given value.
   * The element of the given table can be an Int or another Table. An Int means select on current dimension; A table
   * means narrow on current dimension, the table should has two elements, of which the first is start index and
   * the second is the end index. An empty table is equals to Table(1, size_of_current_dimension)
   * If the table length is less than the tensor dimension, the missing dimension is applied by an empty table
 *
   * @param t subset table
   * @param value value to write
   */
  override def update(t: Table, value: T): Unit = {
    throw new UnsupportedOperationException(s"Unimplemented")
  }

  /**
   * Copy the given tensor value to the select subset of the current tensor
   * The element of the given table can be an Int or another Table. An Int means select on current dimension; A table
   * means narrow on current dimension, the table should has two elements, of which the first is start index and
   * the second is the end index. An empty table is equals to Table(1, size_of_current_dimension)
   * If the table length is less than the tensor dimension, the missing dimension is applied by an empty table
 *
   * @param t subset table
   * @param src tensor to copy
   */
  override def update(t: Table, src: Tensor[T]): Unit = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * Update the value meeting the filter criteria with the give value
 *
   * @param filter filter
   * @param value value to update
   */
  override def update(filter: (T) => Boolean, value: T): Unit = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * Apply a function to each element of the tensor and modified it value if it return a double
 *
   * @param func applied function
   * @return current tensor
   */
  override def apply1(func: (T) => T): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * The Tensor is now going to "view" the same storage as the given tensor. As the result, any modification in the
   * elements of the Tensor will have an impact on the elements of the given tensor, and vice-versa. This is an
   * efficient method, as there is no memory copy!
 *
   * @param other the given tensor
   * @return current tensor
   */
  override def set(other: Tensor[T]): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * The Tensor is now going to "view" the given storage, starting at position storageOffset (>=1) with the given
   * dimension sizes and the optional given strides. As the result, any modification in the elements of the Storage
   * will have a impact on the elements of the Tensor, and vice-versa. This is an efficient method,
   * as there is no memory copy!
   *
   * If only storage is provided, the whole storage will be viewed as a 1D Tensor.
 *
   * @param storage
   * @param storageOffset
   * @param sizes
   * @param strides
   * @return current tensor
   */
  override def set(storage: Storage[T], storageOffset: Int, sizes: Array[Int], strides: Array[Int]): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * Resize the current tensor to the same size of the given tensor. It will still use the same storage if the storage
   * is sufficient for the new size
 *
   * @param src target tensor
   * @return current tensor
   */
  override def resizeAs(src: Tensor[_]): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * Shortcut of transpose(1, 2) for 2D tensor
 *
   * @see transpose()
   */
  override def t(): Tensor[T] = {
    transpose(1, 2)
  }

  override def expandAs(template: Tensor[T]): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * Removes all singleton dimensions of the tensor
 *
   * @return current tensor
   */
  override def squeeze(): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * Removes given dimensions of the tensor if it's singleton
 *
   * @return current tensor
   */
  override def squeeze(dim: Int): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * Check if the size is same with the give tensor
 *
   * @param other tensor to be compared
   * @return true if they have same size
   */
  override def isSameSizeAs(other: Tensor[_]): Boolean = throw new UnsupportedOperationException(s"Unimplemented")

  override def toMLlibMatrix(): Matrix = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * Size of tensor. Return an array of which each value represent the size on the dimension(i + 1), i is the
   * index of the corresponding value
   * It will generate a new array each time you invoke the method
 *
   * @return size array
   */
  override def size(): Array[Int] = _shape

  /**
   * size of the tensor on the given dimension
 *
   * @param dim dimension, count from 1
   * @return size
   */
  override def size(dim: Int): Int = _shape(indices_order(dim - 1))

  /**
   * Fill with random value(uniform distribution).
   * It will change the value of the current tensor and return itself
 *
   * @return current tensor
   */
  override def rand(): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  /** *
    * Create a new tensor which exchanges the given dimensions of the current tensor
 *
    * @param dim1 dimension to be exchanged, count from one
    * @param dim2 dimension to be exchanged, count from one
    * @return new tensor
    */
  override def transpose(dim1: Int, dim2: Int): Tensor[T] = {
    require(2 == this.nDimension, "invalid size")
    val result = new SparseTensor(_indices, _values, _shape)
    result.indices_order(dim1-1) = indices_order(dim2-1)
    result.indices_order(dim2-1) = indices_order(dim1-1)
    result
  }

  /**
   * Copy the value of the given tensor to the current. They should have same size. It will use the old storage
 *
   * @param other source tensor
   * @return current tensor
   */
  override def copy(other: Tensor[T]): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * Splits current tensor along dimension dim into a result table of Tensors of size size (a number) or less
   * (in the case of the last Tensor). The sizes of the non-dim dimensions remain unchanged.
   * Internally, a series of narrows are performed along dimensions dim. Argument dim defaults to 1.
   */
  override def split(size: Int, dim: Int): Array[Tensor[T]] = throw new UnsupportedOperationException(s"Unimplemented")

  override def toBreezeMatrix(): DenseMatrix[T] = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * Remove the dim-th dimension and return the subset part. For instance
   * tensor =
   * 1 2 3
   * 4 5 6
   * tensor.select(1, 1) is [1 2 3]
   * tensor.select(1, 2) is [4 5 6]
   * tensor.select(2, 3) is [3 6]
 *
   * @param dim
   * @param index
   * @return
   */
  override def select(dim: Int, index: Int): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * Fill with random value(bernoulli distribution).
   * It will change the value of the current tensor and return itself
 *
   * @return current tensor
   */
  override def bernoulli(p: Double): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  override def view(sizes: Array[Int]): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * Query tensor on a given index. Tensor should not be empty
 *
   * @param index count from 1
   * @return
   */
  override def apply(index: Int): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * Query the value on a given index. Tensor should not be empty
 *
   * @param indexes the indexes length should be same as the tensor dimension length and each value count from 1
   * @return the value on the given index
   */
  override def apply(indexes: Array[Int]): T = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * Subset the tensor by apply the element of the given table to corresponding dimension of the tensor
   * The element of the given table can be an Int or another Table. An Int means select on current dimension; A table
   * means narrow on current dimension, the table should has two elements, of which the first is start index and
   * the second is the end index. An empty table is equals to Table(1, size_of_current_dimension)
   * If the table length is less than the tensor dimension, the missing dimension is applied by an empty table
 *
   * @see select
   * @see narrow
   * @param t The table length should be less than or equal to the tensor dimensions
   * @return
   */
  override def apply(t: Table): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * Repeating a tensor allocates new memory, unless result is provided, in which case its memory is resized.
   * sizes specify the number of times the tensor is repeated in each dimension.
 *
   * @param sizes
   * @return
   */
  override def repeatTensor(sizes: Array[Int]): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  override def expand(sizes: Array[Int]): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * Jumps between element on the each dimension in the storage.
   * It will generate a new array each time you invoke the method
 *
   * @return strides array
   */
  override def stride(): Array[Int] = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * Jumps between element on the given dimension in the storage.
 *
   * @param dim dimension, count from 1
   * @return jump
   */
  override def stride(dim: Int): Int = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * tensor offset on the storage
 *
   * @return storage offset, count from 1
   */
  override def storageOffset(): Int = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * Check if the tensor is contiguous on the storage
 *
   * @return true if it's contiguous
   */
  override def isContiguous(): Boolean = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * Get a subset of the tensor on dim-th dimension. The offset is given by index, and length is give by size
   * The important difference with select is that it will not reduce the dimension number. For Instance
   * tensor =
   * 1 2 3
   * 4 5 6
   * tensor.narrow(1, 1, 1) is [1 2 3]
   * tensor.narrow(2, 2, 3) is
   * 2 3
   * 5 6
 *
   * @param dim
   * @param index
   * @param size
   * @return
   */
  override def narrow(dim: Int, index: Int, size: Int): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * Fill with a given value. It will change the value of the current tensor and return itself
 *
   * @param v value to fill the tensor
   * @return current tensor
   */
  override def fill(v: T): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  override def toBreezeVector(): DenseVector[T] = throw new UnsupportedOperationException(s"Unimplemented")

  override def valueAt(d1: Int): T = throw new UnsupportedOperationException(s"Unimplemented")

  override def valueAt(d1: Int, d2: Int): T = throw new UnsupportedOperationException(s"Unimplemented")

  override def valueAt(d1: Int, d2: Int, d3: Int): T = throw new UnsupportedOperationException(s"Unimplemented")

  override def valueAt(d1: Int, d2: Int, d3: Int, d4: Int): T = throw new UnsupportedOperationException(s"Unimplemented")

  override def valueAt(d1: Int, d2: Int, d3: Int, d4: Int, d5: Int): T = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * Map value of another tensor to corresponding value of current tensor and apply function on the two value and
   * change the value of the current tensor
   * The another tensor should has the same size of the current tensor
 *
   * @param other another tensor
   * @param func applied funciton
   * @return current tensor
   */
  override def map(other: Tensor[T], func: (T, T) => T): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  override def toMLlibVector(): Vector = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * Get the storage
 *
   * @return storage
   */
  override def storage(): Storage[T] = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * Fill with zero. It will change the value of the current tensor and return itself
 *
   * @return current tensor
   */
  override def zero(): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * divide all elements of this with value in-place.
 *
   * @param value
   * @return
   */
  override def div(value: T): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * Performs the outer-product between vec1 (1D tensor) and vec2 (1D tensor).
   * Optional values v1 and v2 are scalars that multiply mat and vec1 [out] vec2 respectively.
   * In other words,
   * res_ij = (v1 * mat_ij) + (v2 * vec1_i * vec2_j)
 *
   * @param t1
   * @param t2
   * @return
   */
  override def addr(t1: Tensor[T], t2: Tensor[T]): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  override def addr(v1: T, t1: Tensor[T], t2: Tensor[T]): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  override def addr(v1: T, t1: Tensor[T], v2: T, t2: Tensor[T]): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  override def mean(): T = throw new UnsupportedOperationException(s"Unimplemented")

  override def mean(dim: Int): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * multiply all elements of this with value in-place.
 *
   * @param value
   * @return
   */
  override def mul(value: T): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * put the result of x * value in current tensor
 *
   * @param value
   * @return
   */
  override def mul(x: Tensor[T], value: T): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  override def unary_-(): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  override def /(s: T): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  override def /(t: Tensor[T]): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  override def max(): T = throw new UnsupportedOperationException(s"Unimplemented")

  override def max(dim: Int): (Tensor[T], Tensor[T]) = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * Performs a matrix-vector multiplication between mat (2D Tensor) and vec2 (1D Tensor) and add it to vec1.
   * Optional values v1 and v2 are scalars that multiply vec1 and vec2 respectively.
   *
   * In other words,
   * res = (beta * vec1) + alpha * (mat * vec2)
   *
   * Sizes must respect the matrix-multiplication operation: if mat is a n × m matrix, vec2 must be vector of
   * size m and vec1 must be a vector of size n.
   */
  override def addmv(beta: T, vec1: Tensor[T], alpha: T, mat: Tensor[T], vec2: Tensor[T]): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  // res = beta * res + alpha * (mat * vec2)
  override def addmv(beta: T, alpha: T, mat: Tensor[T], vec2: Tensor[T]): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  // res = res + alpha * (mat * vec2)
  override def addmv(alpha: T, mat: Tensor[T], vec2: Tensor[T]): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * return pseudo-random numbers, require 0<=args.length<=2
   * if args.length = 0, return [0, 1)
   * if args.length = 1, return [1, args(0)] or [args(0), 1]
   * if args.length = 2, return [args(0), args(1)]
   *
   * @param args
   */
  override def uniform(args: T*): T = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * Performs a matrix-matrix multiplication between mat1 (2D tensor) and mat2 (2D tensor).
   * Optional values v1 and v2 are scalars that multiply M and mat1 * mat2 respectively.
   * Optional value beta is a scalar that scales the result tensor, before accumulating the result into the tensor.
   * Defaults to 1.0.
   * If mat1 is a n x m matrix, mat2 a m x p matrix, M must be a n x p matrix.
   *
   * res = (v1 * M) + (v2 * mat1*mat2)
 *
   * @param v1
   * @param M
   * @param v2
   * @param mat1
   * @param mat2
   */
  override def addmm(v1: T, M: Tensor[T], v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  // res = M + (mat1*mat2)
  override def addmm(M: Tensor[T], mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  // res = res + mat1 * mat2
  override def addmm(mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  // res = res + v2 * mat1 * mat2
  override def addmm(v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  // res = v1 * res + v2 * mat1*mat2
  override def addmm(v1: T, v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  override def conv2(kernel: Tensor[T], vf: Char): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  override def +(s: T): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  override def +(t: Tensor[T]): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * Performs the dot product. The number of elements must match: both Tensors are seen as a 1D vector.
 *
   * @param y
   * @return
   */
  override def dot(y: Tensor[T]): T = throw new UnsupportedOperationException(s"Unimplemented")

  override def xcorr2(kernel: Tensor[T], vf: Char): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  override def abs(): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  override def sqrt(): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  override def sum(): T = throw new UnsupportedOperationException(s"Unimplemented")

  override def sum(dim: Int): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * Performs the element-wise multiplication of tensor1 by tensor2, multiply the result by the scalar value
   * (1 if not present) and add it to x. The number of elements must match, but sizes do not matter.
   *
   * @param value
   * @param tensor1
   * @param tensor2
   */
  override def addcmul(value: T, tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  override def addcmul(tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  override def -(s: T): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  override def -(t: Tensor[T]): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * Performs the element-wise division of tensor1 by tensor2, multiply the result by the scalar value and add it to x.
   * The number of elements must match, but sizes do not matter.
   *
   * @param value
   * @param tensor1
   * @param tensor2
   * @return
   */
  override def addcdiv(value: T, tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * y.cmul(x) multiplies all elements of y with corresponding elements of x.
   *
   * @param y other tensor
   * @return current tensor
   */
  override def cmul(y: Tensor[T]): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  override def *(s: T): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  override def *(t: Tensor[T]): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * x.add(value,y) multiply-accumulates values of y into x.
   *
   * @param value scalar
   * @param y other tensor
   * @return current tensor
   */
  override def add(value: T, y: Tensor[T]): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  // Puts the result of x + value * y in current tensor
  override def add(x: Tensor[T], value: T, y: Tensor[T]): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  override def add(value: T): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  /**
   * accumulates all elements of y into this
 *
   * @param y other tensor
   * @return current tensor
   */
  override def add(y: Tensor[T]): Tensor[T] = throw new UnsupportedOperationException(s"Unimplemented")

  override def diff(other: Tensor[T], count: Int = 2, reverse: Boolean = false): Boolean = throw new UnsupportedOperationException(s"Unimplemented")

  override def resize(size1: Int): Tensor[T] = ???

  override def resize(size1: Int, size2: Int): Tensor[T] = ???

  override def resize(size1: Int, size2: Int, size3: Int): Tensor[T] = ???

  override def resize(size1: Int, size2: Int, size3: Int, size4: Int): Tensor[T] = ???
}
