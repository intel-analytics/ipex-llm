/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.tensor

import breeze.linalg.{DenseMatrix => BrzDenseMatrix, DenseVector => BrzDenseVector}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector, Matrix, Vector}
import scala.reflect.ClassTag

abstract class QuantizedTensorUnsupported[T: ClassTag] extends Tensor[T] {
  val errorString = s"QuantizeTensor doesn't support this operation now"

  /**
   * Fill with a given value. It will change the value of the current tensor and return itself
   *
   * @param v value to fill the tensor
   * @return current tensor
   */
  override def fill(v: T): Tensor[T] = throw new UnsupportedOperationException(errorString)

  /**
   * Fill with zero. It will change the value of the current tensor and return itself
   *
   * @return current tensor
   */
  override def zero(): Tensor[T] = throw new UnsupportedOperationException(errorString)

  /**
   * Fill with random value(normal gaussian distribution).
   * It will change the value of the current tensor and return itself
   *
   * @return current tensor
   */
  override def randn(): Tensor[T] = throw new UnsupportedOperationException(errorString)

  /**
   * Fill with random value(normal gaussian distribution with the specified mean
   * and stdv).
   * It will change the value of the current tensor and return itself
   *
   * @return current tensor
   */
  override def randn(mean: Double, stdv: Double): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Fill with random value(uniform distribution).
   * It will change the value of the current tensor and return itself
   *
   * @return current tensor
   */
  override def rand(): Tensor[T] = throw new UnsupportedOperationException(errorString)

  /**
   * Fill with random value(uniform distribution between [lowerBound, upperBound])
   * It will change the value of the current tensor and return itself
   *
   * @return current tensor
   */
  override def rand(lowerBound: Double, upperBound: Double): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Fill with random value(bernoulli distribution).
   * It will change the value of the current tensor and return itself
   *
   * @return current tensor
   */
  override def bernoulli(p: Double): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /** *
   * Create a new tensor which exchanges the given dimensions of the current tensor
   *
   * @param dim1 dimension to be exchanged, count from one
   * @param dim2 dimension to be exchanged, count from one
   * @return new tensor
   */
  override def transpose(dim1: Int, dim2: Int): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Shortcut of transpose(1, 2) for 2D tensor
   *
   * @see transpose()
   */
  override def t(): Tensor[T] = throw new UnsupportedOperationException(errorString)

  /**
   * Query tensor on a given index. Tensor should not be empty
   *
   * @param index count from 1
   * @return
   */
  override def apply(index: Int): Tensor[T] = throw new UnsupportedOperationException(errorString)

  /**
   * Query the value on a given index. Tensor should not be empty
   *
   * @param indexes the indexes length should be same as the tensor dimension length and each
   *                value count from 1
   * @return the value on the given index
   */
  override def apply(indexes: Array[Int]): T = throw new UnsupportedOperationException(errorString)

  /**
   * Query the value on a given position. The number of parameters
   * should be equal to the dimension number of the tensor.
   * Tensor should not be empty.
   *
   * @param d1 ,( d2, d3, d4, d5) the given position
   * @return the value on a given position
   */
  override def valueAt(d1: Int): T = throw new UnsupportedOperationException(errorString)

  override def valueAt(d1: Int, d2: Int): T = throw new UnsupportedOperationException(errorString)

  override def valueAt(d1: Int, d2: Int, d3: Int): T =
    throw new UnsupportedOperationException(errorString)

  override def valueAt(d1: Int, d2: Int, d3: Int, d4: Int): T =
    throw new UnsupportedOperationException(errorString)

  override def valueAt(d1: Int, d2: Int, d3: Int, d4: Int, d5: Int): T =
    throw new UnsupportedOperationException(errorString)

  /**
   * Subset the tensor by apply the element of the given table to corresponding dimension of the
   * tensor. The element of the given table can be an Int or another Table.
   * An Int means select on current dimension; A table means narrow on current dimension,
   * the table should has two elements, of which the first is start index and
   * the second is the end index. An empty table is equals to Table(1, size_of_current_dimension)
   * If the table length is less than the tensor dimension, the missing dimension is applied by
   * an empty table
   *
   * @see select
   * @see narrow
   * @param t The table length should be less than or equal to the tensor dimensions
   * @return
   */
  override def apply(t: Table): Tensor[T] = throw new UnsupportedOperationException(errorString)

  /**
   * For tensor(i) = value. If tensor(i) is another tensor, it will fill the selected subset by
   * the given value
   *
   * @param index index
   * @param value value to write
   */
  override def update(index: Int, value: T): Unit =
    throw new UnsupportedOperationException(errorString)

  /**
   * Copy the give tensor value to the select subset of the current tensor by the given index.
   * The subset should
   * has the same size of the given tensor
   *
   * @param index index
   * @param src   tensor to write
   */
  override def update(index: Int, src: Tensor[T]): Unit =
    throw new UnsupportedOperationException(errorString)

  /**
   * Write the value to the value indexed by the given index array
   *
   * @param indexes index array. It should has same length with the tensor dimension
   * @param value   value to write
   */
  override def update(indexes: Array[Int], value: T): Unit =
    throw new UnsupportedOperationException(errorString)

  /**
   * Write the value on a given position. The number of parameters
   * should be equal to the dimension number of the tensor.
   *
   * @param d1    ,( d2, d3, d4, d5) the given position
   * @param value the written value
   * @return
   */
  override def setValue(d1: Int, value: T): this.type =
    throw new UnsupportedOperationException(errorString)

  override def setValue(d1: Int, d2: Int, value: T): this.type =
    throw new UnsupportedOperationException(errorString)

  override def setValue(d1: Int, d2: Int, d3: Int, value: T): this.type =
    throw new UnsupportedOperationException(errorString)

  override def setValue(d1: Int, d2: Int, d3: Int, d4: Int, value: T): this.type =
    throw new UnsupportedOperationException(errorString)

  override def setValue(d1: Int, d2: Int, d3: Int, d4: Int, d5: Int,
    value: T): this.type = throw new UnsupportedOperationException(errorString)

  /**
   * Fill the select subset of the current tensor with the given value.
   * The element of the given table can be an Int or another Table. An Int means select on current
   * dimension; A tablemeans narrow on current dimension, the table should has two elements,
   * of which the first is start index and the second is the end index. An empty table is equals
   * to Table(1, size_of_current_dimension) If the table length is less than the tensor dimension,
   * the missing dimension is applied by an empty table
   *
   * @param t     subset table
   * @param value value to write
   */
  override def update(t: Table, value: T): Unit =
    throw new UnsupportedOperationException(errorString)

  /**
   * Copy the given tensor value to the select subset of the current tensor
   * The element of the given table can be an Int or another Table. An Int means select on current
   * dimension; A table means narrow on current dimension, the table should has two elements,
   * of which the first is start index and the second is the end index. An empty table is equals
   * to Table(1, size_of_current_dimension) If the table length is less than the tensor dimension,
   * the missing dimension is applied by an empty table
   *
   * @param t   subset table
   * @param src tensor to copy
   */
  override def update(t: Table, src: Tensor[T]): Unit =
    throw new UnsupportedOperationException(errorString)

  /**
   * Update the value meeting the filter criteria with the give value
   *
   * @param filter filter
   * @param value  value to update
   */
  override def update(filter: (T) => Boolean, value: T): Unit =
    throw new UnsupportedOperationException(errorString)

  /**
   * Check if the tensor is contiguous on the storage
   *
   * @return true if it's contiguous
   */
  override def isContiguous(): Boolean = throw new UnsupportedOperationException(errorString)

  /**
   * Get a contiguous tensor from current tensor
   *
   * @return the current tensor if it's contiguous; or a new contiguous tensor with separated
   *         storage
   */
  override def contiguous(): Tensor[T] = throw new UnsupportedOperationException(errorString)

  /**
   * Check if the size is same with the give tensor
   *
   * @param other tensor to be compared
   * @return true if they have same size
   */
  override def isSameSizeAs(other: Tensor[_]): Boolean =
    throw new UnsupportedOperationException(errorString)

  /**
   * Resize the current tensor to the same size of the given tensor. It will still use the same
   * storage if the storage
   * is sufficient for the new size
   *
   * @param src target tensor
   * @return current tensor
   */
  override def resizeAs(src: Tensor[_]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

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
  override def select(dim: Int, index: Int): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Get the storage
   *
   * @return storage
   */
  override def storage(): Storage[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * tensor offset on the storage
   *
   * @return storage offset, count from 1
   */
  override def storageOffset(): Int =
    throw new UnsupportedOperationException(errorString)

  /**
   * The Tensor is now going to "view" the given storage, starting at position storageOffset (>=1)
   * with the given dimension sizes and the optional given strides. As the result, any
   * modification in the elements of the Storage will have an impact on the elements of the Tensor,
   * and vice-versa. This is an efficient method, as there is no memory copy!
   *
   * If only storage is provided, the whole storage will be viewed as a 1D Tensor.
   *
   * @param storage
   * @param storageOffset
   * @param sizes
   * @param strides
   * @return current tensor
   */
  override def set(storage: Storage[T], storageOffset: Int, sizes: Array[Int],
    strides: Array[Int]): Tensor[T] = throw new UnsupportedOperationException(errorString)

  /**
   * Get a subset of the tensor on dim-th dimension. The offset is given by index, and length is
   * give by size. The important difference with select is that it will not reduce the dimension
   * number. For Instance
   * tensor =
   * 1 2 3
   * 4 5 6
   * tensor.narrow(1, 1, 1) is [1 2 3]
   * tensor.narrow(2, 2, 2) is
   * 2 3
   * 5 6
   *
   * @param dim
   * @param index
   * @param size
   * @return
   */
  override def narrow(dim: Int, index: Int, size: Int): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Apply a function to each element of the tensor and modified it value if it return a double
   *
   * @param func applied function
   * @return current tensor
   */
  override def apply1(func: (T) => T): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Map value of another tensor to corresponding value of current tensor and apply function on
   * the two value and change the value of the current tensor
   * The another tensor should has the same size of the current tensor
   *
   * @param other another tensor
   * @param func  applied function
   * @return current tensor
   */
  override def map(other: Tensor[T], func: (T, T) => T): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Removes all singleton dimensions of the tensor
   *
   * @return current tensor
   */
  override def squeeze(): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Removes given dimensions of the tensor if it's singleton
   *
   * @return current tensor
   */
  override def squeeze(dim: Int): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Create a new tensor that removes all singleton dimensions of the tensor
   *
   * @return create a new tensor
   */
  override def squeezeNewTensor(): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  override def view(sizes: Array[Int]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   *
   * Returns a tensor which contains all slices of size @param size
   * in the dimension @param dim. Step between two slices is given by @param step.
   *
   * @param dim
   * @param size
   * @param step Step between two slices
   * @return new tensor
   */
  override def unfold(dim: Int, size: Int, step: Int): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Repeating a tensor allocates new memory, unless result is provided, in which case its memory
   * is resized. sizes specify the number of times the tensor is repeated in each dimension.
   *
   * @param sizes
   * @return
   */
  override def repeatTensor(sizes: Array[Int]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * This is equivalent to this.expand(template.size())
   *
   * @param template the given tensor
   * @return
   */
  override def expandAs(template: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Expanding a tensor allocates new memory, tensor where singleton dimensions can be expanded
   * to multiple ones by setting the stride to 0. Any dimension that has size 1 can be expanded
   * to arbitrary value with new memory allocation. Attempting to expand along a dimension that
   * does not have size 1 will result in an error.
   *
   * @param sizes the size that tensor will expend to
   * @return
   */
  override def expand(sizes: Array[Int]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Splits current tensor along dimension dim into a result table of Tensors of size size
   * (a number) or less (in the case of the last Tensor). The sizes of the non-dim dimensions
   * remain unchanged. Internally, a series of narrows are performed along dimensions dim.
   * Argument dim defaults to 1.
   *
   * @param size
   * @param dim
   * @return
   */
  override def split(size: Int, dim: Int): Array[Tensor[T]] =
    throw new UnsupportedOperationException(errorString)

  /**
   * spilt one tensor into multi tensor along the `dim` dimension
   *
   * @param dim the specific dimension
   * @return
   */
  override def split(dim: Int): Array[Tensor[T]] =
    throw new UnsupportedOperationException(errorString)

  /**
   * convert the tensor to BreezeVector, the dimension of the tensor need to be 1.
   *
   * @return BrzDenseVector
   */
  override def toBreezeVector(): BrzDenseVector[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * convert the tensor to MLlibVector, the dimension of the
   * tensor need to be 1, and tensor need to be continuous.
   *
   * @return Vector
   */
  override def toMLlibVector(): Vector =
    throw new UnsupportedOperationException(errorString)

  /**
   * convert the tensor to BreezeMatrix, the dimension of the tensor need to be 2.
   *
   * @return BrzDenseMatrix
   */
  override def toBreezeMatrix(): BrzDenseMatrix[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * convert the tensor to MLlibMatrix, the dimension of the
   * tensor need to be 2, and tensor need to be continuous.
   *
   * @return Matrix
   */
  override def toMLlibMatrix(): Matrix =
    throw new UnsupportedOperationException(errorString)

  /**
   * return the tensor datatype( DoubleType or FloatType)
   *
   * @return
   */
  override def getType(): TensorDataType =
    throw new UnsupportedOperationException(errorString)

  /**
   * Compare and print differences between two tensors
   *
   * @param other
   * @param count
   * @return true if there's difference, vice versa
   */
  override def diff(other: Tensor[T], count: Int, reverse: Boolean): Boolean =
    throw new UnsupportedOperationException(errorString)

  /**
   * view this.tensor and add a Singleton Dimension to `dim` dimension
   *
   * @param t   source tensor
   * @param dim the specific dimension, default is 1
   * @return this
   */
  override def addSingletonDimension(t: Tensor[T], dim: Int): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * view this.tensor and add multiple Dimensions to `dim` dimension
   *
   * @param t source tensor
   * @param dim the specific dimension array, default is [1]
   * @return this
   */
  override def addMultiDimension(t: Tensor[T], dims: Array[Int]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * create a new tensor without any change of the tensor
   *
   * @param sizes the size of the new Tensor
   * @return
   */
  override def reshape(sizes: Array[Int]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Save the tensor to given path
   *
   * @param path
   * @param overWrite
   * @return
   */
  override def save(path: String, overWrite: Boolean): this.type =
    throw new UnsupportedOperationException(errorString)

  // scalastyle:off methodName
  /**
   * Add all elements of this with value not in place.
   * It will allocate new memory.
   *
   * @param s
   * @return
   */
  override def +(s: T): Tensor[T] = throw new UnsupportedOperationException(errorString)

  /**
   * Add a Tensor to another one, return the result in new allocated memory.
   * The number of elements in the Tensors must match, but the sizes do not matter.
   * The size of the returned Tensor will be the size of the first Tensor
   *
   * @param t
   * @return
   */
  override def +(t: Tensor[T]): Tensor[T] = throw new UnsupportedOperationException(errorString)

  /**
   * subtract all elements of this with the value not in place.
   * It will allocate new memory.
   *
   * @param s
   * @return
   */
  override def -(s: T): Tensor[T] = throw new UnsupportedOperationException(errorString)

  /**
   * Subtract a Tensor from another one, return the result in new allocated memory.
   * The number of elements in the Tensors must match, but the sizes do not matter.
   * The size of the returned Tensor will be the size of the first Tensor
   *
   * @param t
   * @return
   */
  override def -(t: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  override def unary_-(): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * divide all elements of this with value not in place.
   * It will allocate new memory.
   *
   * @param s
   * @return
   */
  override def /(s: T): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Divide a Tensor by another one, return the result in new allocated memory.
   * The number of elements in the Tensors must match, but the sizes do not matter.
   * The size of the returned Tensor will be the size of the first Tensor
   *
   * @param t
   * @return
   */
  override def /(t: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * multiply all elements of this with value not in place.
   * It will allocate new memory.
   *
   * @param s
   * @return
   */
  override def *(s: T): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Multiply a Tensor by another one, return the result in new allocated memory.
   * The number of elements in the Tensors must match, but the sizes do not matter.
   * The size of the returned Tensor will be the size of the first Tensor
   *
   * @param t
   * @return
   */
  override def *(t: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)
  // scalastyle:on methodName

  /**
   * returns the sum of the elements of this
   *
   * @return
   */
  override def sum(): T =
    throw new UnsupportedOperationException(errorString)

  /**
   * performs the sum operation over the dimension dim
   *
   * @param dim
   * @return
   */
  override def sum(dim: Int): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  override def sum(x: Tensor[T], dim: Int): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  def prod(): T =
    throw new UnsupportedOperationException(errorString)

  def prod(x: Tensor[T], dim: Int): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * returns the mean of all elements of this.
   *
   * @return
   */
  override def mean(): T =
    throw new UnsupportedOperationException(errorString)

  /**
   * performs the mean operation over the dimension dim.
   *
   * @param dim
   * @return
   */
  override def mean(dim: Int): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * returns the single biggest element of x
   *
   * @return
   */
  override def max(): T =
    throw new UnsupportedOperationException(errorString)

  /**
   * performs the max operation over the dimension n
   *
   * @param dim
   * @return
   */
  override def max(dim: Int): (Tensor[T], Tensor[T]) =
    throw new UnsupportedOperationException(errorString)

  /**
   * performs the max operation over the dimension n
   *
   * @param values
   * @param indices
   * @param dim
   * @return
   */
  override def max(values: Tensor[T], indices: Tensor[T], dim: Int): (Tensor[T], Tensor[T]) =
    throw new UnsupportedOperationException(errorString)

  /**
   * returns the single minimum element of x
   *
   * @return
   */
  override def min(): T =
    throw new UnsupportedOperationException(errorString)

  /**
   * performs the min operation over the dimension n
   *
   * @param dim
   * @return
   */
  override def min(dim: Int): (Tensor[T], Tensor[T]) =
    throw new UnsupportedOperationException(errorString)

  /**
   * performs the min operation over the dimension n
   *
   * @param values
   * @param indices
   * @param dim
   * @return
   */
  override def min(values: Tensor[T], indices: Tensor[T], dim: Int): (Tensor[T], Tensor[T]) =
    throw new UnsupportedOperationException(errorString)

  /**
   * Writes all values from tensor src into this tensor at the specified indices
   *
   * @param dim
   * @param index
   * @param src
   * @return this
   */
  override def scatter(dim: Int, index: Tensor[T], src: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * change this tensor with values from the original tensor by gathering a number of values
   * from each "row", where the rows are along the dimension dim.
   *
   * @param dim
   * @param index
   * @param src
   * @return this
   */
  override def gather(dim: Int, index: Tensor[T], src: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * This function computes 2 dimensional convolution of a single image
   * with a single kernel (2D output). the dimensions of input and kernel
   * need to be 2, and Input image needs to be bigger than kernel. The
   * last argument controls if the convolution is a full ('F') or valid
   * ('V') convolution. The default is valid convolution.
   *
   * @param kernel
   * @param vf full ('F') or valid ('V') convolution.
   * @return
   */
  override def conv2(kernel: Tensor[T], vf: Char): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * This function operates with same options and input/output configurations as conv2,
   * but performs cross-correlation of the input with the kernel k.
   *
   * @param kernel
   * @param vf full ('F') or valid ('V') convolution.
   * @return
   */
  override def xcorr2(kernel: Tensor[T], vf: Char): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * replaces all elements in-place with the square root of the elements of this.
   *
   * @return
   */
  override def sqrt(): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * replaces all elements in-place with the absolute values of the elements of this.
   *
   * @return
   */
  override def abs(): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * x.add(value,y) multiply-accumulates values of y into x.
   *
   * @param value scalar
   * @param y     other tensor
   * @return current tensor
   */
  override def add(value: T, y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * accumulates all elements of y into this
   *
   * @param y other tensor
   * @return current tensor
   */
  override def add(y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * z.add(x, value, y) puts the result of x + value * y in z.
   *
   * @param x
   * @param value
   * @param y
   * @return
   */
  override def add(x: Tensor[T], value: T, y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * x.add(value) : add value to all elements of x in place.
   *
   * @param value
   * @return
   */
  override def add(value: T): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  override def add(x: Tensor[T], y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Performs the dot product. The number of elements must match: both Tensors are seen as a 1D
   * vector.
   *
   * @param y
   * @return
   */
  override def dot(y: Tensor[T]): T =
    throw new UnsupportedOperationException(errorString)

  /**
   * For each elements of the tensor, performs the max operation compared with the given value
   * vector.
   *
   * @param value
   * @return
   */
  override def cmax(value: T): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Performs the p-norm distance calculation between two tensors
   *
   * @param y    the secode Tensor
   * @param norm the norm of distance
   * @return
   */
  override def dist(y: Tensor[T], norm: Int): T =
    throw new UnsupportedOperationException(errorString)

  /**
   * Performs the element-wise multiplication of tensor1 by tensor2, multiply the result by the
   * scalar value (1 if not present) and add it to x. The number of elements must match, but sizes
   * do not matter.
   *
   * @param value
   * @param tensor1
   * @param tensor2
   */
  override def addcmul(value: T, tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  override def addcmul(tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Performs the element-wise division of tensor1 by tensor2, multiply the result by the scalar
   * value and add it to x.
   * The number of elements must match, but sizes do not matter.
   *
   * @param value
   * @param tensor1
   * @param tensor2
   * @return
   */
  override def addcdiv(value: T, tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  override def sub(value: T, y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  override def sub(x: Tensor[T], value: T, y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * subtracts all elements of y from this
   *
   * @param y other tensor
   * @return current tensor
   */
  override def sub(y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  override def sub(x: Tensor[T], y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  override def sub(value: T): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Element-wise multiply
   * x.cmul(y) multiplies all elements of x with corresponding elements of y.
   * x = x * y
   *
   * @param y tensor
   * @return current tensor
   */
  override def cmul(y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Element-wise multiply
   * z.cmul(x, y) equals z = x * y
   *
   * @param x tensor
   * @param y tensor
   * @return current tensor
   */
  override def cmul(x: Tensor[T], y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Element-wise divide
   * x.cdiv(y) all elements of x divide all elements of y.
   * x = x / y
   *
   * @param y tensor
   * @return current tensor
   */
  override def cdiv(y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Element-wise divide
   * z.cdiv(x, y) means z = x / y
   *
   * @param x tensor
   * @param y tensor
   * @return current tensor
   */
  override def cdiv(x: Tensor[T], y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * multiply all elements of this with value in-place.
   *
   * @param value
   * @return
   */
  override def mul(value: T): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * divide all elements of this with value in-place.
   *
   * @param value
   * @return
   */
  override def div(value: T): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * put the result of x * value in current tensor
   *
   * @param value
   * @return
   */
  override def mul(x: Tensor[T], value: T): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Performs a matrix-matrix multiplication between mat1 (2D tensor) and mat2 (2D tensor).
   * Optional values v1 and v2 are scalars that multiply M and mat1 * mat2 respectively.
   * Optional value beta is a scalar that scales the result tensor, before accumulating the result
   * into the tensor. Defaults to 1.0.
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
  override def addmm(v1: T, M: Tensor[T], v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /** res = M + (mat1*mat2) */
  override def addmm(M: Tensor[T], mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /** res = res + mat1 * mat2 */
  override def addmm(mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /** res = res + v2 * mat1 * mat2 */
  override def addmm(v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /** res = v1 * res + v2 * mat1*mat2 */
  override def addmm(v1: T, v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /** res = mat1*mat2 */
  override def mm(mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

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
  override def addr(t1: Tensor[T], t2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  override def addr(v1: T, t1: Tensor[T], t2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  override def addr(v1: T, t1: Tensor[T], v2: T, t2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Performs the outer-product between vec1 (1D Tensor) and vec2 (1D Tensor).
   * Optional values v1 and v2 are scalars that multiply mat and vec1 [out] vec2 respectively.
   * In other words,res_ij = (v1 * mat_ij) + (v2 * vec1_i * vec2_j)
   *
   * @param v1
   * @param t1
   * @param v2
   * @param t2
   * @param t3
   * @return
   */
  override def addr(v1: T, t1: Tensor[T], v2: T, t2: Tensor[T], t3: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * return pseudo-random numbers, require 0<=args.length<=2
   * if args.length = 0, return [0, 1)
   * if args.length = 1, return [1, args(0)] or [args(0), 1]
   * if args.length = 2, return [args(0), args(1)]
   *
   * @param args
   */
  override def uniform(args: T*): T =
    throw new UnsupportedOperationException(errorString)

  /**
   * Performs a matrix-vector multiplication between mat (2D Tensor) and vec2 (1D Tensor) and add
   * it to vec1. Optional values v1 and v2 are scalars that multiply vec1 and vec2 respectively.
   *
   * In other words,
   * res = (beta * vec1) + alpha * (mat * vec2)
   *
   * Sizes must respect the matrix-multiplication operation: if mat is a n × m matrix,
   * vec2 must be vector of size m and vec1 must be a vector of size n.
   */
  override def addmv(beta: T, vec1: Tensor[T], alpha: T, mat: Tensor[T],
    vec2: Tensor[T]): Tensor[T] = throw new UnsupportedOperationException(errorString)

  /** res = beta * res + alpha * (mat * vec2) */
  override def addmv(beta: T, alpha: T, mat: Tensor[T], vec2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /** res = res + alpha * (mat * vec2) */
  override def addmv(alpha: T, mat: Tensor[T], vec2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /** res = res + (mat * vec2) */
  override def mv(mat: Tensor[T], vec2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Perform a batch matrix matrix multiplication of matrices and stored in batch1 and batch2
   * with batch add. batch1 and batch2 must be 3D Tensors each containing the same number of
   * matrices. If batch1 is a b × n × m Tensor, batch2 a b × m × p Tensor, res will be a
   * b × n × p Tensor.
   *
   * In other words,
   * res_i = (beta * M_i) + (alpha * batch1_i * batch2_i)
   */
  override def baddbmm(beta: T, M: Tensor[T], alpha: T, batch1: Tensor[T],
    batch2: Tensor[T]): Tensor[T] = throw new UnsupportedOperationException(errorString)

  /** res_i = (beta * res_i) + (alpha * batch1_i * batch2_i) */
  override def baddbmm(beta: T, alpha: T, batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /** res_i = res_i + (alpha * batch1_i * batch2_i) */
  override def baddbmm(alpha: T, batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /** res_i = res_i + batch1_i * batch2_i */
  override def bmm(batch1: Tensor[T], batch2: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Replaces all elements in-place with the elements of x to the power of n
   *
   * @param y
   * @param n
   * @return current tensor reference
   */
  override def pow(y: Tensor[T], n: T): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  override def pow(n: T): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  override def square(): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Get the top k smallest values and their indices.
   *
   * @param result   result buffer
   * @param indices  indices buffer
   * @param k
   * @param dim      dimension, default is the last dimension
   * @param increase sort order, set it to true if you want to get the smallest top k values
   * @return
   */
  override def topk(k: Int, dim: Int, increase: Boolean, result: Tensor[T],
    indices: Tensor[T], sortedResult: Boolean = true): (Tensor[T], Tensor[T]) =
    throw new UnsupportedOperationException(errorString)

  /**
   * Replaces all elements in-place with the elements of lnx
   *
   * @param y
   * @return current tensor reference
   */
  override def log(y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  override def exp(y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  override def sqrt(y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  override def log1p(y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)
  override def log(): Tensor[T] =
    throw new UnsupportedOperationException(errorString)
  override def exp(): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  override def log1p(): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  override def abs(x: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * returns the p-norms of the Tensor x computed over the dimension dim.
   *
   * @param y result buffer
   * @param value
   * @param dim
   * @return
   */
  override def norm(y: Tensor[T], value: Int, dim: Int): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Implements > operator comparing each element in x with y
   *
   * @param x
   * @param y
   * @return current tensor reference
   */
  override def gt(x: Tensor[T], y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Implements < operator comparing each element in x with y
   *
   * @param x
   * @param y
   * @return current tensor reference
   */
  override def lt(x: Tensor[T], y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Implements <= operator comparing each element in x with y
   *
   * @param x
   * @param y
   * @return current tensor reference
   */
  override def le(x: Tensor[T], y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Implements == operator comparing each element in x with y
   *
   * @param y
   * @return current tensor reference
   */
  override def eq(x: Tensor[T], y: T): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Fills the masked elements of itself with value val
   *
   * @param mask
   * @param e
   * @return current tensor reference
   */
  override def maskedFill(mask: Tensor[T], e: T): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Copies the elements of tensor into mask locations of itself.
   *
   * @param mask
   * @param y
   * @return current tensor reference
   */
  override def maskedCopy(mask: Tensor[T], y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Returns a new Tensor which contains all elements aligned to a 1 in the corresponding mask.
   *
   * @param mask
   * @param y
   * @return current tensor reference
   */
  override def maskedSelect(mask: Tensor[T], y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * returns the sum of the n-norms on the Tensor x
   *
   * @param value the n-norms
   * @return
   */
  override def norm(value: Int): T =
    throw new UnsupportedOperationException(errorString)

  /**
   * returns a new Tensor with the sign (+/- 1 or 0) of the elements of x.
   *
   * @return
   */
  override def sign(): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Implements >= operator comparing each element in x with value
   *
   * @param x
   * @param value
   * @return
   */
  override def ge(x: Tensor[T], value: Double): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Accumulate the elements of tensor into the original tensor by adding to the indices
   * in the order given in index. The shape of tensor must exactly match the elements indexed
   * or an error will be thrown.
   *
   * @param dim
   * @param index
   * @param y
   * @return
   */
  override def indexAdd(dim: Int, index: Tensor[T], y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * Accumulate the elements of tensor into the original tensor by adding to the indices
   * in the order given in index. The shape of tensor must exactly match the elements indexed
   * or an error will be thrown.
   *
   * @param dim
   * @param index
   * @param y
   * @return
   */
  override def index(dim: Int, index: Tensor[T], y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * stores the element-wise maximum of x and y in x.
   * x.cmax(y) = max(x, y)
   *
   * @param y tensor
   * @return current tensor
   */
  override def cmax(y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * stores the element-wise maximum of x and y in x.
   * x.cmin(y) = min(x, y)
   *
   * @param y tensor
   * @return current tensor
   */
  override def cmin(y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * stores the element-wise maximum of x and y in z.
   * z.cmax(x, y) means z = max(x, y)
   *
   * @param x tensor
   * @param y tensor
   */
  override def cmax(x: Tensor[T], y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * stores the element-wise maximum of x and y in z.
   * z.cmin(x, y) means z = min(x, y)
   *
   * @param x tensor
   * @param y tensor
   */
  override def cmin(x: Tensor[T], y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  /**
   * resize this tensor size to floor((xmax - xmin) / step) + 1 and set values from
   * xmin to xmax with step (default to 1).
   *
   * @param xmin
   * @param xmax
   * @param step
   * @return this tensor
   */
  override def range(xmin: Double, xmax: Double, step: Int): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  override def toTensor[D](implicit ev: TensorNumeric[D]): Tensor[D] =
    throw new UnsupportedOperationException(errorString)

  override def tanh(): Tensor[T] = throw new UnsupportedOperationException(errorString)

  override def tanh(y: Tensor[T]): Tensor[T] = throw new UnsupportedOperationException(errorString)

  override def resize(sizes: Array[Int], strides: Array[Int]): this.type =
    throw new UnsupportedOperationException(errorString)

  override def resize(size1: Int): this.type = throw new UnsupportedOperationException(errorString)

  override def resize(size1: Int, size2: Int): this.type =
    throw new UnsupportedOperationException(errorString)

  override def resize(size1: Int, size2: Int, size3: Int): this.type =
    throw new UnsupportedOperationException(errorString)

  override def resize(size1: Int, size2: Int, size3: Int, size4: Int): this.type =
    throw new UnsupportedOperationException(errorString)

  override def resize(size1: Int, size2: Int, size3: Int, size4: Int, size5: Int): this.type =
    throw new UnsupportedOperationException(errorString)

  override def isEmpty: Boolean =
    throw new UnsupportedOperationException(errorString)

  override def isScalar: Boolean =
    throw new UnsupportedOperationException(errorString)

  override def value(): T =
    throw new UnsupportedOperationException(errorString)

  override def setValue(value: T): this.type =
    throw new UnsupportedOperationException(errorString)

  override def zipWith[A: ClassTag, B: ClassTag](t1: Tensor[A], t2: Tensor[B],
    func: (A, B) => T): Tensor[T] = throw new UnsupportedOperationException(errorString)

  override def forceFill(v: Any): Tensor[T] = throw new UnsupportedOperationException(errorString)

  override def emptyInstance(): Tensor[T] = throw new UnsupportedOperationException(errorString)

  override def applyFun[A: ClassTag](t: Tensor[A], func: (A) => T): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  override def cast[D: ClassTag](castTensor: Tensor[D])(implicit ev: TensorNumeric[D]): Tensor[D] =
    throw new UnsupportedOperationException(errorString)

  override def div(y: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  override def floor(y: Tensor[T]): Tensor[T] = throw new UnsupportedOperationException(errorString)

  override def floor(): Tensor[T] = throw new UnsupportedOperationException(errorString)

  override def ceil(): Tensor[T] = throw new UnsupportedOperationException(errorString)

  override def negative(x: Tensor[T]): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  override def inv(): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  override def reduce(dim: Int, result: Tensor[T], reducer: (T, T) => T): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  override def toArray(): Array[T] =
    throw new UnsupportedOperationException(errorString)

  override def erf(): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  override def erfc(): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  override def logGamma(): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  override def digamma(): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  override def clamp(minValue: Double, maxValue: Double): Tensor[T] =
    throw new UnsupportedOperationException(errorString)

  override def sumSquare(): T =
    throw new UnsupportedOperationException(errorString)
}
