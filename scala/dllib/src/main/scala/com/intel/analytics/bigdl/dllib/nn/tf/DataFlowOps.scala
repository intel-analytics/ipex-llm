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
package com.intel.analytics.bigdl.nn.tf

import java.util

import com.intel.analytics.bigdl.nn.ops.Operation
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

private[nn] trait ResourceAllocator {
  def release(): Unit
}

/**
 * This class implement the functionality of TensorArray in tensorflow. See details at
 *   https://www.tensorflow.org/api_docs/python/tf/TensorArray.
 *
 * @param initSize The initial size of the array.
 * @param shape The expected shape of the element in the tensor array.
 * @param dynamicSize Whether write to the tensor array is allowed to grow the size. By default it's
 *                    not allowed.
 * @param clearAfterRead Determines whether the tensors are cleared after read. Default is true.
 * @param identicalElementShapes If all elements in the array should have the same shape. Default is
 *                               false.
 * @tparam D Element numeric type in the tensor array.
 */
private[nn] class TensorArray[D: ClassTag](
  private val initSize: Int,
  private val shape: Array[Int] = null,
  private var dynamicSize: Boolean = false,
  private val clearAfterRead: Boolean = true,
  private val identicalElementShapes: Boolean = false,
  private val multipleWritesAggregate: Boolean = false
)(implicit ev: TensorNumeric[D]) {

  private var otherShape : Array[Int] = null

  private var tensors = new Array[Tensor[D]](initSize)

  def lockSize(): Unit = this.dynamicSize = false

  def apply(index: Int): Tensor[D] = {
    require(tensors(index) != null,
      s"tensor on index $index has not been inited or has been cleared")
    val t = tensors(index)
    if (clearAfterRead) tensors(index) = null
    t
  }

  def grad(): TensorArray[_] = {
    this.lockSize()
    new TensorArray[D](this.size, multipleWritesAggregate = true)
  }

  def size(): Int = tensors.length

  def shapeOf(index: Int): Array[Int] = {
    require(tensors(index) != null,
      s"tensor on index $index has not been inited or has been cleared")
    tensors(index).size()
  }

  def update(index: Int, tensor: Tensor[D]): Unit = {
    if (!multipleWritesAggregate) {
      require(tensors(index) == null, "There's already a tensor on the given index")
    }

    if (identicalElementShapes) {
      if (otherShape == null) {
        otherShape = tensor.size()
      } else {
        val curShape = tensor.size()
        require(curShape.length == otherShape.length,
          "insert tensor dimension does not match other tensor dimension")
        var i = 0
        while(i < curShape.length) {
          require(curShape(i) == otherShape(i),
            "insert tensor size does not match other tensor size")
          i += 1
        }
      }
    }
    if (shape != null) {
      val curShape = tensor.size()
      require(curShape.length == shape.length,
        "insert tensor dimension does not match required dimension")
      var i = 0
      while(i < curShape.length) {
        require(curShape(i) == shape(i),
          "insert tensor size does not match required size")
        i += 1
      }
    }

    if (dynamicSize && index >= tensors.size) {
      val newTensors = new Array[Tensor[D]](index + 1)
      var i = 0
      while(i < tensors.length) {
        newTensors(i) = tensors(i)
        i += 1
      }
      tensors = newTensors
    } else {
      require(index < initSize, "cannot grow size when dynamicSize is false")
    }

    if (tensors(index) == null) {
      tensors(index) = Tensor[D]().resizeAs(tensor).copy(tensor)
    } else {
      tensors(index).add(tensor)
    }
  }
}

private[nn] object TensorArray {
  private val arrays = new util.WeakHashMap[String, TensorArray[_]]()

  def apply[D](key: String): TensorArray[D] = this.synchronized {
    require(arrays.containsKey(key), s"Cannot find TensorArray for name $key")
    arrays.get(key).asInstanceOf[TensorArray[D]]
  }

  def get(key: String): TensorArray[_] = this.synchronized {
    require(arrays.containsKey(key), s"Cannot find TensorArray for name $key")
    arrays.get(key)
  }

  def update(key: String, value: TensorArray[_]): Unit = this.synchronized {
    arrays.put(key, value)
  }

  def exist(key: String): Boolean = this.synchronized {
    arrays.containsKey(key)
  }

  def release(key : String): Unit = this.synchronized {
    if (arrays.containsKey(key)) arrays.remove(key)
  }

  // A scalar used to control gradient flow
  val FlowOut: Tensor[Float] = Tensor.scalar(0.0f)
}

/**
 * Create a tensor array in the context. Return the handle of the tensor array and a control flow
 * scalar.
 *
 * @param shape The expected shape of the element in the tensor array.
 * @param dynamicSize Whether write to the tensor array is allowed to grow the size. By default it's
 *                    not allowed.
 * @param clearAfterRead Determines whether the tensors are cleared after read. Default is true.
 * @param identicalElementShapes If all elements in the array should have the same shape. Default is
 *                               false.
 * @param tensorArrayName a unique string which is used to find the created tensor array.
 * @tparam T Model parameter numeric type.
 * @tparam D Element numeric type in the tensor array.
 */
private[bigdl] class TensorArrayCreator[T: ClassTag, D: ClassTag](
  shape: Array[Int] = null,
  dynamicSize: Boolean = false,
  clearAfterRead: Boolean = true,
  identicalElementShapes: Boolean = false,
  tensorArrayName: String = ""
)(implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends Operation[Tensor[Int], Table, T] with ResourceAllocator {

  override def updateOutput(input: Tensor[Int]): Table = {
    require(input.isScalar, "input size must be a int scalar")

    val handle = getHandleName()

    TensorArray(handle) = new TensorArray[D](input.value(), shape, dynamicSize, clearAfterRead,
      identicalElementShapes)

    output = T(
      Tensor.scalar(handle),
      TensorArray.FlowOut
    )
    output
  }

  override def release(): Unit = {
    TensorArray.release(getHandleName())
  }

  private def getHandleName(): String = {
    if (tensorArrayName == "") {
      this.getName() + System.identityHashCode(this)
    } else {
      tensorArrayName + System.identityHashCode(this)
    }
  }

  override def getClassTagNumerics(): (Array[ClassManifest[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}

/**
 * Create a TensorArray to store the gradient of values in the given handle. Return the handle of
 * the gradient TensorArray and a control flow scalar.
 *
 * If the given TensorArray gradients already exists, just return a reference.
 *
 * Locks the size of the original TensorArray by disabling its dynamic size flag.
 *
 * @param source a suffix to append to the name of the passed in TensorArray, used as key to locate
 *               the gradient TensorArray
 * @tparam T Model parameter numeric type.
 */
private[bigdl] class TensorArrayGrad[T: ClassTag](source: String)(
  implicit ev: TensorNumeric[T]) extends Operation[Table, Table, T]{

  override def updateOutput(input: Table): Table = {
    val handle = input[Tensor[String]](1)
    require(handle.isScalar, "Handle of a TensorArray must be a scalar")

    val tensorArray = TensorArray.get(handle.value())
    val name = handle.value() + source
    if (!TensorArray.exist(name)) {
      TensorArray(name) = tensorArray.grad()
    }
    output = T(
      Tensor.scalar[String](name),
      TensorArray.FlowOut
    )
    output
  }
}

/**
 * Insert an element tensor to tensor array. It accepts a TensorArray handle and an Int scalar
 * index, and returns a control flow object.
 *
 * @tparam T Model parameter numeric type.
 * @tparam D Element numeric type in the tensor array.
 */
private[bigdl] class TensorArrayWrite[T: ClassTag, D: ClassTag]()(
  implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends Operation[Table, Tensor[Float], T]{

  output = TensorArray.FlowOut

  override def updateOutput(input: Table): Tensor[Float] = {
    val handle = input[Tensor[String]](1)
    val index = input[Tensor[Int]](2)
    val value = input[Tensor[D]](3)
    require(handle.isScalar, "Handle of a TensorArray must be a scalar")
    require(index.isScalar, "Index must be a scalar")

    val tensorArray = TensorArray[D](handle.value())
    tensorArray(index.value()) = value
    output
  }

  override def getClassTagNumerics(): (Array[ClassManifest[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }

  override def clearState(): this.type = {
    // Do nothing in clearState as we don't want to change the TensorArray.FlowOut object
    this
  }
}

/**
 * Read an element from the TensorArray into output `value`. It accepts a TensorArray handle and an
 * Int scalar index, and returns the tensor object.
 *
 * @tparam T Model parameter numeric type.
 * @tparam D Element numeric type in the tensor array.
 */
private[bigdl] class TensorArrayRead[T: ClassTag, D: ClassTag]()(
  implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends Operation[Table, Tensor[D], T]{

  override def updateOutput(input: Table): Tensor[D] = {
    val handle = input[Tensor[String]](1)
    val index = input[Tensor[Int]](2)
    require(handle.isScalar, "Handle of a TensorArray must be a scalar")
    require(index.isScalar, "Index must be a scalar")

    val tensorArray = TensorArray[D](handle.value())
    output = tensorArray(index.value())
    output
  }

  override def getClassTagNumerics(): (Array[ClassManifest[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}

/**
 * Gather specific elements from the TensorArray into output `value`. It accepts two input:
 *   handle: The handle to a TensorArray.
 *   indices: The locations in the TensorArray from which to read tensor elements.
 *
 * It returns a gathered tensor.
 *
 * All elements selected by `indices` must have the same shape.
 *
 * @tparam T Model parameter numeric type.
 * @tparam D Element numeric type in the tensor array.
 */
private[bigdl] class TensorArrayGather[T: ClassTag, D: ClassTag]()(
  implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends Operation[Table, Tensor[D], T]{

  output = Tensor[D]()

  override def updateOutput(input: Table): Tensor[D] = {
    val handle = input[Tensor[String]](1)
    val indices = input[Tensor[Int]](2)
    require(handle.isScalar, "Handle of a TensorArray must be a scalar")
    require(indices.nDimension() == 1, "indices must be a vector")

    val tensorArray = TensorArray[D](handle.value())

    var sizes : Array[Int] = null
    var i = 1
    while(i <= indices.size(1)) {
      if (sizes == null) {
        sizes = tensorArray.shapeOf(indices.valueAt(i))
      } else {
        val curSizes = tensorArray.shapeOf(indices.valueAt(i))
        require(curSizes.length == sizes.length, "the selected tensors have different dimensions")
        var j = 0
        while(j < sizes.length) {
          require(sizes(j) == curSizes(j), "the selected tensors have different sizes")
          j += 1
        }
      }
      i += 1
    }

    output.resize(Array(indices.size(1)) ++ sizes)
    i = 1
    while(i <= indices.size(1)) {
      output.select(1, i).copy(tensorArray(indices.valueAt(i)))
      i += 1
    }

    output
  }

  override def getClassTagNumerics(): (Array[ClassManifest[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}

/**
 * Scatter the data from the input value into specific TensorArray elements. It is a 'reverse'
 * operation of the gather.
 *
 * It accepts three inputs:
 *   handle: The handle to a TensorArray.
 *   indices: The locations at which to write the tensor elements.
 *   value: The concatenated tensor to write to the TensorArray.
 *
 * And returns a control flow objects
 *
 * @tparam T Model parameter numeric type.
 * @tparam D Element numeric type in the tensor array.
 */
private[bigdl] class TensorArrayScatter[T: ClassTag, D: ClassTag]()(
  implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends Operation[Table, Tensor[Float], T]{

  output = TensorArray.FlowOut

  override def updateOutput(input: Table): Tensor[Float] = {
    val handle = input[Tensor[String]](1)
    val indices = input[Tensor[Int]](2)
    val value = input[Tensor[D]](3)
    require(handle.isScalar, "Handle of a TensorArray must be a scalar")
    require(indices.nDimension() == 1, "indices must be a vector")
    require(indices.size(1) == value.size(1), "indices length does not match value first dimension")

    val tensorArray = TensorArray[D](handle.value())

    var i = 1
    while(i <= indices.size(1)) {
      tensorArray(indices.valueAt(i)) = value.select(1, i)
      i += 1
    }

    output
  }

  override def getClassTagNumerics(): (Array[ClassManifest[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }

  override def clearState(): this.type = {
    // Do nothing in clearState as we don't want to change the TensorArray.FlowOut object
    this
  }
}

/**
 * Concat the elements from the TensorArray into value `value`.
 *
 * Takes `T` elements of shapes
 *   (n0 x d0 x d1 x ...), (n1 x d0 x d1 x ...), ..., (n(T-1) x d0 x d1 x ...)
 *
 * and concatenates them into a Tensor of shape:
 *   (n0 + n1 + ... + n(T-1) x d0 x d1 x ...)
 *
 * It return the concated value.
 *
 * All elements must have the same shape (excepting the first dimension).
 *
 * @tparam T Model parameter numeric type.
 * @tparam D Element numeric type in the tensor array.
 */
private[bigdl] class TensorArrayConcat[T: ClassTag, D: ClassTag]()(
  implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends Operation[Table, Table, T] {

  override def updateOutput(input: Table): Table = {
    val handle = input[Tensor[String]](1)
    require(handle.isScalar, "Handle of a TensorArray must be a scalar")

    val tensorArray = TensorArray[D](handle.value())

    val size = tensorArray.shapeOf(0)
    size(0) = 0
    val lengths = Tensor[Int](tensorArray.size)
    var i = 0
    while(i < tensorArray.size) {
      size(0) += tensorArray.shapeOf(i)(0)
      lengths.setValue(i + 1, tensorArray.shapeOf(i)(0))
      i += 1
    }

    val value = Tensor[D]().resize(size)
    i = 0
    var index = 1
    while(i < tensorArray.size) {
      val curSize = tensorArray.shapeOf(i)(0)
      value.narrow(1, index, curSize).copy(tensorArray(i))
      index += curSize
      i += 1
    }

    output = T(value, lengths)
    output
  }

  override def getClassTagNumerics(): (Array[ClassManifest[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}

/**
 * Split the data from the input value into TensorArray elements. It is a 'reverse' operation of
 * concat. It accepts:
 *   handle: The handle to a TensorArray.
 *   value: The concatenated tensor to write to the TensorArray.
 *   lengths: The vector of lengths, how to split the rows of value into the TensorArray.
 *
 * It return a control flow object.
 *
 * Assuming that `lengths` takes on values
 *   (n0, n1, ..., n(T-1))
 * and that value has shape
 *   (n0 + n1 + ... + n(T-1) x d0 x d1 x ...),
 * this splits values into a TensorArray with T tensors.
 * TensorArray index t will be the subtensor of values with starting position
 *   `(n0 + n1 + ... + n(t-1), 0, 0, ...)```
 * and having size
 *   nt x d0 x d1 x ...
 *
 * @tparam T Model parameter numeric type.
 * @tparam D Element numeric type in the tensor array.
 */
private[bigdl] class TensorArraySplit[T: ClassTag, D: ClassTag]()(
  implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends Operation[Table, Tensor[Float], T] {

  output = TensorArray.FlowOut

  override def updateOutput(input: Table): Tensor[Float] = {
    val handle = input[Tensor[String]](1)
    val value = input[Tensor[D]](2)
    val lengths = input[Tensor[Int]](3)
    require(handle.isScalar, "Handle of a TensorArray must be a scalar")
    require(lengths.nDimension() == 1, "lengths must be a vector")

    val tensorArray = TensorArray[D](handle.value())

    var i = 1
    var index = 1
    while(i <= lengths.size(1)) {
      tensorArray(i - 1) = value.narrow(1, index, lengths.valueAt(i))
      index += lengths.valueAt(i)
      i += 1
    }

    output
  }

  override def getClassTagNumerics(): (Array[ClassManifest[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }

  override def clearState(): this.type = {
    // Do nothing in clearState as we don't want to change the TensorArray.FlowOut object
    this
  }
}

/**
 * Get the current size of the TensorArray.
 *
 * @tparam T Model parameter numeric type.
 */
private[bigdl] class TensorArraySize[T: ClassTag]()(implicit ev: TensorNumeric[T])
  extends Operation[Table, Tensor[Int], T] {

  override def updateOutput(input: Table): Tensor[Int] = {
    val handle = input[Tensor[String]](1)
    require(handle.isScalar, "Handle of a TensorArray must be a scalar")

    val tensorArray = TensorArray(handle.value())

    output = Tensor.scalar[Int](tensorArray.size)
    output
  }
}

/**
 * Delete the TensorArray from the context.
 *
 * @tparam T Model parameter numeric type.
 */
private[bigdl] class TensorArrayClose[T: ClassTag]()(implicit ev: TensorNumeric[T])
  extends Operation[Tensor[String], Tensor[Float], T] {

  output = TensorArray.FlowOut

  override def updateOutput(input: Tensor[String]): Tensor[Float] = {
    require(input.isScalar, "Handle of a TensorArray must be a scalar")
    TensorArray.release(input.value())
    output
  }

  override def clearState(): this.type = {
    // Do nothing in clearState as we don't want to change the TensorArray.FlowOut object
    this
  }
}

private[bigdl] class Stack[D](maxSize: Int) {
  private var count = 0
  private val tensors = new ArrayBuffer[Tensor[D]]()

  def pop(): Tensor[D] = {
    require(count > 0, "There's no tensors in the stack")
    count -= 1
    tensors.remove(count)
  }

  def push(t: Tensor[D]): Unit = {
    require(count < maxSize, "Stack is full")
    tensors.append(t.clone())
    count += 1
  }
}

private[bigdl] object Stack {
  private val stacks = new util.WeakHashMap[String, Stack[_]]()

  def apply[D](key: String): Stack[D] = this.synchronized {
    require(stacks.containsKey(key), s"Cannot find Stack for name $key")
    stacks.get(key).asInstanceOf[Stack[D]]
  }

  def update(key: String, value: Stack[_]): Unit = this.synchronized {
    stacks.put(key, value)
  }

  def release(key : String): Unit = this.synchronized {
    if (stacks.containsKey(key)) stacks.remove(key)
  }
}

private[bigdl] class StackCreator[T: ClassTag, D: ClassTag](
  private val name: String = "")(implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends Operation[Tensor[Int], Tensor[String], T] with WithoutInput with ResourceAllocator {
  override def updateOutput(input: Tensor[Int]): Tensor[String] = {
    require(input == null || input.isScalar,
      "StackCreator: Input tensor should be a scalar or no input")

    val handle = getHandleName()

    Stack(handle) = new Stack[D](
      if (input == null || input.value() < 0) Int.MaxValue else input.value())
    output = Tensor.scalar(handle)
    output
  }

  override def release(): Unit = {
    Stack.release(getHandleName())
  }

  private def getHandleName(): String = {
    if (name == "") {
      this.getName() + System.identityHashCode(this)
    } else {
      name + System.identityHashCode(this)
    }
  }

  override def getClassTagNumerics(): (Array[ClassManifest[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}

private[bigdl] class StackPop[T: ClassTag, D: ClassTag]()
  (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends Operation[Tensor[String], Tensor[D], T]{
  override def updateOutput(input: Tensor[String]): Tensor[D] = {
    require(input.isScalar, "StackPop: Input tensor should be a scalar")
    val handle = input.value()
    output = Stack[D](handle).pop()
    output
  }

  override def getClassTagNumerics(): (Array[ClassManifest[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}

private[bigdl] class StackPush[T: ClassTag, D: ClassTag]()
  (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends Operation[Table, Tensor[D], T]{
  override def updateOutput(input: Table): Tensor[D] = {
    val handleTensor = input[Tensor[String]](1)
    require(handleTensor.isScalar, "StackPush: Input tensor should be a scalar")
    val handle = handleTensor.value()
    val data = input[Tensor[D]](2)
    Stack[D](handle).push(data)
    output = data
    output
  }

  override def getClassTagNumerics(): (Array[ClassManifest[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}
