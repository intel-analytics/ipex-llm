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
package com.intel.analytics.bigdl.nn.ops

import java.util.concurrent.ConcurrentHashMap

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{RandomGenerator, T, Table}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

private[nn] class TensorArrayInstance[D: ClassTag](
  val size: Int,
  val shape: Option[Seq[Int]] = None,
  val dynamicSize: Boolean = false,
  val clearAfterRead: Boolean = true,
  val identicalElementShapes: Boolean = false,
  val tensorArrayName: String = null)(implicit ev: TensorNumeric[D]) {

  private val tensors = new ArrayBuffer[Tensor[D]](size)

  def apply(index: Int): Tensor[D] = tensors(index)

  def update(index: Int, tensor: Tensor[D]): Unit = {
    tensors(index) = tensor
  }
}

class TensorArray[T: ClassTag, D: ClassTag](
  shape: Option[Seq[Int]] = None,
  dynamicSize: Boolean = false,
  clearAfterRead: Boolean = true,
  identicalElementShapes: Boolean = false,
  tensorArrayName: String = null
)(implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends Operation[Tensor[Int], Table, T]{

  override def updateOutput(input: Tensor[Int]): Table = {
    val handle = if (tensorArrayName == null) {
      RandomGenerator.RNG.random().toString
    } else {
      tensorArrayName + RandomGenerator.RNG.random()
    }

    TensorArray.tensorArrayInstances.put(handle,
      new TensorArrayInstance[D](input.value(), shape, dynamicSize, clearAfterRead,
        identicalElementShapes, handle))
    output = T(
      Tensor.scalar[String](handle),
      Tensor.scalar[Float](0.0f)
    )
    output
  }
}

object TensorArray {
  private[nn] val tensorArrayInstances = new ConcurrentHashMap[String, TensorArrayInstance[_]]()

  def apply[T: ClassTag, D: ClassTag](
    shape: Option[Seq[Int]] = None,
    dynamicSize: Boolean = false,
    clearAfterRead: Boolean = true,
    identicalElementShapes: Boolean = false,
    tensorArrayName: String = null
  )(implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]): TensorArray[T, D] =
    new TensorArray(shape, dynamicSize, clearAfterRead, identicalElementShapes, tensorArrayName)
}

class TensorArrayGrad[T: ClassTag, D: ClassTag](
  source: String
)(implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends Operation[Table, Table, T]{
  override def updateOutput(input: Table): Table = {
    val handle = input[Tensor[String]](1)
    val flowIn = input[Tensor[Float]](2)

    val originalTensorArray = TensorArray.tensorArrayInstances.get(handle.value())
    TensorArray.tensorArrayInstances.put(handle.value() + source,
      new TensorArrayInstance[D](originalTensorArray.size,
        originalTensorArray.shape,
        originalTensorArray.dynamicSize,
        originalTensorArray.clearAfterRead,
        originalTensorArray.identicalElementShapes,
        handle.value() + source))
    output = T(
      Tensor.scalar[String](handle.value() + source),
      Tensor.scalar[Float](0.0f)
    )
    output
  }
}

object TensorArrayGrad {
  def apply[T: ClassTag, D: ClassTag](source: String)(
    implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]): TensorArrayGrad[T, D] =
    new TensorArrayGrad(source)
}

class TensorArrayWrite[T: ClassTag, D: ClassTag]()(
  implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends Operation[Table, Tensor[Float], T]{

  override def updateOutput(input: Table): Tensor[Float] = {
    val handle = input[Tensor[String]](1)
    val index = input[Tensor[Int]](2)
    val value = input[Tensor[D]](3)
    val flowIn = input[Tensor[Float]](4)

    val tensorArray = TensorArray.tensorArrayInstances.get(handle.value())
      .asInstanceOf[TensorArrayInstance[D]]
    tensorArray(index.value()) = value
    output = Tensor.scalar[Float](0.0f)
    output
  }
}

object TensorArrayWrite {
  def apply[T: ClassTag, D: ClassTag]()(
    implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]): TensorArrayWrite[T, D] =
    new TensorArrayWrite()
}

class TensorArrayRead[T: ClassTag, D: ClassTag]()(
  implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends Operation[Table, Tensor[D], T]{

  override def updateOutput(input: Table): Tensor[Float] = {
    val handle = input[Tensor[String]](1)
    val index = input[Tensor[Int]](2)
    val flowIn = input[Tensor[Float]](4)

    val tensorArray = TensorArray.tensorArrayInstances.get(handle.value())
      .asInstanceOf[TensorArrayInstance[D]]
    output = tensorArray(index.value())
    output
  }
}

object TensorArrayRead {
  def apply[T: ClassTag, D: ClassTag]()(
    implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]): TensorArrayRead[T, D] =
    new TensorArrayRead()
}

