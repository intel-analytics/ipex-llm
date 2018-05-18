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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.utils.serializer.{DeserializeContext, ModuleSerializable, SerializeContext}
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}
import com.intel.analytics.bigdl.utils.Table

import scala.collection.BitSet
import scala.reflect.ClassTag
import scala.reflect.runtime.universe

/**
 * Extracts a strided slice of a tensor.
 * The input of this layer should have 4 input, the first one is input data,
 * the second one is begin index of slicing, the third one is end index of slicing,
 * the third one is strides.
 * begin, end and strides should be 1D tensor, and have.
 *
 * In each mask field (beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask)
 * the ith bit will correspond to the ith spec.
 * @param beginMask If ith bit is set, begin(i) is ignored and the fullest possible
 *                  range in that dimension is used instead.
 * @param endMask If ith bit is set, end(i) is ignored and the fullest possible
 *                range in that dimension is used instead.
 * @param ellipsisMask Unsupported currently.
 *                     If ith bit is set, as many unspecified dimensions as needed
 *                     will be inserted between other dimensions.
 * @param newAxisMask Unsupported currently.
 *                    If ith bit is set, begin, end, and stride are ignored and a
 *                    new length 1 dimension is added at this point in the output tensor.
 * @param shrinkAxisMask If the ith bit is set, it implies that the ith specification
 *                       shrinks the dimensionality by 1.
 * @param startFromZero if begin, end is counted from zero.
 */
@SerialVersionUID(4436600172725317184L)
private[bigdl] class StridedSlice[T: ClassTag, D: ClassTag](
    val beginMask: Int = 0,
    val endMask: Int = 0,
    val ellipsisMask: Int = 0,
    val newAxisMask: Int = 0,
    val shrinkAxisMask: Int = 0,
    val startFromZero: Boolean = false)(implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends AbstractModule[Table, Tensor[D], T] {

  // TODO: support ellipsisMask and newAxisMask
  require(ellipsisMask == 0, s"Only support ellipsisMask equals 0, but got $ellipsisMask")
  require(newAxisMask == 0, s"Only support newAxisMask equals 0, but got $newAxisMask")
  output = Tensor[D]()
  gradInput(1) = Tensor[D]()

  val beginBuffer: Tensor[Int] = Tensor[Int]()
  val endBuffer: Tensor[Int] = Tensor[Int]()
  val stridesBuffer: Tensor[Int] = Tensor[Int]()

  protected def checkSize(strides: Tensor[Int], indx: Tensor[Int], indxName: String): Unit = {
    require(indx.dim() == 1, s"$indxName should be a 1D tensor, but got ${indx.dim()}D tensor.")
    require(indx.nElement() == strides.nElement(), s"$indxName have ${strides.nElement()} " +
      s"elements, but got ${indx.nElement()}.")
  }

  protected def maskValue(mask: Int, ith: Int): Boolean = {
    (mask >> ith) % 2 == 1
  }

  protected def getPositiveIndices(
        inputSize: Array[Int],
        indices: Tensor[Int],
        buffer: Tensor[Int]): Tensor[Int] = {
    buffer.resizeAs(indices)
    var i = 1
    while(i <= inputSize.length) {
      val index = indices.valueAt(i)
      if (index >= 0) {
        buffer.setValue(i, index)
      } else {
        buffer.setValue(i, index + inputSize(i - 1))
      }
      i += 1
    }
    if (startFromZero) {
      buffer.apply1(_ + 1)
    }
    buffer
  }

  override def updateOutput(input: Table): Tensor[D] = {
    val inputs = input[Tensor[D]](1)
    val begin = input[Tensor[Int]](2)
    val end = input[Tensor[Int]](3)
    val strides = input[Tensor[Int]](4)

    require(strides.dim() == 1, s"strides should be a 1D tensor, but got ${strides.dim()}D tensor.")
    checkSize(strides, begin, "Begin indices")
    checkSize(strides, end, "End indices")

    strides.apply1 { v =>
      require(v == 1, s"Unsupported strides, only support stride 1. but got strides: \n${strides}")
      v
    }

    val inputSize = inputs.size()
    getPositiveIndices(inputSize, begin, beginBuffer)
    getPositiveIndices(inputSize, end, endBuffer)

    var tmp: Tensor[D] = inputs
    var i = 0
    var currentDim = 1
    while (i < beginBuffer.nElement()) {
      if (maskValue(shrinkAxisMask, i)) {
        tmp = tmp.select(currentDim, beginBuffer.valueAt(i + 1))
      } else {
        val beginIndex =
          if (beginMask != 0 && maskValue(beginMask, i)) {
          1
        } else {
          beginBuffer.valueAt(i + 1)
        }
        val endIndex = if (endMask != 0 && maskValue(endMask, i)) {
          inputSize(i) + 1
        } else {
          endBuffer.valueAt(i + 1)
        }
        tmp = tmp.narrow(currentDim, beginIndex, endIndex - beginIndex)
        currentDim += 1
      }
      i += 1
    }

    if (tmp.dim() == 1 && tmp.size(1) == 1) tmp = Tensor.scalar[D](tmp.valueAt(1))
    output.resizeAs(tmp)
    output.copy(tmp)
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[D]): Table = {
    var tmp = gradInput[Tensor[D]](1).resizeAs(input[Tensor[D]](1)).zero()

    val inputSize = tmp.size()

    var i = 0
    var currentDim = 1
    while (i < beginBuffer.nElement()) {
      if (maskValue(shrinkAxisMask, i)) {
        tmp = tmp.select(currentDim, beginBuffer.valueAt(i + 1))
      } else {
        val beginIndex =
          if (beginMask != 0 && maskValue(beginMask, i)) {
            1
          } else {
            beginBuffer.valueAt(i + 1)
          }
        val endIndex = if (endMask != 0 && maskValue(endMask, i)) {
          inputSize(i) + 1
        } else {
          endBuffer.valueAt(i + 1)
        }
        tmp = tmp.narrow(currentDim, beginIndex, endIndex - beginIndex)
        currentDim += 1
      }
      i += 1
    }
    tmp.copy(gradOutput)

    gradInput
  }

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}

private[bigdl] object StridedSlice {
  def apply[T: ClassTag, D: ClassTag](
      beginMask: Int = 0,
      endMask: Int = 0,
      ellipsisMask: Int = 0,
      newAxisMask: Int = 0,
      shrinkAxisMask: Int = 0,
      startFromZero: Boolean = false)(
      implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]): StridedSlice[T, D] = {
    new StridedSlice[T, D](beginMask, endMask, ellipsisMask,
      newAxisMask, shrinkAxisMask, startFromZero)
  }
}

