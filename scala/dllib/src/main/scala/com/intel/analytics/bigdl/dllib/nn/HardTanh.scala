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
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, TensorModule}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.Engine

import scala.concurrent.Future
import scala.reflect.ClassTag

/**
 * Applies HardTanh to each element of input, HardTanh is defined:
 *          ⎧  maxValue, if x > maxValue
 *   f(x) = ⎨  minValue, if x < minValue
 *          ⎩  x, otherwise
 *
 * @param minValue minValue in f(x), default is -1.
 * @param maxValue maxValue in f(x), default is 1.
 * @param inplace inplace model.
 */
@SerialVersionUID(- 8953866090802444183L)
class HardTanh[T: ClassTag, D: ClassTag](
  val minValue: Double = -1,
  val maxValue: Double = 1,
  val inplace: Boolean = false
)(implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends AbstractModule[Tensor[D], Tensor[D], T] {
  require(maxValue > minValue, "maxValue must be larger than minValue, " +
    s"maxValue ${maxValue}, " +
    s"minValue ${minValue}")

  output = Tensor[D]()
  gradInput = Tensor[D]()

  val min = ev2.fromType[Double](minValue)
  val max = ev2.fromType[Double](maxValue)

  override def updateOutput(input: Tensor[D]): Tensor[D] = {
    if (inplace) {
      output.set(input)
    }
    else {
      output.resizeAs(input)
    }

    if (input.dim() == 1 || !input.isContiguous() || !output.isContiguous()) {
      if (inplace) {
        val func = new TensorFunc2[D] {
          override def apply(data: Array[D], index: Int): Unit = {
            if (ev2.isGreater(min, data(index))) {
              data(index) = ev2.fromType[Double](minValue)
            } else if (ev2.isGreater(data(index), max)) {
              data(index) = ev2.fromType[Double](maxValue)
            }
          }
        }
        DenseTensorApply.apply1[D](input, func)
      } else {
        val func2 = new TensorFunc4[D] {
          override def apply(data1: Array[D], index1: Int, data2: Array[D], index2: Int): Unit = {
            if (ev2.isGreater(min, data2(index2))) {
              data1(index1) = min
            } else if (ev2.isGreaterEq(max, data2(index2))) {
              data1(index1) = data2(index2)
            } else {
              data1(index1) = max
            }
          }
        }
        DenseTensorApply.apply2[D](output, input, func2)
      }
    } else {
      val inputData = input.storage().array()
      val inputOffset = input.storageOffset() - 1
      val outputData = output.storage().array()
      val outputOffset = input.storageOffset() - 1

      var i = 0
      if (inplace) {
        while (i < input.nElement()) {
          if (ev2.isGreater(min, inputData(i + inputOffset))) {
            inputData.update(i + inputOffset, min)
          } else if (ev2.isGreater(inputData(i + inputOffset), max)) {
            inputData.update(i + inputOffset, max)
          }
          i += 1
        }
      } else {
        while (i < input.nElement()) {
          if (ev2.isGreater(min, inputData(i + inputOffset))) {
            outputData.update(i + outputOffset, min)
          } else if (ev2.isGreaterEq(max, inputData(i + inputOffset))) {
            outputData.update(i + outputOffset, inputData(i + inputOffset))
          } else {
            outputData.update(i + outputOffset, max)
          }
          i += 1
        }
      }
    }

    output
  }



  override def updateGradInput(input: Tensor[D], gradOutput: Tensor[D]): Tensor[D] = {
    require(input.nElement() == gradOutput.nElement(),
      s"the number of input element (${input.nElement()}) " +
        s"should equal the number of " +
        s"gradOutput element (${gradOutput.nElement()}), ")
    if (inplace) {
      gradInput.set(gradOutput)
    } else {
      gradInput.resizeAs(input)
    }

    if (input.dim() == 1 || !input.isContiguous() || !gradOutput.isContiguous()
      || !gradInput.isContiguous()) {
      if (inplace) {
        val func = new TensorFunc4[D] {
          override def apply(data1: Array[D], index1: Int, data2: Array[D], index2: Int): Unit = {
            if (ev2.isGreaterEq(min, data2(index2)) || ev2.isGreaterEq(data2(index2), max)) {
              data1(index1) = ev2.fromType[Double](0)
            }
          }
        }
        DenseTensorApply.apply2[D](gradOutput, input, func)
      } else {
        val func = new TensorFunc6[D] {
          override def apply(data1: Array[D], offset1: Int, data2: Array[D],
            offset2: Int, data3: Array[D], offset3: Int): Unit = {
            if (ev2.isGreaterEq(min, data3(offset3)) || ev2.isGreaterEq(data3(offset3), max)) {
              data1(offset1) = ev2.fromType[Double](0)
            } else {
              data1(offset1) = data2(offset2)
            }
          }
        }
        DenseTensorApply.apply3[D](gradInput, gradOutput, input, func)
      }
    } else {
      val inputData = input.storage().array()
      val inputOffset = input.storageOffset() - 1
      val gradOutputData = gradOutput.storage().array()
      val gradOutputOffset = gradOutput.storageOffset() - 1
      val gradInputData = gradInput.storage().array()
      val gradInputOffset = gradInput.storageOffset() - 1

      var i = 0
      if (inplace) {
        while (i < input.nElement()) {
          if (ev2.isGreaterEq(min, inputData(i + inputOffset))
            || ev2.isGreaterEq(inputData(i + inputOffset), max)) {
            gradInputData.update(i + gradInputOffset, ev2.fromType[Double](0))
          }
          i += 1
        }
      } else {
        while (i < input.nElement()) {
          if (ev2.isGreaterEq(min, inputData(i + inputOffset))
            || ev2.isGreaterEq(inputData(i + inputOffset), max)) {
            gradInputData.update(i + gradInputOffset, ev2.fromType[Double](0))
          } else {
            gradInputData.update(i + gradInputOffset, gradOutputData(i + gradOutputOffset))
          }
          i += 1
        }
      }
    }

    gradInput
  }

  override def toString: String = {
    s"nn.HardTanh"
  }

  override def clearState(): this.type = {
    if (!inplace) {
      super.clearState()
    }
    this
  }

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}

object HardTanh {
  def apply[@specialized(Float, Double) T: ClassTag, D: ClassTag](
      minValue: Double = -1,
      maxValue: Double = 1,
      inplace: Boolean = false)
      (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]): HardTanh[T, D] = {
    new HardTanh[T, D](minValue, maxValue, inplace)
  }
}
