/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.tensor._
import com.intel.analytics.sparkdl.utils.Engine

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.reflect.ClassTag

class HardTanh[T: ClassTag](
  val minValue: Double = -1,
  val maxValue: Double = 1,
  val inplace: Boolean = false
)(implicit ev: TensorNumeric[T])
  extends TensorModule[T] {
  require(maxValue > minValue, "maxValue must be larger than minValue")
  @transient
  private var tasks: Array[Future[Unit]] = null

  val min = ev.fromType[Double](minValue)
  val max = ev.fromType[Double](maxValue)

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (inplace) {
      output.set(input)
    }
    else {
      output.resizeAs(input)
    }

    if (input.dim() == 1 || !input.isContiguous() || !output.isContiguous()) {
      if (inplace) {
        val func = new TensorFunc2[T] {
          override def apply(data: Array[T], index: Int): Unit = {
            if (ev.isGreater(min, data(index))) {
              data(index) = ev.fromType[Double](minValue)
            } else if (ev.isGreater(data(index), max)) {
              data(index) = ev.fromType[Double](maxValue)
            }
          }
        }
        DenseTensorApply.apply1[T](input, func)
      } else {
        val func2 = new TensorFunc4[T] {
          override def apply(data1: Array[T], index1: Int, data2: Array[T], index2: Int): Unit = {
            if (ev.isGreater(min, data2(index2))) {
              data1(index1) = min
            } else if (ev.isGreaterEq(max, data2(index2))) {
              data1(index1) = data2(index2)
            } else {
              data1(index1) = max
            }
          }
        }
        DenseTensorApply.apply2[T](output, input, func2)
      }
    } else {
      val inputData = input.storage().array()
      val inputOffset = input.storageOffset() - 1
      val outputData = output.storage().array()
      val outputOffset = input.storageOffset() - 1

      if (tasks == null || tasks.length != inputData.length) {
        tasks = new Array[Future[Unit]](inputData.length)
      }

      var i = 0
      if (inplace) {
        while (i < input.nElement()) {
          val _i = i
          tasks(_i) = Future {
            if (ev.isGreater(min, inputData(_i + inputOffset))) {
              inputData.update(_i + inputOffset, min)
            } else if (ev.isGreater(inputData(_i + inputOffset), max)) {
              inputData.update(_i + inputOffset, max)
            }
          }(Engine.getInstance())
          i += 1
        }
        i = 0
        while (i < input.nElement()) {
          Await.result(tasks(i), Duration.Inf)
          i += 1
        }
      } else {
        while (i < input.nElement()) {
          val _i = i
          tasks(_i) = Future {
            if (ev.isGreater(min, inputData(_i + inputOffset))) {
              outputData.update(_i + outputOffset, min)
            } else if (ev.isGreaterEq(max, inputData(_i + inputOffset))) {
              outputData.update(_i + outputOffset, inputData(_i + inputOffset))
            } else {
              outputData.update(_i + outputOffset, max)
            }
          }(Engine.getInstance())
          i += 1
        }
        i = 0
        while (i < input.nElement()) {
          Await.result(tasks(i), Duration.Inf)
          i += 1
        }
      }
    }

    output
  }



  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.nElement() == gradOutput.nElement(),
      "the number of input element should equal the number of gradOutput element")
    if (inplace) {
      gradInput.set(gradOutput)
    } else {
      gradInput.resizeAs(input)
    }

    if (input.dim() == 1 || !input.isContiguous() || !gradOutput.isContiguous()
      || !gradInput.isContiguous()) {
      if (inplace) {
        val func = new TensorFunc4[T] {
          override def apply(data1: Array[T], index1: Int, data2: Array[T], index2: Int): Unit = {
            if (ev.isGreaterEq(min, data2(index2)) || ev.isGreaterEq(data2(index2), max)) {
              data1(index1) = ev.fromType[Double](0)
            }
          }
        }
        DenseTensorApply.apply2[T](gradOutput, input, func)
      } else {
        val func = new TensorFunc6[T] {
          override def apply(data1: Array[T], offset1: Int, data2: Array[T],
            offset2: Int, data3: Array[T], offset3: Int): Unit = {
            if (ev.isGreaterEq(min, data3(offset3)) || ev.isGreaterEq(data3(offset3), max)) {
              data1(offset1) = ev.fromType[Double](0)
            } else {
              data1(offset1) = data2(offset2)
            }
          }
        }
        DenseTensorApply.apply3[T](gradInput, gradOutput, input, func)
      }
    } else {
      val inputData = input.storage().array()
      val inputOffset = input.storageOffset() - 1
      val gradOutputData = gradOutput.storage().array()
      val gradOutputOffset = gradOutput.storageOffset() - 1
      val gradInputData = gradInput.storage().array()
      val gradInputOffset = gradInput.storageOffset() - 1

      if (tasks == null || tasks.length != inputData.length) {
        tasks = new Array[Future[Unit]](inputData.length)
      }

      var i = 0
      if (inplace) {
        while (i < input.nElement()) {
          val _i = i
          tasks(_i) = Future {
            if (ev.isGreaterEq(min, inputData(_i + inputOffset))
              || ev.isGreaterEq(inputData(_i + inputOffset), max)) {
              gradInputData.update(_i + gradInputOffset, ev.fromType[Double](0))
            }
          }(Engine.getInstance())
          i += 1
        }
        i = 0
        while (i < input.nElement()) {
          Await.result(tasks(i), Duration.Inf)
          i += 1
        }
      } else {
        while (i < input.nElement()) {
          val _i = i
          tasks(_i) = Future {
            if (ev.isGreaterEq(min, inputData(_i + inputOffset))
              || ev.isGreaterEq(inputData(_i + inputOffset), max)) {
              gradInputData.update(_i + gradInputOffset, ev.fromType[Double](0))
            } else {
              gradInputData.update(_i + gradInputOffset, gradOutputData(_i + gradOutputOffset))
            }
          }(Engine.getInstance())
          i += 1
        }
        i = 0
        while (i < input.nElement()) {
          Await.result(tasks(i), Duration.Inf)
          i += 1
        }
      }
    }

    gradInput
  }

  override def toString: String = {
    s"nn.HardTanh"
  }
}
