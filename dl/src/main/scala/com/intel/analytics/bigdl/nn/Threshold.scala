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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._

import scala.collection.mutable.ArrayBuffer
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.reflect.ClassTag
import com.intel.analytics.bigdl.utils.Engine

class Threshold[@specialized(Float, Double) T: ClassTag](
  th: Double = 1e-6, v: Double = 0.0, ip: Boolean = false)(
  implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  var threshold = th
  var value = v
  var inPlace = ip
  validateParameters()

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.isContiguous())
    validateParameters()

    val taskSize = input.nElement() / Engine.coresNum
    var extraTaskSize = input.nElement() % Engine.coresNum
    var allocated = 0
    val tasks = new ArrayBuffer[(Int, Int)]()
    while (allocated < input.nElement()) {
      val end = math.min(input.nElement(), if (extraTaskSize > 0) {
        extraTaskSize -= 1
        allocated + taskSize + 1
      } else {
        allocated + taskSize
      })
      tasks += ((allocated, end))
      allocated = end
    }

    val taskArray = tasks.toArray
    val results = new Array[Future[Unit]](taskArray.length)

    if (inPlace) {
      output = input
      ev.getType() match {
        case "Double" =>
          val inputDouble = input.asInstanceOf[Tensor[Double]]
          val inputData = inputDouble.storage().array()
          val inputOffset = inputDouble.storageOffset() - 1

          var t = 0
          while (t < taskArray.length) {
            val _t = t
            results(_t) = Future {
              var i = taskArray(_t)._1
              while (i < taskArray(_t)._2) {
                inputData(inputOffset + i) =
                  if (inputData(inputOffset + i) <= threshold) {
                    value
                  } else {
                    inputData(inputOffset + i)
                  }
                i += 1
              }
            }(Engine.getInstance())
            t += 1
          }
        case "Float" =>
          val inputDouble = input.asInstanceOf[Tensor[Float]]
          val inputData = inputDouble.storage().array()
          val inputOffset = inputDouble.storageOffset() - 1

          val valueFloat = value.toFloat
          var t = 0
          while (t < taskArray.length) {
            val _t = t
            results(_t) = Future {
              var i = taskArray(_t)._1
              while (i < taskArray(_t)._2) {
                inputData(inputOffset + i) =
                  if (inputData(inputOffset + i) <= threshold) {
                    valueFloat
                  } else {
                    inputData(inputOffset + i)
                  }
                i += 1
              }
            }(Engine.getInstance())
            t += 1
          }
        case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
      }
      input
    }
    else {
      ev.getType() match {
        case "Double" =>
          output.asInstanceOf[Tensor[Double]].resizeAs(input.asInstanceOf[Tensor[Double]])

          val inputDouble = input.asInstanceOf[Tensor[Double]]
          val inputData = inputDouble.storage().array()
          val inputOffset = inputDouble.storageOffset() - 1
          val outputDouble = output.asInstanceOf[Tensor[Double]]
          val outputData = outputDouble.storage().array()
          val outputOffset = outputDouble.storageOffset() - 1

          var t = 0
          while (t < taskArray.length) {
            val _t = t
            results(_t) = Future {
              var i = taskArray(_t)._1
              while (i < taskArray(_t)._2) {
                outputData(outputOffset + i) =
                  if (inputData(inputOffset + i) > threshold) {
                    inputData(inputOffset + i)
                  } else {
                    value
                  }
                i += 1
              }
            }(Engine.getInstance())
            t += 1
          }
        case "Float" =>
          output.asInstanceOf[Tensor[Float]].resizeAs(input.asInstanceOf[Tensor[Float]])

          val inputFloat = input.asInstanceOf[Tensor[Float]]
          val inputData = inputFloat.storage().array()
          val inputOffset = inputFloat.storageOffset() - 1
          val outputFloat = output.asInstanceOf[Tensor[Float]]
          val outputData = outputFloat.storage().array()
          val outputOffset = outputFloat.storageOffset() - 1

          val valueFloat = value.toFloat
          var t = 0
          while (t < taskArray.length) {
            val _t = t
            results(_t) = Future {
              var i = taskArray(_t)._1
              while (i < taskArray(_t)._2) {
                outputData(outputOffset + i) =
                  if (inputData(inputOffset + i) > threshold) {
                    inputData(inputOffset + i)
                  } else {
                    valueFloat
                  }
                i += 1
              }
            }(Engine.getInstance())
            t += 1
          }
        case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
      }
    }
    var r = 0
    while (r < results.length) {
      Await.result(results(r), Duration.Inf)
      r += 1
    }
    output
  }

  private def updateGradInputNoContinuous(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    validateParameters()
    if (inPlace) {
      gradInput = gradOutput
      ev.getType() match {
        case "Double" =>
          gradInput.asInstanceOf[Tensor[Double]].map(input.asInstanceOf[Tensor[Double]], (g, i) =>
            if (i <= threshold) 0 else g)
        case "Float" =>
          gradInput.asInstanceOf[Tensor[Float]].map(input.asInstanceOf[Tensor[Float]], (g, i) =>
            if (i <= threshold) 0 else g)
        case _ =>
          throw new UnsupportedOperationException(s"Only Float/Double supported")
      }
    }
    else {
      gradInput.resizeAs(gradOutput)
      gradInput.copy(gradOutput)
      ev.getType() match {
        case "Double" =>
          gradInput.asInstanceOf[Tensor[Double]].map(input.asInstanceOf[Tensor[Double]], (g, i) =>
            if (i > threshold) g else 0)
        case "Float" =>
          gradInput.asInstanceOf[Tensor[Float]].map(input.asInstanceOf[Tensor[Float]], (g, i) =>
            if (i > threshold) g else 0)
        case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
      }
    }
    gradInput
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    validateParameters()

    var i = 1
    while (i <= input.nDimension()) {
      if (input.stride(i) != gradOutput.stride(i)) {
        return updateGradInputNoContinuous(input, gradOutput)
      }
      i += 1
    }

    val taskSize = gradOutput.nElement() / Engine.coresNum
    var extraTaskSize = gradOutput.nElement() % Engine.coresNum
    var allocated = 0
    val tasks = new ArrayBuffer[(Int, Int)]()
    while (allocated < gradOutput.nElement()) {
      val end = math.min(gradOutput.nElement(), if (extraTaskSize > 0) {
        extraTaskSize -= 1
        allocated + taskSize + 1
      } else {
        allocated + taskSize
      })
      tasks += ((allocated, end))
      allocated = end
    }

    val taskArray = tasks.toArray
    val results = new Array[Future[Unit]](taskArray.length)

    if (inPlace) {
      gradInput = gradOutput
      ev.getType() match {
        case "Double" =>
          val gradInputDouble = gradInput.asInstanceOf[Tensor[Double]]
          val inputDouble = input.asInstanceOf[Tensor[Double]]
          val gradInputData = gradInputDouble.storage().array()
          val gradInputOffset = gradInputDouble.storageOffset() - 1
          val inputData = inputDouble.storage().array()
          val inputOffset = inputDouble.storageOffset() - 1

          var t = 0
          while (t < taskArray.length) {
            val _t = t
            results(_t) = Future {
              var i = taskArray(_t)._1
              while (i < taskArray(_t)._2) {
                gradInputData(gradInputOffset + i) =
                  if (inputData(inputOffset + i) <= threshold) {
                    0.0
                  } else {
                    gradInputData(gradInputOffset + i)
                  }
                i += 1
              }
            }(Engine.getInstance())
            t += 1
          }

        case "Float" =>
          val gradInputFloat = gradInput.asInstanceOf[Tensor[Float]]
          val inputFloat = input.asInstanceOf[Tensor[Float]]
          val gradInputData = gradInputFloat.storage().array()
          val gradInputOffset = gradInputFloat.storageOffset() - 1
          val inputData = inputFloat.storage().array()
          val inputOffset = inputFloat.storageOffset() - 1

          var t = 0
          while (t < taskArray.length) {
            val _t = t
            results(_t) = Future {
              var i = taskArray(_t)._1
              while (i < taskArray(_t)._2) {
                gradInputData(gradInputOffset + i) =
                  if (inputData(inputOffset + i) <= threshold) {
                    0.0f
                  } else {
                    gradInputData(gradInputOffset + i)
                  }
                i += 1
              }
            }(Engine.getInstance())
            t += 1
          }
        case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
      }
    }
    else {
      ev.getType() match {
        case "Double" =>
          gradInput.asInstanceOf[Tensor[Double]].resizeAs(gradOutput.asInstanceOf[Tensor[Double]])
          gradInput.asInstanceOf[Tensor[Double]].copy(gradOutput.asInstanceOf[Tensor[Double]])
          val gradInputDouble = gradInput.asInstanceOf[Tensor[Double]]
          val inputDouble = input.asInstanceOf[Tensor[Double]]
          val gradInputData = gradInputDouble.storage().array()
          val gradInputOffset = gradInputDouble.storageOffset() - 1
          val inputData = inputDouble.storage().array()
          val inputOffset = inputDouble.storageOffset() - 1

          var t = 0
          while (t < taskArray.length) {
            val _t = t
            results(_t) = Future {
              var i = taskArray(_t)._1
              while (i < taskArray(_t)._2) {
                gradInputData(gradInputOffset + i) =
                  if (inputData(inputOffset + i) <= threshold) {
                    0.0
                  } else {
                    gradInputData(gradInputOffset + i)
                  }
                i += 1
              }
            }(Engine.getInstance())
            t += 1
          }
        case "Float" =>
          gradInput.asInstanceOf[Tensor[Float]].resizeAs(gradOutput.asInstanceOf[Tensor[Float]])
          gradInput.asInstanceOf[Tensor[Float]].copy(gradOutput.asInstanceOf[Tensor[Float]])
          val gradInputFloat = gradInput.asInstanceOf[Tensor[Float]]
          val inputFloat = input.asInstanceOf[Tensor[Float]]
          val gradInputData = gradInputFloat.storage().array()
          val gradInputOffset = gradInputFloat.storageOffset() - 1
          val inputData = inputFloat.storage().array()
          val inputOffset = inputFloat.storageOffset() - 1

          var t = 0
          while (t < taskArray.length) {
            val _t = t
            results(_t) = Future {
              var i = taskArray(_t)._1
              while (i < taskArray(_t)._2) {
                gradInputData(gradInputOffset + i) =
                  if (inputData(inputOffset + i) <= threshold) {
                    0.0f
                  } else {
                    gradInputData(gradInputOffset + i)
                  }
                i += 1
              }
            }(Engine.getInstance())
            t += 1
          }
        case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
      }
    }

    var r = 0
    while (r < results.length) {
      Await.result(results(r), Duration.Inf)
      r += 1
    }
    gradInput
  }

  def validateParameters(): Unit = {
    if (inPlace) {
      require(value <= threshold, "in-place processing requires value (" +
        value + "') not exceed threshold (" + threshold + ")")
    }
  }

  override def equals(obj: Any): Boolean = {

    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[Threshold[T]]) {
      return false
    }
    val other = obj.asInstanceOf[Threshold[T]]
    if (this.eq(other)) {
      return true
    }
    threshold == other.threshold && value == other.value && inPlace == other.inPlace
  }

  override def hashCode() : Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + threshold.hashCode()
    hash = hash * seed + value.hashCode()
    hash = hash * seed + inPlace.hashCode()

    hash
  }

  override def toString(): String = {
    s"nn.Threshold($th, $v)"
  }
}

object Threshold {
  def apply[@specialized(Float, Double) T: ClassTag](
      th: Double = 1e-6,
      v: Double = 0.0,
      ip: Boolean = false)(implicit ev: TensorNumeric[T]) : Threshold[T] = {
    new Threshold[T](th, v, ip)
  }
}
