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

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.Engine

import scala.collection.mutable.ArrayBuffer
import scala.concurrent.Future
import scala.reflect.ClassTag

/**
 * Threshold input Tensor.
 * If values in the Tensor smaller than th, then replace it with v
 *
 * @param th the threshold to compare with
 * @param ip inplace mode
 */

@SerialVersionUID(4932292249027276581L)
class BinaryThreshold[T: ClassTag](
  val th: Double = 1e-6, val ip: Boolean = false)(
  implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  var threshold = th
  var inPlace = ip

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.isContiguous())

    val taskSize = input.nElement() / Engine.model.getPoolSize
    var extraTaskSize = input.nElement() % Engine.model.getPoolSize
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
        case DoubleType =>
          val inputDouble = input.asInstanceOf[Tensor[Double]]
          val inputData = inputDouble.storage().array()
          val inputOffset = inputDouble.storageOffset() - 1

          var t = 0
          while (t < taskArray.length) {
            val _t = t
            results(_t) = Engine.model.invoke(() => {
              var i = taskArray(_t)._1
              while (i < taskArray(_t)._2) {
                inputData(inputOffset + i) =
                  if (inputData(inputOffset + i) <= threshold) {
                    0.0
                  } else {
                    1.0
                  }
                i += 1
              }
            })
            t += 1
          }
        case FloatType =>
          val inputDouble = input.asInstanceOf[Tensor[Float]]
          val inputData = inputDouble.storage().array()
          val inputOffset = inputDouble.storageOffset() - 1

          val valueFloat = 0.0f
          var t = 0
          while (t < taskArray.length) {
            val _t = t
            results(_t) = Engine.model.invoke(() => {
              var i = taskArray(_t)._1
              while (i < taskArray(_t)._2) {
                inputData(inputOffset + i) =
                  if (inputData(inputOffset + i) <= threshold) {
                    valueFloat
                  } else {
                    1.0f
                  }
                i += 1
              }
            })
            t += 1
          }
        case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
      }
      input
    }
    else {
      ev.getType() match {
        case DoubleType =>
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
            results(_t) = Engine.model.invoke(() => {
              var i = taskArray(_t)._1
              while (i < taskArray(_t)._2) {
                outputData(outputOffset + i) =
                  if (inputData(inputOffset + i) > threshold) {
                    1.0
                  } else {
                    0.0
                  }
                i += 1
              }
            })
            t += 1
          }
        case FloatType =>
          output.asInstanceOf[Tensor[Float]].resizeAs(input.asInstanceOf[Tensor[Float]])

          val inputFloat = input.asInstanceOf[Tensor[Float]]
          val inputData = inputFloat.storage().array()
          val inputOffset = inputFloat.storageOffset() - 1
          val outputFloat = output.asInstanceOf[Tensor[Float]]
          val outputData = outputFloat.storage().array()
          val outputOffset = outputFloat.storageOffset() - 1

          var t = 0
          while (t < taskArray.length) {
            val _t = t
            results(_t) = Engine.model.invoke(() => {
              var i = taskArray(_t)._1
              while (i < taskArray(_t)._2) {
                outputData(outputOffset + i) =
                  if (inputData(inputOffset + i) > threshold) {
                    1.0f
                  } else {
                    0.0f
                  }
                i += 1
              }
            })
            t += 1
          }
        case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
      }
    }
    Engine.model.sync(results)
    output
  }

  private def updateGradInputNoContinuous(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (inPlace) {
      gradInput = gradOutput
      ev.getType() match {
        case DoubleType =>
          gradInput.asInstanceOf[Tensor[Double]].map(input.asInstanceOf[Tensor[Double]], (g, i) =>
            if (i <= threshold) 0 else g)
        case FloatType =>
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
        case DoubleType =>
          gradInput.asInstanceOf[Tensor[Double]].map(input.asInstanceOf[Tensor[Double]], (g, i) =>
            if (i > threshold) g else 0)
        case FloatType =>
          gradInput.asInstanceOf[Tensor[Float]].map(input.asInstanceOf[Tensor[Float]], (g, i) =>
            if (i > threshold) g else 0)
        case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
      }
    }
    gradInput
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {

    var i = 1
    while (i <= input.nDimension()) {
      if (input.stride(i) != gradOutput.stride(i)) {
        return updateGradInputNoContinuous(input, gradOutput)
      }
      i += 1
    }

    val taskSize = gradOutput.nElement() / Engine.model.getPoolSize
    var extraTaskSize = gradOutput.nElement() % Engine.model.getPoolSize
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
        case DoubleType =>
          val gradInputDouble = gradInput.asInstanceOf[Tensor[Double]]
          val inputDouble = input.asInstanceOf[Tensor[Double]]
          val gradInputData = gradInputDouble.storage().array()
          val gradInputOffset = gradInputDouble.storageOffset() - 1
          val inputData = inputDouble.storage().array()
          val inputOffset = inputDouble.storageOffset() - 1

          var t = 0
          while (t < taskArray.length) {
            val _t = t
            results(_t) = Engine.model.invoke(() => {
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
            })
            t += 1
          }

        case FloatType =>
          val gradInputFloat = gradInput.asInstanceOf[Tensor[Float]]
          val inputFloat = input.asInstanceOf[Tensor[Float]]
          val gradInputData = gradInputFloat.storage().array()
          val gradInputOffset = gradInputFloat.storageOffset() - 1
          val inputData = inputFloat.storage().array()
          val inputOffset = inputFloat.storageOffset() - 1

          var t = 0
          while (t < taskArray.length) {
            val _t = t
            results(_t) = Engine.model.invoke(() => {
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
            })
            t += 1
          }
        case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
      }
    }
    else {
      ev.getType() match {
        case DoubleType =>
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
            results(_t) = Engine.model.invoke(() => {
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
            })
            t += 1
          }
        case FloatType =>
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
            results(_t) = Engine.model.invoke(() => {
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
            })
            t += 1
          }
        case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
      }
    }

    Engine.model.sync(results)
    gradInput
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
    threshold == other.threshold && inPlace == other.inPlace
  }

  override def hashCode() : Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + threshold.hashCode()
    hash = hash * seed + inPlace.hashCode()

    hash
  }

  override def toString(): String = {
    s"${getPrintName}($th)"
  }
}

object BinaryThreshold {
  def apply[@specialized(Float, Double) T: ClassTag](
    th: Double = 1e-6,
    ip: Boolean = false)(implicit ev: TensorNumeric[T]) : BinaryThreshold[T] = {
    new BinaryThreshold[T](th, ip)
  }
}
