/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

class Recurrent[T : ClassTag] (
  hiddenSize: Int = 3,
  bpttTruncate: Int = 2)
  (implicit ev: TensorNumeric[T]) extends Container[Tensor[T], Tensor[T], T] {

  var hidden: Activity = null
  var gradHidden: Activity = null
  val inputs = T()
  val batchDim = 1
  val timeDim = 2
  val hidDim = 1
  val inputDim = 2
  var (batchSize, times) = (0, 0)

  override def reset(): Unit = {
    modules(0).reset()
    if (modules(0).isInstanceOf[RnnCell[T]]) {
      hidden = Tensor[T]()
      gradHidden = Tensor[T]()
    } else if (modules(0).isInstanceOf[LSTMCell[T]]
      || modules(0).isInstanceOf[FastLSTMCell[T]]) {
      hidden = T(Tensor[T](), Tensor[T]())
      gradHidden = T(Tensor[T](), Tensor[T]())
    } else {
      throw new IllegalArgumentException("Cell not implemented")
    }
  }

  /**
   * Clone N models; N depends on the time dimension of the input
   * @param size
   */
  private def extend(size: Int): Unit = {
    require(modules != null && modules.length >= 1,
      "Recurrent extend: should contain at least one module")
    var t = modules.length
    var flag = false
    if (t < size) {
      flag = true
    }
    while (t < size) {
      super.add(modules(0).cloneModule())
      t += 1
    }
    if (flag) {
      share()
    }
  }

  /**
   * Sharing weights, gradWeights across all the modules in time dim
   */
  def share(): Unit = {
    val params = modules(0).parameters()
    var i = 1
    while (i < modules.length) {
      val curParams = modules(i).parameters()
      var j = 0
      while (j < curParams._1.length) {
        curParams._1(j).set(params._1(j))
        j += 1
      }
      j = 0
      while (j < curParams._2.length) {
        curParams._2(j).set(params._2(j))
        j += 1
      }
      i += 1
    }
  }

  private def hidInit(hid: Activity, size: Array[Int]): Unit = {
    if (hid.isInstanceOf[Tensor[T]]) {
      hid.asInstanceOf[Tensor[T]].resize(size)
    } else {
      var j = 1
      while (j <= hid.asInstanceOf[Table].length) {
        hid.asInstanceOf[Table](j).asInstanceOf[Tensor[T]].resize(size)
        j += 1
      }
    }
  }

  /**
   * cloning inputs for reusing in backward
   * @param tbl
   * @return
   */
  private def shallowCopy(tbl: Table): Table = {
    val ntbl = T()
    var i = 1
    while (i <= tbl.length) {
      if (tbl(i).isInstanceOf[Tensor[T]]) {
        ntbl(i) = Tensor[T]().resizeAs(tbl(i)).copy(tbl(i))
      } else if (tbl(i).isInstanceOf[Table]) {
        var j = 1
        val table = tbl[Table](i)
        while (j <= table.length) {
          ntbl(i + j - 1) = Tensor[T]().resizeAs(table(j)).copy(table(j))
          j += 1
        }
        i += j - 1
      }
      i += 1
    }
    ntbl
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim == 3,
      "Recurrent: input should be a 3D Tensor, e.g [batch, times, nDim], " +
        s"current input.dim = ${input.dim}")

    batchSize = input.size(batchDim)
    times = input.size(timeDim)
    extend(times)

    output.resize(Array(batchSize, times, hiddenSize))

    hidInit(hidden, Array(batchSize, hiddenSize))
    var currentOutput = T(hidden)
    var i = 1
    while (i <= times) {
      currentOutput(inputDim) = input.select(timeDim, i)
      inputs(i) = shallowCopy(currentOutput)
      currentOutput = modules(i - 1).updateOutput(currentOutput).toTable
      output.select(timeDim, i).copy(currentOutput(inputDim))
      i += 1
    }
    output
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T],
    scale: Double = 1.0): Unit = {
    var currentGradOutput = T(Tensor[T]().resizeAs(gradOutput.select(timeDim, 1)))
    var i = times
    while (i >= 1) {
      currentGradOutput(inputDim) = gradOutput.select(timeDim, i)
      modules(i - 1).accGradParameters(inputs(i), currentGradOutput, scale)
      currentGradOutput = modules(i - 1).gradInput.toTable
      i -= 1
    }
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input)
    hidInit(gradHidden, Array(batchSize, hiddenSize))
    var currentGradOutput = T(gradHidden)
    var i = times
    while (i >= 1) {
      currentGradOutput(inputDim) = gradOutput.select(timeDim, i)
      currentGradOutput =
        modules(i - 1).updateGradInput(inputs(i), currentGradOutput).toTable
      gradInput.select(timeDim, i).copy(currentGradOutput(inputDim))
      i -= 1
    }
    gradInput
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    val params = modules(0).parameters()
    (params._1, params._2)
  }

  override def updateParameters(learningRate: T): Unit = {
    modules(0).updateParameters(learningRate)
  }

  override def zeroGradParameters(): Unit = {
    modules(0).zeroGradParameters()
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[Recurrent[T]]

  override def equals(other: Any): Boolean = other match {
    case that: Recurrent[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        hidden == that.hidden &&
        batchSize == that.batchSize &&
        times == that.times
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), hidden, batchSize, times)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }

  override def toString(): String = {
    val str = "nn.Recurrent"
    str
  }
}

object Recurrent {
  def apply[@specialized(Float, Double) T: ClassTag](
    hiddenSize: Int = 3,
    bpttTruncate: Int = 2)
    (implicit ev: TensorNumeric[T]) : Recurrent[T] = {
    new Recurrent[T](hiddenSize, bpttTruncate)
  }
}
