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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Engine

import scala.concurrent.Future
import scala.math.exp
import scala.reflect.ClassTag

/**
 * The [[LogSoftMax]] module applies a LogSoftMax transformation to the input data
 * which is defined as:
 * f_i(x) = log(1 / a exp(x_i))
 * where a = sum_j[exp(x_j)]
 *
 * The input given in `forward(input)` must be either
 * a vector (1D tensor) or matrix (2D tensor).
 */
@SerialVersionUID(- 2954501946670913825L)
class LogSoftMax[T: ClassTag](
  implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  @transient
  private var results: Array[Future[Unit]] = null

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 1 || input.dim() == 2,
      "LogSoftMax: " + ErrorInfo.constrainInputAsVectorOrBatch)
    output.resizeAs(input)
    val (nframe, dim) =
      if (input.nDimension() == 1) (1, input.size(1)) else (input.size(1), input.size(2))

    if (nframe == 1) {
      updateOutputFrame(input, output)
    } else {
      if (results == null || results.length != nframe) {
        results = new Array[Future[Unit]](nframe)
      }
      var t = 1
      while (t <= nframe) {
        val _t = t
        results(_t - 1) = Engine.model.invoke(() => {
          updateOutputFrame(input.select(1, _t), output.select(1, _t))
        })
        t += 1
      }
      Engine.model.sync(results)
    }

    output
  }

  private def updateOutputFrame(in: Tensor[T], out: Tensor[T]): Unit = {
    var logsum = ev.fromType[Int](0)
    val maxInput = in.max()
    in.apply1(v => {
      logsum = ev.plus(logsum, ev.exp(ev.minus(v, maxInput))); v
    })
    logsum = ev.plus(maxInput, ev.log(logsum))

    out.map(in, (outData, inData) => {
      ev.minus(inData, logsum)
    })
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(output.nDimension() == 1 || output.nDimension() == 2, "vector or matrix expected")
    require(gradOutput.dim() == input.dim(), "LogSoftMax: input and gradOutput shapes do not " +
      "match, input_dim: " + input.dim() + ", gradOutput_dim: " + gradOutput.dim())
    gradInput.resizeAs(input)
    val (nframe, dim) =
      if (output.nDimension() == 1) (1, output.size(1)) else (output.size(1), output.size(2))

    if (results == null || results.length != nframe) {
      results = new Array[Future[Unit]](nframe)
    }

    var t = 1
    while (t <= nframe) {
      val _t = t
      results(_t - 1) = Engine.model.invoke(() => {
        var sum = 0.0
        var d = 1
        while (d <= dim) {
          if (gradOutput.dim() == 1) {
            sum += ev.toType[Double](gradOutput.valueAt(d))
          } else {
            sum += ev.toType[Double](gradOutput.valueAt(_t, d))
          }
          d += 1
        }

        d = 1
        while (d <= dim) {
          if (input.dim() == 1) {
            gradInput.setValue(d, ev.minus(gradOutput.valueAt(d),
              ev.times(ev.exp(output.valueAt(d)), ev.fromType[Double](sum))))
          } else {
            gradInput.setValue(_t, d, ev.minus(gradOutput.valueAt(_t, d),
              ev.times(ev.exp(output.valueAt(_t, d)), ev.fromType[Double](sum))))
          }
          d += 1
        }
      })
      t += 1
    }
    Engine.model.sync(results)

    gradInput
  }
}

object LogSoftMax {

  def apply[@specialized(Float, Double) T: ClassTag]()
      (implicit ev: TensorNumeric[T]) : LogSoftMax[T] = {
    new LogSoftMax[T]()
  }
  private val A0 = 1.0
  private val A1 = 0.125
  private val A2 = 0.0078125
  private val A3 = 0.00032552083
  private val A4 = 1.0172526e-5

  def expMinusApprox(x: Double): Double = {
    if (x < 0) {
      return exp(-x)
    } else {
      var y = 0.0
      if (x < 13.0) {
        y = A0 + x * (A1 + x * (A2 + x * (A3 + x * A4)))
        y *= y
        y *= y
        y *= y
        y = 1 / y
        return y
      }
    }

    return 0.0
  }
}

