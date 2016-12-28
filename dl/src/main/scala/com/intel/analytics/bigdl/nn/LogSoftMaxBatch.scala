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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Engine

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.math.exp
import scala.reflect.ClassTag


class LogSoftMaxBatch[T: ClassTag](
  implicit ev: TensorNumeric[T]) extends LogSoftMax[T] {
  @transient
  private var results: Array[Future[Unit]] = null

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 1 || input.dim() == 2
      || input.dim() == 3, "vector, matrix or batch matrix expected")
    output.resizeAs(input)
    if (input.dim() != 3) {
      super.updateOutput(input)
    } else {
      // output.resizeAs(input)
      val (batchSize, nframe, dim) = (input.size(1), input.size(2), input.size(3))

      if (results == null || results.length != batchSize) {
        results = new Array[Future[Unit]](batchSize)
      }

      var t = 1
      while (t <= batchSize) {
        val _t = t
        results(_t - 1) = Engine.model.invoke(() => {
          updateOutputFrame(input.select(1, _t), output.select(1, _t))
        })
        t += 1
      }
      Engine.model.sync(results)

      output
    }
  }

  private def updateOutputFrame(in: Tensor[T], out: Tensor[T]): Unit = {
    var i = 1
    val nframe = in.size(1)
    while (i <= nframe) {
      var logsum = ev.fromType[Int](0)
      val maxInput = in(i).max()
      in(i).apply1(v => {
        logsum = ev.plus(logsum, ev.exp(ev.minus(v, maxInput)));
        v
      })
      logsum = ev.plus(maxInput, ev.log(logsum))

      out(i).map(in(i), (outData, inData) => {
        ev.minus(inData, logsum)
      })
      i += 1
    }
  }

  private def updateGradOutputFrame(in: Tensor[T], gradOut: Tensor[T],
                                   gradInputT: Tensor[T], outputT: Tensor[T]): Unit = {
    val (nframe, dim) = (in.size(1), in.size(2))
    var i = 1
    while (i <= nframe) {
      var sum = 0.0
      var d = 1
      while (d <= dim) {
        sum += ev.toType[Double](gradOut.valueAt(i, d))
        d += 1
      }

      d = 1
      while (d <= dim) {
        gradInputT.setValue(i, d, ev.minus(gradOut.valueAt(i, d),
          ev.times(ev.exp(outputT.valueAt(i, d)), ev.fromType[Double](sum))))
        d += 1
      }
      i += 1
    }
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.dim() == 1 || input.dim() == 2
      || input.dim() == 3, "vector, matrix or batch matrix expected")
    gradInput.resizeAs(input)
    if (input.dim() != 3) {
      super.updateGradInput(input, gradOutput)
    } else {
      val (batchSize, nframe, dim) = (output.size(1), output.size(2), output.size(3))

      if (results == null || results.length != batchSize) {
        results = new Array[Future[Unit]](batchSize)
      }

      var t = 1
      while (t <= batchSize) {
        val _t = t
        results(_t - 1) = Engine.model.invoke(() => {
          updateGradOutputFrame(input.select(1, _t), gradOutput.select(1, _t),
            gradInput.select(1, _t), output.select(1, _t))
        })
        t += 1
      }
      Engine.model.sync(results)

      gradInput
    }
  }

  override def toString(): String = {
    s"nn.LogSoftMaxBatch"
  }
}
