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
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

/**
 * Normalizes the input Tensor to have unit L_p norm. The smoothing parameter eps prevents
 * division by zero when the input contains all zero elements (default = 1e-10).
 * The input can be 1d, 2d or 4d
 * If the input is 4d, it should follow the format (n, c, h, w) where n is the batch number,
 * c is the channel number, h is the height and w is the width
 * @param p L_p norm
 * @param eps smoothing parameter
 * @tparam T The numeric type in the criterion, usually which are [[Float]] or [[Double]]
 */
@SerialVersionUID(1504221556573977764L)
class Normalize[T: ClassTag](val p: Double, val eps: Double = 1e-10
  )(implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  require(p > 0, s"Normalize: $p-norm not supported, norm number must be bigger than zero")

  // buffer
  val norm = Tensor[T]()
  val normp = Tensor[T]()

  val buffer = Tensor[T]()
  val buffer2 = Tensor[T]()

  var inputBuffer = Tensor[T]()

  val cross = Tensor[T]()
  val crossBuffer = Tensor[T]()
  val indices = Tensor[T]()

  var cmul: CMul[T] = null

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() <= 2 || input.dim() == 4, s"Normalize: only 1d , 2d" +
      s"or 4d layer supported, " +
      s"but got input dim ${ input.dim() }")
    inputBuffer = if (input.dim() == 1) input.view(1, input.nElement()) else input
    output.resizeAs(inputBuffer)

    if (p == Double.MaxValue) {
      buffer.resizeAs(inputBuffer).abs(inputBuffer)
      buffer.max(norm, indices, 2)
      norm.add(ev.fromType(eps))
    } else {
      if (p%2 != 0) {
        buffer.resizeAs(inputBuffer).abs(inputBuffer).pow(ev.fromType(p))
      } else {
        buffer.resizeAs(inputBuffer).pow(inputBuffer, ev.fromType(p))
      }
      // normp.sum(buffer, 2).add(ev.fromType(eps))
      // perf fix: the sum operation of tensor will call Java + element wise
      // perf fix: start
      if (buffer.nDimension() <= 2) {
         normp.sum(buffer, 2).add(ev.fromType(eps))
      } else {
        normp.resize(Array(buffer.size(1), 1, buffer.size(3), buffer.size(4)))
        var batchSize = 0
        while (batchSize < normp.size(1)) {
          val normpPerBatch = normp.select(1, batchSize + 1).zero
          val inputPerBatch = buffer.narrow(1, batchSize + 1, 1)

          var channel = 0
          while (channel < buffer.size(2)) {
            normpPerBatch.add(inputPerBatch.select(2, channel + 1))
            channel += 1
          }
          batchSize += 1
        }
        normp.add(ev.fromType(eps))
      }
      // perf fix: end
      norm.resizeAs(normp).pow(normp, ev.fromType(1.0 / p))
    }
    if (norm.dim() <= 2) {
      output.cdiv(inputBuffer, norm.view(norm.nElement(), 1).expandAs(inputBuffer))
    } else if (norm.dim() == 4) {
      // output.cdiv(inputBuffer, norm.view(norm.size()).expandAs(inputBuffer))
      // perf fix: after expand, the tensor will be not contiguous.
      // perf fix: start
      var batchSize = 0
      while (batchSize < output.size(1)) {

        val outputPerBatch = output.narrow(1, batchSize + 1, 1)
        val normPerBatch = norm.select(1, batchSize + 1)
        val inputPerBatch = inputBuffer.narrow(1, batchSize + 1, 1)

        var channel = 0
        while (channel < output.size(2)) {
          outputPerBatch.select(2, channel + 1)
            .cdiv(inputPerBatch.select(2, channel + 1), normPerBatch)
          channel += 1
        }

        batchSize += 1
      }
      // perf fix: end
    }

    output = output.view(input.size())
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.dim() <= 2 || input.dim() == 4, s"Normalize: only 1d, 2d," +
      s"or 4d layer supported, " +
      s"but got input dim ${ input.dim() }")
    require(gradOutput.dim() <= 2 || gradOutput.dim() == 4,
      s"Normalize: only 1d or 4d layer supported, " +
        s"but got gradOutput dim ${ gradOutput.dim() }")

    inputBuffer = if (input.dim() == 1) input.view(1, input.nElement()) else input
    val n = inputBuffer.size(1)
    val d = inputBuffer.size(2)

    // compute diagonal term with gradOutput
    gradInput.resizeAs(inputBuffer)
    if (p == Double.MaxValue) {
      gradInput.cmul(norm.view(n, 1, 1).expand(Array(n, d, 1)), gradOutput)
      buffer.resizeAs(inputBuffer).zero()
      cross.resize(n, 1)
      cross.gather(2, indices, inputBuffer)
      cross.cdiv(norm)
      buffer.scatter(2, indices, cross)
    } else {
      if (input.dim() <= 2) {
        gradInput.cmul(normp.view(n, 1).expand(Array(n, d)), gradOutput)
      } else {
        gradInput.cmul(normp.view(n, 1, inputBuffer.size(3), inputBuffer.size(4))
          .expandAs(inputBuffer), gradOutput)
      }

      if (p%2 != 0) {
        if (p < 2) {
          buffer.abs(inputBuffer).add(ev.fromType(eps)).pow(ev.fromType(p - 2)).cmul(inputBuffer)
        } else {
          buffer.abs(inputBuffer).pow(ev.fromType(p - 2)).cmul(inputBuffer)
        }
      } else if (p == 2) {
        buffer.copy(inputBuffer)
      } else {
        buffer.pow(inputBuffer, ev.fromType(p - 2)).cmul(inputBuffer)
      }
    }

    buffer2.resizeAs(inputBuffer).cmul(inputBuffer, gradOutput)
    cross.resize(n, 1).sum(buffer2, 2)

    crossBuffer.resizeAs(cross).copy(cross)
    buffer.cmul(crossBuffer.expandAs(buffer))

    gradInput.add(ev.fromType(-1), buffer)

    if (p == Double.MaxValue) {
      cross.cmul(norm, norm)
    } else {
      cross.resizeAs(norm).cmul(normp, norm)
    }

    gradInput.cdiv(cross.expandAs(inputBuffer))
    gradInput = gradInput.view(input.size())

    gradInput
  }

  override def toString(): String = {
    s"${getPrintName}($p, $eps)"
  }

  override def clearState() : this.type = {
    super.clearState()
    norm.set()
    normp.set()
    buffer.set()
    buffer2.set()
    inputBuffer.set()
    cross.set()
    crossBuffer.set()
    indices.set()
    this
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[Normalize[T]]

  override def equals(other: Any): Boolean = other match {
    case that: Normalize[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        p == that.p &&
        eps == that.eps
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(super.hashCode(), p, eps)
    state.map(getHashCode).foldLeft(0)((a, b) => 31 * a + b)
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    inputShape
  }
}

object Normalize {
  def apply[@specialized(Float, Double) T: ClassTag](
    p: Double,
    eps: Double = 1e-10)(implicit ev: TensorNumeric[T]) : Normalize[T] = {
    new Normalize(p, eps)
  }
}
