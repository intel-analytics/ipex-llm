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

import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.{SparseTensorBLAS, SparseTensorMath, SparseType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * SparseLinear is the sparse version of module Linear. SparseLinear has two different from Linear:
 * firstly, SparseLinear's input Tensor is a SparseTensor. Secondly, SparseLinear doesn't backward
 * gradient to next layer in the backpropagation by default, as the gradInput of SparseLinear is
 * useless and very big in most cases.
 *
 * But, considering model like Wide&Deep, we provide backwardStart and backwardLength to backward
 * part of the gradient to next layer.
 *
 * @param inputSize the size the each input sample
 * @param outputSize the size of the module output of each sample
 * @param backwardStart backwardStart index, counting from 1
 * @param backwardLength backward length
 * @param withBias if has bias
 * @param wRegularizer: instance of [[Regularizer]]
 *                    (eg. L1 or L2 regularization), applied to the input weights matrices.
 * @param bRegularizer: instance of [[Regularizer]]
 *                    applied to the bias.
 */
class SparseLinear[T: ClassTag](
      inputSize: Int,
      outputSize: Int,
      val backwardStart: Int = -1,
      val backwardLength: Int = -1,
      withBias: Boolean = true,
      wRegularizer: Regularizer[T] = null,
      bRegularizer: Regularizer[T] = null,
      initWeight: Tensor[T] = null,
      initBias: Tensor[T] = null,
      initGradWeight: Tensor[T] = null,
      initGradBias: Tensor[T] = null)(implicit ev: TensorNumeric[T]) extends Linear[T](
  inputSize, outputSize, withBias, wRegularizer, bRegularizer,
  initWeight, initBias, initGradWeight, initGradBias) {

  // input should be a sparseTensor
  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.getTensorType == SparseType, s"SparseLinear's input must be a SparseTensor," +
      s"but got ${input.getTensorType}")
    require(input.dim() == 2,
      "SparseLinear: " + ErrorInfo.constrainInputAsVectorOrBatch)

    val nFrame = input.size(1)
    val nElement = output.nElement
    val t = Array(nFrame, weight.size(1))
    output.resize(t)
    if (output.nElement() != nElement) {
      output.zero()
    }

    if (addBuffer.nElement() != nFrame) {
      addBuffer.resize(Array(nFrame)).fill(ev.one)
    }

    SparseTensorMath.addmm(output, ev.zero, output, ev.one, input, weight.t)
    if(withBias) output.addr(ev.one, addBuffer, bias)
    output
  }

  // just backward a part of the gradOutput. Input is sparse, while gradOutput is dense.
  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.dim() == 2,
      "SparseLinear: " + ErrorInfo.constrainInputAsVectorOrBatch)
    if (backwardStart >= 0 && backwardLength > 0) {
      val _inputSize = Array(input.size(1), backwardLength)
      val _weight = weight.narrow(2, backwardStart, backwardLength)

      val nElement = gradInput.nElement()
      gradInput.resize(_inputSize)
      if (nElement != gradInput.nElement()) {
        gradInput.zero()
      }

      gradInput.addmm(ev.zero, ev.one, gradOutput, _weight)
    }
    gradInput
  }

  override def accGradParameters(
        input: Tensor[T],
        gradOutput: Tensor[T]): Unit = {
    require(input.dim() == 2,
      "SparseLinear: " + ErrorInfo.constrainInputAsVectorOrBatch)

    gradWeight.resize(outputSize, inputSize)
    if (withBias) {
      gradBias.resize(outputSize)
    }

    if (scaleW != 0) {
      SparseTensorMath.addmm(gradWeight, ev.one, gradWeight,
        ev.fromType[Double](scaleW), gradOutput.t, input)
    }

    if (withBias && scaleB != 0) {
      gradBias.addmv(ev.fromType[Double](scaleB), gradOutput.t, addBuffer)
    }

    if (null != wRegularizer && scaleW != 0) {
      wRegularizer.accRegularization(weight, gradWeight, scaleW)
    }
    if (null != bRegularizer && scaleB != 0) {
      bRegularizer.accRegularization(bias, gradBias, scaleB)
    }
  }

  override def toString() : String = {
    s"nn.SparseLinear($inputSize -> $outputSize)"
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[SparseLinear[T]]

  override def equals(other: Any): Boolean = other match {
    case that: SparseLinear[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        backwardStart == that.backwardStart &&
        backwardLength == that.backwardLength
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), backwardStart, backwardLength)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object SparseLinear {
  def apply[@specialized(Float, Double) T: ClassTag](
          inputSize: Int,
          outputSize: Int,
          withBias: Boolean = true,
          backwardStart: Int = -1,
          backwardLength: Int = -1,
          wRegularizer: Regularizer[T] = null,
          bRegularizer: Regularizer[T] = null,
          initWeight: Tensor[T] = null,
          initBias: Tensor[T] = null,
          initGradWeight: Tensor[T] = null,
          initGradBias: Tensor[T] = null
        )(implicit ev: TensorNumeric[T]): SparseLinear[T] = {
    new SparseLinear[T](inputSize, outputSize, backwardStart, backwardLength,
      withBias, wRegularizer, bRegularizer, initWeight, initBias, initGradWeight, initGradBias)
  }
}
