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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.ops.BatchMatMul
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag


/**
 * General Matrix multiplication
 *
 * Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
 * input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
 * and output tensor Y has shape (M, N).
 *
 * @param alpha Scalar multiplier for the product of input tensors A * B.
 * @param beta  Scalar multiplier for input tensor C.
 * @param transA Whether A should be transposed
 * @param transB Whether B should be transposed
 * @param ev
 * @tparam T The numeric type in this module parameters.
 */
class Gemm[T: ClassTag](
  alpha: Float = 1, beta: Float = 1,
  transA: Boolean = false, transB: Boolean = false
)(implicit ev: TensorNumeric[T])
extends AbstractModule[Table, Tensor[T], T] {

  private val internalModel: Module[T] = {
    val tensorA = Input()
    val tensorB = Input()
    val tensorC = Input()
    val mul = BatchMatMul().inputs(Array(tensorA, tensorB))
    val add = CAddTable().inputs(Array(mul, tensorC))
    Graph(Array(tensorA, tensorB, tensorC), add)
  }


  override def updateOutput(input: Table): Tensor[T] = {
    require(input.length() == 3)

    input.get(1).get.asInstanceOf[Tensor[T]].mul(alpha.asInstanceOf[T])

    if (transA) {
      input.update(1, input.get(1).get.asInstanceOf[Tensor[T]].t())
    }

    if (transB) {
      input.update(2, input.get(2).get.asInstanceOf[Tensor[T]].t())
    }

    internalModel.forward(input)
    output = internalModel.output.asInstanceOf[Tensor[T]]
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    gradInput = internalModel.gradInput.asInstanceOf[Table]
    gradInput
  }

}

object Gemm {
  def apply[@specialized(Float, Double) T: ClassTag](
    alpha: Float = 1, beta: Float = 1,
    transA: Boolean = false, transB: Boolean = false
  )(implicit ev: TensorNumeric[T]): Gemm[T] = {
    new Gemm[T](alpha, beta, transA, transB)
  }

}
