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
import com.intel.analytics.bigdl.nn.ops.{BatchMatMul, Operation}
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
  val alpha: Float = 1, val beta: Float = 1,
  val transA: Boolean = false, val transB: Boolean = false
)(implicit ev: TensorNumeric[T])
extends Operation[Table, Tensor[T], T] {

  private val internalModel: Module[T] = {
    val tensorA = Input()
    val tensorB = Input()
    val tensorC = Input()
    val alphaMul = MulConstant(scalar = alpha, inplace = true).inputs(
      BatchMatMul(adjX = transA, adjY = transB).inputs(Array(tensorA, tensorB))
    )
    val betaAdd = CAddTable().inputs(Array(alphaMul,
      MulConstant(scalar = beta, inplace = true).inputs(tensorC)))

    Graph(Array(tensorA, tensorB, tensorC), betaAdd)
  }

  override def updateOutput(input: Table): Tensor[T] = {
    require(input.length() == 3, "Input should be a table contains 3 tensors, actually size is: "
      + input.toTable.length())
    internalModel.forward(input)
    output = internalModel.output.asInstanceOf[Tensor[T]]
    output
  }

  override def release(): Unit = {
    internalModel.release()
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
