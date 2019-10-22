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

package com.intel.analytics.bigdl.nn.onnx

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.ops.{BatchMatMul, Operation}
import com.intel.analytics.bigdl.nn.{CAddTable, Graph, Input, MulConstant, Sequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag


/**
 * General Matrix multiplication
 *
 * Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
 * input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
 * and output tensor Y has shape (M, N).
 *
 * @param alpha  Scalar multiplier for the product of input tensors A * B.
 * @param beta   Scalar multiplier for input tensor C.
 * @param transA Whether A should be transposed
 * @param transB Whether B should be transposed
 * @param matrixB matrix B
 * @param matrixC matrix C
 * @param ev
 * @tparam T The numeric type in this module parameters.
 */
private[bigdl] class Gemm[T: ClassTag](
  val alpha: Float, val beta: Float,
  val transA: Boolean, val transB: Boolean,
  val matrixB: Tensor[T],
  val matrixC: Tensor[T]
)(implicit ev: TensorNumeric[T])
extends Operation[Tensor[T], Tensor[T], T] {

//  require(matrixB.dim() == 2, "Matrix B should be 2D")
//  require(matrixC.dim() == 2, "Matrix C should be 2D")

  // alpha * B'
  val transformedMatrixB = (if (transB == true) matrixB.t() else matrixB).mul(ev.fromType(alpha))
  // beta * C
  val transformedMatrixC = matrixC.mul(ev.fromType(beta))

  // alpha * A' * B' + beta * C
  val gemmGraph: Module[T] = {
    val inputA = Input()
    val inputB = Input()
    val inputC = Input()
    // alpha * A' * B'
    val alphaMul = BatchMatMul(adjX = transA).inputs(Array(inputA, inputB))
    // alpha * A' * B' + beta * C
    val betaAdd = CAddTable().inputs(Array(alphaMul, inputC))
    Graph(Array(inputA, inputB, inputC), betaAdd)
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output = gemmGraph.forward(T(input,
      transformedMatrixB, transformedMatrixC)).asInstanceOf[Tensor[T]]
    output
  }

  override def release(): Unit = {
    gemmGraph.release()
    release()
  }

}


object Gemm {
  def apply[@specialized(Float, Double) T: ClassTag](
    alpha: Float, beta: Float,
    transA: Boolean, transB: Boolean,
    matrixB: Tensor[T], matrixC: Tensor[T]
  )(implicit ev: TensorNumeric[T]): Gemm[T] = {
    new Gemm[T](alpha = alpha, beta = beta, transA = transA, transB = transB,
      matrixB = matrixB, matrixC = matrixC)
  }
}
