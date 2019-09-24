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

package com.intel.analytics.bigdl.python.api

import scala.reflect.ClassTag

import com.intel.analytics.bigdl.nn.onnx._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric


private[bigdl] object PythonBigDLOnnx {

  def ofFloat(): PythonBigDLOnnx[Float] = new PythonBigDLOnnx[Float]()

  def ofDouble(): PythonBigDLOnnx[Double] = new PythonBigDLOnnx[Double]()

}


class PythonBigDLOnnx[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonBigDL[T] {

  def createGemm(alpha: Float, beta: Float, transA: Int, transB: Int,
                 matrixB: JTensor, matrixC: JTensor): Gemm[T] = {
    Gemm(alpha, beta,
      (if (transA == 0) false else true),
      (if (transB == 0) false else true),
      toTensor(matrixB), toTensor(matrixC))
  }


  def createShape(): Shape[T] = {
    Shape[T]()
  }

}
