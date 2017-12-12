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
package com.intel.analytics.bigdl.nn.ops

import com.intel.analytics.bigdl.nn.SpatialCrossMapLRN
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * LRNGrad calculate the backprop gradients of the Local response normalization layer.
 *
 * @param depthRadius
 * @param bias
 * @param alpha
 * @param beta
 * @param ev$1
 * @param ev
 * @param ev2
 * @tparam T Numeric type. Only support float/double now
 */
class LRNGrad[T: ClassTag](
  depthRadius: Int = 5,
  bias: Float = 1.0f,
  alpha: Float = 1.0f,
  beta: Float = 0.5f
)(implicit ev: TensorNumeric[T], ev2: TensorNumeric[Float])
  extends Operation[Table, Tensor[Float], T] {

  output = Tensor[Float]()

  override def updateOutput(input: Table): Tensor[Float] = {
    val gradOutput = input[Tensor[Float]](1)
    val inputTensor = input[Tensor[Float]](2)
    val outputTensor = input[Tensor[Float]](3)

    output.resizeAs(inputTensor)
    var b = 1
    while(b <= inputTensor.size(1)) {
      SpatialCrossMapLRN.backwardFrameNHWCFloat(
        gradOutput.select(1, b),
        inputTensor.select(1, b),
        output.select(1, b),
        outputTensor.select(1, b),
        alpha * (2 * depthRadius + 1), 2 * depthRadius + 1, beta, bias
      )
      b += 1
    }
    output
  }

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}

object LRNGrad {
  def apply[T: ClassTag](
    depthRadius: Int = 5,
    bias: Float = 1.0f,
    alpha: Float = 1.0f,
    beta: Float = 0.5f
  )(implicit ev: TensorNumeric[T]): LRNGrad[T]
  = new LRNGrad(depthRadius, bias, alpha, beta)
}
