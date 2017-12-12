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

import com.intel.analytics.bigdl.nn.{BatchNormalization, SpatialBatchNormalization}
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * This is the gradient operation coressponding to the FusedBatchNorm. It will calculate the
 * activity, weight and bias gradients of the spatial batch normalization.
 *
 * The formula is
 *      x_backprop = scale * rsqrt(variance + epsilon) * [y_backprop - mean(y_backprop) -
 *     (x - mean(x)) * mean(y_backprop * (x - mean(x))) / (variance + epsilon)]
 *     weight_backprop = sum(y_backprop * (x - mean(x)) * rsqrt(variance + epsilon))
 *     bias_backprop = sum(y_backprop)
 *
 * @param epsilon
 * @param dataFormat
 * @param isTrain
 * @param ev$1
 * @param ev
 * @tparam T Numeric type. Only support float/double now
 */
class FusedBatchNormGrad[T: ClassTag](
  val epsilon: Float, val dataFormat: DataFormat,
  val isTrain: Boolean = false)(implicit ev: TensorNumeric[T])
  extends Operation[Table, Table, T]{


  private val gMean = Tensor[Float]()
  private val gxMean = Tensor[Float]()
  private val saveStd = Tensor[Float]()

  override def updateOutput(input: Table): Table = {
    val gradOutput = input[Tensor[Float]](1)
    val x = input[Tensor[Float]](2)
    val scale = input[Tensor[Float]](3)
    val saveMean = input[Tensor[Float]](4)
    val saveVar = input[Tensor[Float]](5)

    if (output.length() == 0) {
      output(1) = Tensor[Float]().resizeAs(x) // gradInput
      output(2) = Tensor[Float](x.size(4)) // weight gradient
      output(3) = Tensor[Float](x.size(4)) // bias gradient
      saveStd.resize(x.size(4)) // bias gradient
    }
    saveStd.copy(saveVar)
    saveStd.add(epsilon).pow(-0.5f)
    val gradInput = output[Tensor[Float]](1)
    val gradWeight = output[Tensor[Float]](2)
    val gradBias = output[Tensor[Float]](3)

    SpatialBatchNormalization.updateGradInputNHWCTrainFloat(
      x, gradOutput, gradInput, scale, saveMean, saveStd, gMean, gxMean)

    gradWeight.zero()
    gradBias.zero()
    SpatialBatchNormalization.accGradientNHWCFloat(
      gradOutput, gradWeight, gradBias, x, saveMean, saveStd, 1.0f, 1.0f)

    output
  }
}

object FusedBatchNormGrad {
  def apply[T: ClassTag](epsilon: Float = 0.0001f, dataFormat: DataFormat = DataFormat.NHWC,
    isTraining: Boolean = true)(implicit ev: TensorNumeric[T]): FusedBatchNormGrad[T] =
    new FusedBatchNormGrad(epsilon, dataFormat, isTraining)
}
