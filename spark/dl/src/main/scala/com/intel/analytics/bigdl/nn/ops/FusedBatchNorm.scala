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

import com.intel.analytics.bigdl.nn.SpatialBatchNormalization
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * This is similar to SpatialBatchNormalization.
 *
 * When isTraining is true, it takes three tensors as inputs, which is image,
 * scale, offset.
 *
 * The operation implemented is:
 *
 *         ( image - batch-mean(x) )
 * y = ---------------------------------- * weight + offset
 *      batch-standard-deviation(x)
 *
 * The operation will output y, mean and variance tensors.
 *
 * If the isTraining is false, it takes five tensors as inputs, which is image, scale, offset, mean,
 * and variance.
 *
 * @param epsilon
 * @param isTraining
 * @param momentum
 * @param dataFormat
 * @param ev$1
 * @param ev
 * @tparam T Numeric type. Only support float/double now
 */
class FusedBatchNorm[T: ClassTag](
  epsilon: Float = 0.0001f,
  isTraining: Boolean = true,
  momentum: Float = 0.1f,
  dataFormat: DataFormat = DataFormat.NHWC
)(implicit ev: TensorNumeric[T]) extends Operation[Table, Table, T]{

  @transient
  private var runningMean: Tensor[Float] = null

  @transient
  private var runningVar: Tensor[Float] = null

  @transient
  private var saveStd: Tensor[Float] = null

  override def updateOutput(input: Table): Table = {
    val x = input[Tensor[Float]](1)
    val scale = input[Tensor[Float]](2)
    val offset = input[Tensor[Float]](3)
    val mean = input[Tensor[Float]](4)
    val variance = input[Tensor[Float]](5)

    if (output.length() == 0) {
      output(1) = Tensor[Float]().resizeAs(x) // y
      output(2) = Tensor[Float](x.size(4)) // batch mean
      output(3) = Tensor[Float](x.size(4)) // batch var
      output(4) = Tensor[Float](x.size(4)) // save mean
      output(5) = Tensor[Float](x.size(4)) // save var
      runningMean = Tensor[Float](x.size(4)) // running mean
      runningVar = Tensor[Float](x.size(4)) // running var
      saveStd = Tensor[Float](x.size(4)) // save std
    }

    val y = output[Tensor[Float]](1)
    val batchMean = output[Tensor[Float]](2)
    val batchVar = output[Tensor[Float]](3)
    val saveMean = output[Tensor[Float]](4)
    val saveVar = output[Tensor[Float]](5)

    if (isTraining) {
      if (dataFormat == DataFormat.NHWC) {
        SpatialBatchNormalization.updateOutputNHWCTrainFloat(
          x, y, batchMean, saveStd, runningMean, runningVar, scale, offset, epsilon, momentum,
          batchVar, saveVar
        )
      } else {
        SpatialBatchNormalization.updateOutputNCHWTrainFloat(
          x, y, batchMean, saveStd, runningMean, runningVar, scale, offset, epsilon, momentum,
          batchVar, saveVar
        )
      }
      saveMean.copy(batchMean)
    } else {
      if (dataFormat == DataFormat.NHWC) {
        SpatialBatchNormalization.updateOutputNHWCInferFloat(
          x, y, mean, variance, scale, offset, epsilon
        )
      } else {
        SpatialBatchNormalization.updateOutputNCHWInferFloat(
          x, y, mean, variance, scale, offset, epsilon
        )
      }
    }

    output
  }
}

object FusedBatchNorm {
  def apply[T: ClassTag](epsilon: Float = 0.0001f, isTrainning: Boolean = true,
    momentum: Float = 0.1f, dataFormat: DataFormat = DataFormat.NHWC)
    (implicit ev: TensorNumeric[T]): FusedBatchNorm[T]
  = new FusedBatchNorm(epsilon, isTrainning, momentum, dataFormat)
}
