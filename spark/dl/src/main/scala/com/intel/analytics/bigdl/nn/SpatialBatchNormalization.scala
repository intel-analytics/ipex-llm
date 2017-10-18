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

import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric


import scala.reflect.ClassTag

/**
 * This file implements Batch Normalization as described in the paper:
 * "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
 * by Sergey Ioffe, Christian Szegedy
 * This implementation is useful for inputs coming from convolution layers.
 * For non-convolutional layers, see [[BatchNormalization]]
 * The operation implemented is:
 *
 * ( x - mean(x) )
 * y = -------------------- * gamma + beta
 * standard-deviation(x)
 *
 * where gamma and beta are learnable parameters.
 * The learning of gamma and beta is optional.
 */
@SerialVersionUID(-9106336963903528047L)
class SpatialBatchNormalization[T: ClassTag](
  nOutput: Int, eps: Double = 1e-5, momentum: Double = 0.1, affine: Boolean = true,
  initWeight: Tensor[T] = null,
  initBias: Tensor[T] = null,
  initGradWeight: Tensor[T] = null,
  initGradBias: Tensor[T] = null,
  format: DataFormat = DataFormat.NCHW)(
  implicit ev: TensorNumeric[T])
  extends BatchNormalization[T](nOutput, eps, momentum, affine,
    initWeight, initBias, initGradWeight, initGradBias) {
  override val nDim = 4

  override def toString(): String = {
    s"${getPrintName}[${ev.getType()}]($nOutput, $eps, $momentum, $affine)"
  }

  override val batchFeatDim: Int = {
    format match {
      case DataFormat.NCHW => 2
      case DataFormat.NHWC => 4
    }
  }

  private val varBuffer = Tensor[T](nOutput)

  private val scaleBuffer = Tensor[T](nOutput)

  private val biasBuffer = Tensor[T](nOutput)

  private val invStdBuffer = Tensor[T](nOutput)

  private val saveVar = Tensor[T](nOutput)

  private val ones = if (!affine && format.isInstanceOf[DataFormat.NHWC.type ]) {
    Tensor[T](nOutput)
  } else {
    null
  }

  override def clearState(): this.type = {
    super.clearState()
    varBuffer.set()
    scaleBuffer.set()
    biasBuffer.set()
    invStdBuffer.set()
    saveVar.set()
    this
  }

  override protected def updateOutputDouble(
      inputTensor: Tensor[Double],
      outputTensor: Tensor[Double]): Unit = {
    format match {
      case DataFormat.NCHW =>
        super.updateOutputDouble(inputTensor, outputTensor)
      case DataFormat.NHWC =>
        updateOutputDoubleNHWC(inputTensor, outputTensor)
    }
  }

  override protected def updateOutputFloat(
      inputTensor: Tensor[Float],
      outputTensor: Tensor[Float]): Unit = {
    format match {
      case DataFormat.NCHW =>
        super.updateOutputFloat(inputTensor, outputTensor)
      case DataFormat.NHWC =>
        updateOutputFloatNHWC(inputTensor, outputTensor)
    }
  }

  override protected def backwardFloat(
    inputTensor: Tensor[Float],
    gradOutputTensor: Tensor[Float],
    gradInputTensor: Tensor[Float], nInput: Int, n: Int, scaleW: Float, scaleB: Float,
    gradWeightTensor: Tensor[Float], gradBiasTensor: Tensor[Float]): Unit = {
    format match {
      case DataFormat.NCHW =>
        super.backwardFloat(inputTensor, gradOutputTensor,
          gradInputTensor, nInput, n,
          scaleW, scaleB,
          gradWeightTensor,
          gradBiasTensor)
      case DataFormat.NHWC =>
        backwardFloatNHWC(inputTensor, gradOutputTensor,
          gradInputTensor, nInput, n,
          scaleW, scaleB,
          gradWeightTensor,
          gradBiasTensor)
    }
  }

  override protected def backwardDouble(
    inputTensor: Tensor[Double],
    gradOutputTensor: Tensor[Double],
    gradInputTensor: Tensor[Double], nInput: Int, n: Int, scaleW: Double, scaleB: Double,
    gradWeightTensor: Tensor[Double], gradBiasTensor: Tensor[Double]): Unit = {
    format match {
      case DataFormat.NCHW =>
        super.backwardDouble(inputTensor, gradOutputTensor,
          gradInputTensor, nInput, n,
          scaleW, scaleB,
          gradWeightTensor,
          gradBiasTensor)
      case DataFormat.NHWC =>
        backwardDoubleNHWC(inputTensor, gradOutputTensor,
          gradInputTensor, nInput, n,
          scaleW, scaleB,
          gradWeightTensor,
          gradBiasTensor)
    }
  }

  private def backwardDoubleNHWC(
    inputTensor: Tensor[Double],
    gradOutputTensor: Tensor[Double],
    gradInputTensor: Tensor[Double], nInput: Int, n: Int, scaleW: Double, scaleB: Double,
    gradWeightTensor: Tensor[Double], gradBiasTensor: Tensor[Double]): Unit = {


    val batchSize = inputTensor.size(1)
    val sampleSize = inputTensor.size(2) * inputTensor.size(3)
    val channel = inputTensor.size(4)

    val gradOut2d = gradOutputTensor.view(Array(batchSize * sampleSize, channel))
    val input2d = inputTensor.view(Array(batchSize * sampleSize, channel))
    val gradInput2d = gradInputTensor.view(Array(batchSize * sampleSize, channel))
    val w = if (null != weight) {
      weight.asInstanceOf[Tensor[Double]]
    } else {
      ones.asInstanceOf[Tensor[Double]]
    }
    val (mean, invstd) = if (train) {
      (saveMean.asInstanceOf[Tensor[Double]], saveStd.asInstanceOf[Tensor[Double]])
    } else {
      (runningMean.asInstanceOf[Tensor[Double]], saveStd.asInstanceOf[Tensor[Double]])
    }

    val sum = Tensor[Double](nOutput).sum(gradOut2d, 1).squeeze()

    val dotp = Tensor[Double](nOutput)
    val dotpBuffer = Tensor[Double](nOutput)
    var i = 0
    while (i < batchSize * sampleSize) {
      val inputSlice = input2d.select(1, i + 1)
      val gradOutSlice = gradOut2d.select(1, i + 1)
      dotpBuffer.copy(inputSlice).sub(mean).cmul(gradOutSlice)
      dotp.add(dotpBuffer)
      i = i + 1
    }

    if (null != gradInputTensor) {
      val k = Tensor[Double](nOutput).fill(1.0)
      k.cmul(dotp).cmul(invstd).cmul(invstd).div(n)
      if (train) {
        i = 0
        while (i < batchSize * sampleSize) {
          val inputSlice = input2d.select(1, i + 1)
          val gradInSlice = gradInput2d.select(1, i + 1)
          gradInSlice.copy(inputSlice).sub(mean).cmul(k)
          i = i + 1
        }

        val minusGradMean = Tensor[Double](nOutput).copy(sum).div(-n)
        i = 0
        while (i < batchSize * sampleSize) {
          val gradOutSlice = gradOut2d.select(1, i + 1)
          val gradInSlice = gradInput2d.select(1, i + 1)
          gradInSlice.mul(-1).add(minusGradMean).add(gradOutSlice).cmul(invstd).cmul(w)
          i = i + 1
        }
      } else {
        i = 0
        while (i <  batchSize * sampleSize) {
          val gradOutSlice = gradOut2d.select(1, i + 1)
          val gradInSlice = gradInput2d.select(1, i + 1)
          gradInSlice.copy(gradOutSlice).cmul(invstd).cmul(w)
          i = i + 1
        }
      }

      if (null != gradWeightTensor && scaleW != 0) {
        gradWeightTensor.addcmul(scaleW, dotp, invstd)
      }

      if (null != gradBiasTensor && scaleB != 0) {
        gradBiasTensor.add(scaleB, sum)
      }
    }

  }

  private def backwardFloatNHWC(
    inputTensor: Tensor[Float],
    gradOutputTensor: Tensor[Float],
    gradInputTensor: Tensor[Float], nInput: Int, n: Int, scaleW: Float, scaleB: Float,
    gradWeightTensor: Tensor[Float], gradBiasTensor: Tensor[Float]): Unit = {

    val batchSize = inputTensor.size(1)
    val sampleSize = inputTensor.size(2) * inputTensor.size(3)
    val channel = inputTensor.size(4)

    val gradOut2d = gradOutputTensor.view(Array(batchSize * sampleSize, channel))
    val input2d = inputTensor.view(Array(batchSize * sampleSize, channel))
    val gradInput2d = gradInputTensor.view(Array(batchSize * sampleSize, channel))
    val w = if (null != weight) {
      weight.asInstanceOf[Tensor[Float]]
    } else {
      ones.asInstanceOf[Tensor[Float]]
    }
    val (mean, invstd) = if (train) {
      (saveMean.asInstanceOf[Tensor[Float]], saveStd.asInstanceOf[Tensor[Float]])
    } else {
      (runningMean.asInstanceOf[Tensor[Float]], saveStd.asInstanceOf[Tensor[Float]])
    }

    val sum = Tensor[Float](nOutput).sum(gradOut2d, 1).squeeze()

    val dotp = Tensor[Float](nOutput)
    val dotpBuffer = Tensor[Float](nOutput)
    var i = 0
    while (i < batchSize * sampleSize) {
      val inputSlice = input2d.select(1, i + 1)
      val gradOutSlice = gradOut2d.select(1, i + 1)
      dotpBuffer.copy(inputSlice).sub(mean).cmul(gradOutSlice)
      dotp.add(dotpBuffer)
      i = i + 1
    }

    if (null != gradInputTensor) {
      val k = Tensor[Float](nOutput).fill(1.0f)
      k.cmul(dotp).cmul(invstd).cmul(invstd).div(n)
      if (train) {
        i = 0
        while (i < batchSize * sampleSize) {
          val inputSlice = input2d.select(1, i + 1)
          val gradInSlice = gradInput2d.select(1, i + 1)
          gradInSlice.copy(inputSlice).sub(mean).cmul(k)
          i = i + 1
        }

        val minusGradMean = Tensor[Float](nOutput).copy(sum).div(-n)
        i = 0
        while (i < batchSize * sampleSize) {
          val gradOutSlice = gradOut2d.select(1, i + 1)
          val gradInSlice = gradInput2d.select(1, i + 1)
          gradInSlice.mul(-1).add(minusGradMean).add(gradOutSlice).cmul(invstd).cmul(w)
          i = i + 1
        }
      } else {
        i = 0
        while (i <  batchSize * sampleSize) {
          val gradOutSlice = gradOut2d.select(1, i + 1)
          val gradInSlice = gradInput2d.select(1, i + 1)
          gradInSlice.copy(gradOutSlice).cmul(invstd).cmul(w)
          i = i + 1
        }
      }

      if (null != gradWeightTensor && scaleW != 0) {
        gradWeightTensor.addcmul(scaleW, dotp, invstd)
      }

      if (null != gradBiasTensor && scaleB != 0) {
        gradBiasTensor.add(scaleB, sum)
      }
    }

  }


  private def updateOutputDoubleNHWC(
                                      inputTensor: Tensor[Double],
                                      outputTensor: Tensor[Double]): Unit = {
    val batchSize = inputTensor.size(1)
    val sampleSize = inputTensor.size(2) * inputTensor.size(3)
    val channel = inputTensor.size(4)
    val input2d = inputTensor.view(Array(batchSize * sampleSize, channel))
    val output2d = outputTensor.view(Array(batchSize * sampleSize, channel))
    val saveMeanDouble = saveMean.asInstanceOf[Tensor[Double]].zero()
    val varBufferDouble = varBuffer.asInstanceOf[Tensor[Double]].zero()
    val saveVarDouble = saveVar.asInstanceOf[Tensor[Double]].zero()
    val invStdBuffDouble = saveStd.asInstanceOf[Tensor[Double]].zero()
    val scaleBuffDouble = scaleBuffer.asInstanceOf[Tensor[Double]].zero()
    val biasBuffDouble = biasBuffer.asInstanceOf[Tensor[Double]].zero()
    val weightDouble = weight.asInstanceOf[Tensor[Double]]
    val biasDouble = bias.asInstanceOf[Tensor[Double]]
    val runningMeanDouble = runningMean.asInstanceOf[Tensor[Double]]
    val runningVarDouble = runningVar.asInstanceOf[Tensor[Double]]

    val (mean, invStd) = if (train) {
      saveMeanDouble.sum(input2d, 1)
      saveMeanDouble.div(batchSize * sampleSize)

      var i = 0
      while (i < batchSize * sampleSize) {
        val channels = input2d.select(1, i + 1)
        varBufferDouble.sub(channels, saveMeanDouble)
        saveVarDouble.addcmul(varBufferDouble, varBufferDouble)
        i = i + 1
      }

      saveVarDouble.div(batchSize * sampleSize)

      runningMeanDouble.mul(1.0 - momentum).add(momentum, saveMeanDouble)
      runningVarDouble.mul(1.0 - momentum).add(momentum, saveVarDouble)

      invStdBuffDouble.zero()
      invStdBuffDouble.add(saveVarDouble).add(eps).pow(-1).sqrt()
      (saveMeanDouble, invStdBuffDouble)
    } else {
      invStdBuffDouble.zero()
      invStdBuffDouble.add(runningVarDouble).add(eps).pow(-1).sqrt()
      (runningMeanDouble, invStdBuffDouble)
    }

    val (w, b) = if (affine) {
      scaleBuffDouble.cmul(invStdBuffDouble, weightDouble)
      biasBuffDouble.cmul(invStd, mean).cmul(weightDouble)
      biasBuffDouble.sub(biasDouble, biasBuffDouble)
      (scaleBuffDouble, biasBuffDouble)
    } else {
      biasBuffDouble.cmul(mean, invStd).mul(-1)
      (weightDouble, biasBuffDouble)
    }

    var i = 0
    while (i < batchSize * sampleSize) {
      val inSlice = input2d.select(1, i + 1)
      val outSlice = output2d.select(1, i + 1)
      outSlice.cmul(inSlice, w).add(b)
      i = i + 1
    }
  }


  private def updateOutputFloatNHWC(
                                      inputTensor: Tensor[Float],
                                      outputTensor: Tensor[Float]): Unit = {
    val batchSize = inputTensor.size(1)
    val sampleSize = inputTensor.size(2) * inputTensor.size(3)
    val channel = inputTensor.size(4)
    val input2d = inputTensor.view(Array(batchSize * sampleSize, channel))
    val output2d = outputTensor.view(Array(batchSize * sampleSize, channel))
    val saveMeanFloat = saveMean.asInstanceOf[Tensor[Float]]
    val varBufferFloat = varBuffer.asInstanceOf[Tensor[Float]]
    val saveVarFloat = saveVar.asInstanceOf[Tensor[Float]]
    val runningMeanFloat = runningMean.asInstanceOf[Tensor[Float]]
    val runningVarFloat = runningVar.asInstanceOf[Tensor[Float]]
    val invStdBuffFloat = saveStd.asInstanceOf[Tensor[Float]]
    val scaleBuffFloat = scaleBuffer.asInstanceOf[Tensor[Float]]
    val biasBuffFloat = biasBuffer.asInstanceOf[Tensor[Float]]
    val weightFloat = weight.asInstanceOf[Tensor[Float]]
    val biasFloat = bias.asInstanceOf[Tensor[Float]]

    val (mean, invStd) = if (train) {
      saveMeanFloat.sum(input2d, 1)
      saveMeanFloat.div(batchSize * sampleSize)

      var i = 0
      while (i < batchSize * sampleSize) {
        val channels = input2d.select(1, i + 1)
        varBufferFloat.sub(channels, saveMeanFloat)
        saveVarFloat.addcmul(varBufferFloat, varBufferFloat)
        i = i + 1
      }

      saveVarFloat.div(batchSize * sampleSize)

      runningMeanFloat.mul(1.0f - momentum.toFloat).add(momentum.toFloat, saveMeanFloat)
      runningVarFloat.mul(1.0f - momentum.toFloat).add(momentum.toFloat, saveVarFloat)

      invStdBuffFloat.zero()
      invStdBuffFloat.add(saveVarFloat).add(eps.toFloat).pow(-1).sqrt()
      (saveMeanFloat, invStdBuffFloat)
    } else {
      invStdBuffFloat.zero()
      invStdBuffFloat.add(runningVarFloat).add(eps.toFloat).pow(-1).sqrt()
      (runningMeanFloat, invStdBuffFloat)
    }

    val (w, b) = if (affine) {
      scaleBuffFloat.cmul(invStdBuffFloat, weightFloat)
      biasBuffFloat.cmul(invStd, mean).cmul(weightFloat)
      biasBuffFloat.sub(biasFloat, biasBuffFloat)
      (scaleBuffFloat, biasBuffFloat)
    } else {
      biasBuffFloat.cmul(mean, invStd).mul(-1)
      (weightFloat, biasBuffFloat)
    }

    var i = 0
    while (i < batchSize * sampleSize) {
      val inSlice = input2d.select(1, i + 1)
      val outSlice = output2d.select(1, i + 1)
      outSlice.cmul(inSlice, w).add(b)
      i = i + 1
    }
  }
}

object SpatialBatchNormalization {
  def apply[@specialized(Float, Double) T: ClassTag](
      nOutput: Int,
      eps: Double = 1e-5,
      momentum: Double = 0.1,
      affine: Boolean = true,
      initWeight: Tensor[T] = null,
      initBias: Tensor[T] = null,
      initGradWeight: Tensor[T] = null,
      initGradBias: Tensor[T] = null,
      format: DataFormat = DataFormat.NCHW)(implicit ev: TensorNumeric[T])
  : SpatialBatchNormalization[T] = {
    new SpatialBatchNormalization[T](nOutput, eps, momentum, affine,
      initWeight, initBias, initGradWeight, initGradBias, format)
  }
}
