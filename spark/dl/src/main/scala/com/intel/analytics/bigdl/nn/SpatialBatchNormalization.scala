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
import com.intel.analytics.bigdl.tensor.{FloatType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, ParameterSynchronizer}

import scala.collection.JavaConverters._
import scala.reflect.ClassTag

/**
 * This file implements Batch Normalization as described in the paper:
 * "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
 * by Sergey Ioffe, Christian Szegedy
 * This implementation is useful for inputs coming from convolution layers.
 * For non-convolutional layers, see [[BatchNormalization]]
 * The operation implemented is:
 *
 *         ( x - mean(x) )
 * y = -------------------- * gamma + beta
 *      standard-deviation(x)
 *
 * where gamma and beta are learnable parameters.
 * The learning of gamma and beta is optional.
 */
@SerialVersionUID(- 9106336963903528047L)
class SpatialBatchNormalization[T: ClassTag](
  nOutput: Int, eps: Double = 1e-5, momentum: Double = 0.1, affine: Boolean = true,
  initWeight: Tensor[T] = null,
  initBias: Tensor[T] = null,
  initGradWeight: Tensor[T] = null,
  initGradBias: Tensor[T] = null, dataFormat: DataFormat = DataFormat.NCHW)(
  implicit ev: TensorNumeric[T])
  extends BatchNormalization[T](nOutput, eps, momentum, affine,
    initWeight, initBias, initGradWeight, initGradBias) {
  override val nDim = 4

  override def updateOutput(input: Tensor[T]): Tensor[T] = {

    val parallism = getParallism().getOrElse(1)

    val meanKeyWithId = s"${this.meanKey}_${this.getId}"
    val stdKeyWithId = s"${this.stdKey}_${this.getId}"
    val gmKeyWithId = s"${this.gmKey}_${this.getId}"
    val gxmKeyWithId = s"${this.gxmKey}_${this.getId}"

    val needSync = if (parallism != 1) {
      ParameterSynchronizer.register(meanKeyWithId, parallism)
      ParameterSynchronizer.register(stdKeyWithId, parallism)
      ParameterSynchronizer.register(gmKeyWithId, parallism)
      ParameterSynchronizer.register(gxmKeyWithId, parallism)
      true
    } else false

    checkInputDim(input)
    output.resizeAs(input)

    _input.set(input)
    makeBatch(_input)
    val nInput = if (dataFormat == DataFormat.NCHW) {
      _input.size(2)
    } else {
      _input.size(4)
    }

    if (runningMean.nElement == 0 || runningMean.nElement < nInput) {
      initializeBuffer(nInput)
    }

    saveMean.resizeAs(runningMean).zero
    saveStd.resizeAs(runningVar).fill(ev.zero)

    val nChannels = _input.size(2)

    if (globalMean.size < nChannels) {
      globalMean = new Array[T](nChannels)
    }

    if (globalStd.size < nChannels) {
      globalStd = new Array[T](nChannels)
    }

    if (dataFormat == DataFormat.NCHW) {
      if (train) {
        if (ev.getType() == FloatType) {
          SpatialBatchNormalization.updateOutputNCHWTrainFloat(
            _input.asInstanceOf[Tensor[Float]], output.asInstanceOf[Tensor[Float]],
            saveMean.asInstanceOf[Tensor[Float]], saveStd.asInstanceOf[Tensor[Float]],
            runningMean.asInstanceOf[Tensor[Float]], runningVar.asInstanceOf[Tensor[Float]],
            weight.asInstanceOf[Tensor[Float]], bias.asInstanceOf[Tensor[Float]],
            eps.toFloat, momentum.toFloat, needFix = needFix,
            globalMean = globalMean.asInstanceOf[Array[Float]],
            globalStd = globalStd.asInstanceOf[Array[Float]],
            meanKey = meanKeyWithId, stdKey = stdKeyWithId, needSync = needSync)
        } else {
          SpatialBatchNormalization.updateOutputNCHWTrainDouble(
            _input.asInstanceOf[Tensor[Double]], output.asInstanceOf[Tensor[Double]],
            saveMean.asInstanceOf[Tensor[Double]], saveStd.asInstanceOf[Tensor[Double]],
            runningMean.asInstanceOf[Tensor[Double]], runningVar.asInstanceOf[Tensor[Double]],
            weight.asInstanceOf[Tensor[Double]], bias.asInstanceOf[Tensor[Double]],
            eps, momentum, needFix = needFix,
            globalMean = globalMean.asInstanceOf[Array[Double]],
            globalStd = globalStd.asInstanceOf[Array[Double]],
            meanKey = meanKeyWithId, stdKey = stdKeyWithId, needSync = needSync)
        }
      } else {
        if (ev.getType() == FloatType) {
          SpatialBatchNormalization.updateOutputNCHWInferFloat(
            _input.asInstanceOf[Tensor[Float]], output.asInstanceOf[Tensor[Float]],
            runningMean.asInstanceOf[Tensor[Float]], runningVar.asInstanceOf[Tensor[Float]],
            weight.asInstanceOf[Tensor[Float]], bias.asInstanceOf[Tensor[Float]], eps.toFloat)
        } else {
          SpatialBatchNormalization.updateOutputNCHWInferDouble(
            _input.asInstanceOf[Tensor[Double]], output.asInstanceOf[Tensor[Double]],
            runningMean.asInstanceOf[Tensor[Double]], runningVar.asInstanceOf[Tensor[Double]],
            weight.asInstanceOf[Tensor[Double]], bias.asInstanceOf[Tensor[Double]], eps)
        }
      }
    } else {
      if (train) {
        if (ev.getType() == FloatType) {
          SpatialBatchNormalization.updateOutputNHWCTrainFloat(
            _input.asInstanceOf[Tensor[Float]], output.asInstanceOf[Tensor[Float]],
            saveMean.asInstanceOf[Tensor[Float]], saveStd.asInstanceOf[Tensor[Float]],
            runningMean.asInstanceOf[Tensor[Float]], runningVar.asInstanceOf[Tensor[Float]],
            weight.asInstanceOf[Tensor[Float]], bias.asInstanceOf[Tensor[Float]],
            eps.toFloat, momentum.toFloat)
        } else {
          SpatialBatchNormalization.updateOutputNHWCTrainDouble(
            _input.asInstanceOf[Tensor[Double]], output.asInstanceOf[Tensor[Double]],
            saveMean.asInstanceOf[Tensor[Double]], saveStd.asInstanceOf[Tensor[Double]],
            runningMean.asInstanceOf[Tensor[Double]], runningVar.asInstanceOf[Tensor[Double]],
            weight.asInstanceOf[Tensor[Double]], bias.asInstanceOf[Tensor[Double]],
            eps.toFloat, momentum.toFloat)
        }
      } else {
        if (ev.getType() == FloatType) {
          SpatialBatchNormalization.updateOutputNHWCInferFloat(
            _input.asInstanceOf[Tensor[Float]], output.asInstanceOf[Tensor[Float]],
            runningMean.asInstanceOf[Tensor[Float]], runningVar.asInstanceOf[Tensor[Float]],
            weight.asInstanceOf[Tensor[Float]], bias.asInstanceOf[Tensor[Float]], eps.toFloat)
        } else {
          SpatialBatchNormalization.updateOutputNHWCInferDouble(
            _input.asInstanceOf[Tensor[Double]], output.asInstanceOf[Tensor[Double]],
            runningMean.asInstanceOf[Tensor[Double]], runningVar.asInstanceOf[Tensor[Double]],
            weight.asInstanceOf[Tensor[Double]], bias.asInstanceOf[Tensor[Double]], eps)
        }
      }
    }

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val gmKeyWithId = s"${this.gmKey}_${this.getId}"
    val gxmKeyWithId = s"${this.gxmKey}_${this.getId}"
    val needSync = getParallism() != None && getParallism().get > 1
    _gradOutput.set(gradOutput)
    makeBatch(_gradOutput)
    gxMean.zero()
    gMean.zero()
    val nChannel = _gradOutput.size(2)
    if (globalGMean.size < nChannel) {
      globalGMean = new Array[T](nChannel)
    }
    if (globalGxmMean.size < nChannel) {
      globalGxmMean = new Array[T](nChannel)
    }
    if (dataFormat == DataFormat.NCHW) {
      if (train) {
        if (ev.getType() == FloatType) {
          SpatialBatchNormalization.updateGradInputNCHWTrainFloat(
            _input.asInstanceOf[Tensor[Float]], _gradOutput.asInstanceOf[Tensor[Float]],
            gradInput.asInstanceOf[Tensor[Float]], weight.asInstanceOf[Tensor[Float]],
            saveMean.asInstanceOf[Tensor[Float]], saveStd.asInstanceOf[Tensor[Float]],
            gMean.asInstanceOf[Tensor[Float]], gxMean.asInstanceOf[Tensor[Float]],
            globalGMean.asInstanceOf[Array[Float]], globalGxmMean.asInstanceOf[Array[Float]],
            gMeanKey = gmKeyWithId, gxMeanKey = gxmKeyWithId, needSync = needSync)
        } else {
          SpatialBatchNormalization.updateGradInputNCHWTrainDouble(
            _input.asInstanceOf[Tensor[Double]], _gradOutput.asInstanceOf[Tensor[Double]],
            gradInput.asInstanceOf[Tensor[Double]], weight.asInstanceOf[Tensor[Double]],
            saveMean.asInstanceOf[Tensor[Double]], saveStd.asInstanceOf[Tensor[Double]],
            gMean.asInstanceOf[Tensor[Double]], gxMean.asInstanceOf[Tensor[Double]],
            globalGMean.asInstanceOf[Array[Double]], globalGxmMean.asInstanceOf[Array[Double]],
            gMeanKey = gmKeyWithId, gxMeanKey = gxmKeyWithId, needSync = needSync)
        }
      } else {
        if (ev.getType() == FloatType) {
          SpatialBatchNormalization.updateGradInputNCHWInferFloat(
            _gradOutput.asInstanceOf[Tensor[Float]],
            gradInput.asInstanceOf[Tensor[Float]], weight.asInstanceOf[Tensor[Float]],
            bias.asInstanceOf[Tensor[Float]])
        } else {
          SpatialBatchNormalization.updateGradInputNCHWInferDouble(
            _gradOutput.asInstanceOf[Tensor[Double]],
            gradInput.asInstanceOf[Tensor[Double]], weight.asInstanceOf[Tensor[Double]],
            bias.asInstanceOf[Tensor[Double]])
        }
      }
    } else {
      if (train) {
        if (ev.getType() == FloatType) {
          SpatialBatchNormalization.updateGradInputNHWCTrainFloat(
            _input.asInstanceOf[Tensor[Float]], _gradOutput.asInstanceOf[Tensor[Float]],
            gradInput.asInstanceOf[Tensor[Float]], weight.asInstanceOf[Tensor[Float]],
            saveMean.asInstanceOf[Tensor[Float]], saveStd.asInstanceOf[Tensor[Float]],
            gMean.asInstanceOf[Tensor[Float]], gxMean.asInstanceOf[Tensor[Float]])
        } else {
          SpatialBatchNormalization.updateGradInputNHWCTrainDouble(
            _input.asInstanceOf[Tensor[Double]], _gradOutput.asInstanceOf[Tensor[Double]],
            gradInput.asInstanceOf[Tensor[Double]], weight.asInstanceOf[Tensor[Double]],
            saveMean.asInstanceOf[Tensor[Double]], saveStd.asInstanceOf[Tensor[Double]],
            gMean.asInstanceOf[Tensor[Double]], gxMean.asInstanceOf[Tensor[Double]])
        }
      } else {
        if (ev.getType() == FloatType) {
          SpatialBatchNormalization.updateGradInputNHWCInferFloat(
            _gradOutput.asInstanceOf[Tensor[Float]],
            gradInput.asInstanceOf[Tensor[Float]], weight.asInstanceOf[Tensor[Float]],
            bias.asInstanceOf[Tensor[Float]])
        } else {
          SpatialBatchNormalization.updateGradInputNHWCInferDouble(
            _gradOutput.asInstanceOf[Tensor[Double]],
            gradInput.asInstanceOf[Tensor[Double]], weight.asInstanceOf[Tensor[Double]],
            bias.asInstanceOf[Tensor[Double]])
        }
      }
    }
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    if (weight == null || scaleW == 0) {
      return
    }

    if (dataFormat == DataFormat.NCHW) {
      if (ev.getType() == FloatType) {
        SpatialBatchNormalization.accGradientNCHWFloat(_gradOutput.asInstanceOf[Tensor[Float]],
          gradWeight.asInstanceOf[Tensor[Float]], gradBias.asInstanceOf[Tensor[Float]],
          _input.asInstanceOf[Tensor[Float]], saveMean.asInstanceOf[Tensor[Float]],
          saveStd.asInstanceOf[Tensor[Float]], scaleW.toFloat, scaleB.toFloat)
      } else {
        SpatialBatchNormalization.accGradientNCHWDouble(_gradOutput.asInstanceOf[Tensor[Double]],
          gradWeight.asInstanceOf[Tensor[Double]], gradBias.asInstanceOf[Tensor[Double]],
          _input.asInstanceOf[Tensor[Double]], saveMean.asInstanceOf[Tensor[Double]],
          saveStd.asInstanceOf[Tensor[Double]], scaleW, scaleB)
      }
    } else {
      if (ev.getType() == FloatType) {
        SpatialBatchNormalization.accGradientNHWCFloat(_gradOutput.asInstanceOf[Tensor[Float]],
          gradWeight.asInstanceOf[Tensor[Float]], gradBias.asInstanceOf[Tensor[Float]],
          _input.asInstanceOf[Tensor[Float]], saveMean.asInstanceOf[Tensor[Float]],
          saveStd.asInstanceOf[Tensor[Float]], scaleW.toFloat, scaleB.toFloat)
      } else {
        SpatialBatchNormalization.accGradientNHWCDouble(_gradOutput.asInstanceOf[Tensor[Double]],
          gradWeight.asInstanceOf[Tensor[Double]], gradBias.asInstanceOf[Tensor[Double]],
          _input.asInstanceOf[Tensor[Double]], saveMean.asInstanceOf[Tensor[Double]],
          saveStd.asInstanceOf[Tensor[Double]], scaleW, scaleB)
      }
    }
  }

  override def toString(): String = {
    s"${getPrintName}[${ev.getType()}]($nOutput, $eps, $momentum, $affine)"
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
    dataFormat: DataFormat = DataFormat.NCHW
  )(implicit ev: TensorNumeric[T])
  : SpatialBatchNormalization[T] = {
    new SpatialBatchNormalization[T](nOutput, eps, momentum, affine,
      initWeight, initBias, initGradWeight, initGradBias, dataFormat)
  }

  private[bigdl] def updateOutputNHWCInferFloat(input: Tensor[Float], output: Tensor[Float],
    mean: Tensor[Float], variance: Tensor[Float], scale: Tensor[Float], offset: Tensor[Float],
    eps: Float): Unit = {

    require(input.isContiguous(), "BatchNorm NHWC require a contiguous input")
    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1
    val outputData = output.storage().array()
    val outputOffset = output.storageOffset() - 1
    val nChannels = input.size(4)
    val n = input.nElement()
    val meanData = mean.storage().array()
    val meanOffset = mean.storageOffset() - 1
    val varData = variance.storage().array()
    val varOffset = variance.storageOffset() - 1

    if (scale != null) {
      val scaleData = scale.storage().array()
      val scaleOffset = scale.storageOffset() - 1
      val offsetData = offset.storage().array()
      val offsetOffset = offset.storageOffset() - 1
      var isIdenticalScale = false
      var identicalScale = 0.0f

      if (scaleData.length == 1) {
        isIdenticalScale = true
        identicalScale = scaleData(0)
      }
      if (isIdenticalScale) {
        var i = 0
        while (i < n) {
          var c = 0
          while (c < nChannels) {
            val invStd = 1 / Math.sqrt(varData(varOffset + c) + eps).toFloat
            outputData(i + outputOffset + c) = (inputData(i + inputOffset + c) -
              meanData(c + meanOffset)) * invStd * identicalScale +
              offsetData(offsetOffset + c)
            c += 1
          }
          i += nChannels
        }
      } else {
        var i = 0
        while (i < n) {
          var c = 0
          while (c < nChannels) {
            val invStd = 1 / Math.sqrt(varData(varOffset + c) + eps).toFloat
            outputData(i + outputOffset + c) = (inputData(i + inputOffset + c) -
              meanData(c + meanOffset)) * invStd * scaleData(scaleOffset + c) +
              offsetData(offsetOffset + c)
            c += 1
          }
          i += nChannels
        }
      }
    } else {
      var i = 0
      while (i < n) {
        var c = 0
        while (c < nChannels) {
          val invStd = 1 / Math.sqrt(varData(varOffset + c) + eps).toFloat
          outputData(i + outputOffset + c) = (inputData(i + inputOffset + c) -
            meanData(c + meanOffset)) * invStd
          c += 1
        }
        i += nChannels
      }
    }
  }

  private[bigdl] def updateOutputNHWCInferDouble(input: Tensor[Double], output: Tensor[Double],
    mean: Tensor[Double], variance: Tensor[Double], scale: Tensor[Double], offset: Tensor[Double],
    eps: Double): Unit = {

    require(input.isContiguous(), "BatchNorm NHWC require a contiguous input")
    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1
    val outputData = output.storage().array()
    val outputOffset = output.storageOffset() - 1
    val nChannels = input.size(4)
    val n = input.nElement()
    val meanData = mean.storage().array()
    val meanOffset = mean.storageOffset() - 1
    val varData = variance.storage().array()
    val varOffset = variance.storageOffset() - 1

    if (scale != null) {
      val scaleData = scale.storage().array()
      val scaleOffset = scale.storageOffset() - 1
      val offsetData = offset.storage().array()
      val offsetOffset = offset.storageOffset() - 1
      var i = 0
      while (i < n) {
        var c = 0
        while (c < nChannels) {
          val invStd = 1 / Math.sqrt(varData(varOffset + c) + eps)
          outputData(i + outputOffset + c) = (inputData(i + inputOffset + c) -
            meanData(meanOffset + c)) * invStd * scaleData(scaleOffset + c) +
            offsetData(offsetOffset + c)
          c += 1
        }
        i += nChannels
      }
    } else {
      var i = 0
      while (i < n) {
        var c = 0
        while (c < nChannels) {
          val invStd = 1 / Math.sqrt(varData(varOffset + c) + eps)
          outputData(i + outputOffset + c) = (inputData(i + inputOffset + c) -
            meanData(meanOffset + c)) * invStd
          c += 1
        }
        i += nChannels
      }
    }
  }

  private[bigdl] def updateOutputNHWCTrainFloat(input: Tensor[Float], output: Tensor[Float],
    saveMean: Tensor[Float], saveStd: Tensor[Float], runningMean: Tensor[Float],
    runningVar: Tensor[Float], scale: Tensor[Float], offset: Tensor[Float],
    eps: Float, momentum: Float,
    batchVar: Tensor[Float] = null, saveVar: Tensor[Float] = null): Unit = {
    require(input.isContiguous(), "BatchNorm NHWC require a contiguous input")
    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1
    val outputData = output.storage().array()
    val outputOffset = output.storageOffset() - 1
    val nChannels = input.size(4)
    if(saveMean.size(1) != nChannels) {
      saveMean.resize(nChannels)
      saveStd.resize(nChannels)
      runningMean.resize(nChannels)
      runningVar.resize(nChannels)
    }
    val meanData = saveMean.storage().array()
    val meanOffset = saveMean.storageOffset() - 1
    var i = 0
    val n = input.nElement()
    val frameSize = n / nChannels
    while(i < n) {
      var c = 0
      while(c < nChannels) {
        meanData(meanOffset + c) += inputData(inputOffset + i + c)
        c += 1
      }
      i += nChannels
    }

    var c = 0
    val runningMeanData = runningMean.storage().array()
    val runningMeanDataOffset = runningMean.storageOffset() - 1
    while(c < nChannels) {
      meanData(meanOffset + c) /= frameSize
      runningMeanData(runningMeanDataOffset + c) = meanData(meanOffset + c) * momentum +
        (1 - momentum) * runningMeanData(c + runningMeanDataOffset)
      c += 1
    }

    val stdData = saveStd.storage().array()
    val stdOffset = saveStd.storageOffset() - 1
    i = 0
    while(i < n) {
      var c = 0
      while(c < nChannels) {
        val diff = (inputData(inputOffset + i + c) - meanData(meanOffset + c))
        stdData(stdOffset + c) += diff * diff
        c += 1
      }
      i += nChannels
    }

    c = 0
    val runningVarData = runningVar.storage().array()
    val runningVarOffset = runningVar.storageOffset() - 1
    while(c < nChannels) {
      if (stdData(c + stdOffset) == 0 && eps == 0) {
        stdData(c + stdOffset) = 0
        if (saveVar != null) {
          saveVar.setValue(c + 1, 0f)
        }
        if (batchVar != null) {
          batchVar.setValue(c + 1, 0f)
        }
      } else {
        val s = stdData(c + stdOffset)
        val unbiasedVar = s / (frameSize - 1)
        if (saveVar != null) {
          saveVar.setValue(c + 1, s / frameSize)
        }
        if (batchVar != null) {
          batchVar.setValue(c + 1, unbiasedVar)
        }
        stdData(c + stdOffset) = 1.0f / Math.sqrt(s / frameSize + eps).toFloat
        runningVarData(c + runningVarOffset) = momentum * unbiasedVar +
          (1 - momentum) * runningVarData(c + runningVarOffset)
      }
      c += 1
    }

    if (scale != null) {
      val scaleData = scale.storage().array()
      val scaleOffset = scale.storageOffset() - 1
      val offsetData = offset.storage().array()
      val offsetOffset = offset.storageOffset() - 1
      i = 0
      while (i < n) {
        var c = 0
        while (c < nChannels) {
          outputData(i + outputOffset + c) = (inputData(i + inputOffset + c) -
            meanData(meanOffset + c)) * stdData(c + stdOffset) *
            scaleData(scaleOffset + c) + offsetData(offsetOffset + c)
          c += 1
        }
        i += nChannels
      }
    } else {
      i = 0
      while (i < n) {
        var c = 0
        while (c < nChannels) {
          outputData(i + outputOffset + c) = (inputData(i + inputOffset + c) -
            meanData(meanOffset + c)) * stdData(c + stdOffset)
          c += 1
        }
        i += nChannels
      }
    }
  }

  private[bigdl] def updateOutputNHWCTrainDouble(input: Tensor[Double], output: Tensor[Double],
    saveMean: Tensor[Double], saveStd: Tensor[Double], runningMean: Tensor[Double],
    runningVar: Tensor[Double], scale: Tensor[Double], offset: Tensor[Double],
    eps: Double, momentum: Double,
    batchVar: Tensor[Double] = null, saveVar: Tensor[Double] = null): Unit = {
    require(input.isContiguous(), "BatchNorm NHWC require a contiguous input")
    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1
    val outputData = output.storage().array()
    val outputOffset = output.storageOffset() - 1
    val nChannels = input.size(4)
    if(saveMean.size(1) != nChannels) {
      saveMean.resize(nChannels)
      saveStd.resize(nChannels)
      runningMean.resize(nChannels)
      runningVar.resize(nChannels)
    }
    val meanData = saveMean.storage().array()
    val meanOffset = saveMean.storageOffset() - 1
    var i = 0
    val n = input.nElement()
    val frameSize = n / nChannels
    while(i < n) {
      var c = 0
      while(c < nChannels) {
        meanData(c + meanOffset) += inputData(inputOffset + i + c)
        c += 1
      }
      i += nChannels
    }

    var c = 0
    val runningMeanData = runningMean.storage().array()
    val runningMeanOffset = runningMean.storageOffset() - 1
    while(c < nChannels) {
      meanData(c + meanOffset) /= frameSize
      runningMeanData(c + runningMeanOffset) = meanData(c + meanOffset) * momentum +
        (1 - momentum) * runningMeanData(c + runningMeanOffset)
      c += 1
    }

    val stdData = saveStd.storage().array()
    val stdOffset = saveStd.storageOffset() - 1
    i = 0
    while(i < n) {
      var c = 0
      while(c < nChannels) {
        val diff = (inputData(inputOffset + i + c) - meanData(c + meanOffset))
        stdData(c + stdOffset) += diff * diff
        c += 1
      }
      i += nChannels
    }

    c = 0
    val runningVarData = runningVar.storage().array()
    val runningVarOffset = runningVar.storageOffset() - 1
    while(c < nChannels) {
      if (stdData(c + stdOffset) == 0 && eps == 0) {
        stdData(c + stdOffset) = 0
        if (saveVar != null) {
          saveVar.setValue(c + 1, 0f)
        }
        if (batchVar != null) {
          batchVar.setValue(c + 1, 0f)
        }
      } else {
        val s = stdData(c + stdOffset)
        val unbiasedVar = s / (frameSize - 1)
        if (saveVar != null) {
          saveVar.setValue(c + 1, s / frameSize)
        }
        if (batchVar != null) {
          batchVar.setValue(c + 1, unbiasedVar)
        }
        stdData(c + stdOffset) = 1.0f / Math.sqrt(s / frameSize + eps).toFloat
        runningVarData(c + runningVarOffset) = momentum * unbiasedVar +
          (1 - momentum) * runningVarData(c + runningVarOffset)
      }
      c += 1
    }

    if (scale != null) {
      val scaleData = scale.storage().array()
      val scaleOffset = scale.storageOffset() - 1
      val offsetData = offset.storage().array()
      val offsetOffset = offset.storageOffset() - 1
      i = 0
      while (i < n) {
        var c = 0
        while (c < nChannels) {
          outputData(i + outputOffset + c) = (inputData(i + inputOffset + c) -
            meanData(c + meanOffset)) * stdData(c + stdOffset) *
            scaleData(c + scaleOffset) + offsetData(c + offsetOffset)
          c += 1
        }
        i += nChannels
      }
    } else {
      i = 0
      while (i < n) {
        var c = 0
        while (c < nChannels) {
          outputData(i + outputOffset + c) = (inputData(i + inputOffset + c) -
            meanData(c + meanOffset)) * stdData(c + stdOffset)
          c += 1
        }
        i += nChannels
      }
    }
  }

  private[bigdl] def updateOutputNCHWInferFloat(input: Tensor[Float], output: Tensor[Float],
    mean: Tensor[Float], variance: Tensor[Float], scale: Tensor[Float],
    offset: Tensor[Float], eps: Float): Unit = {

    require(input.isContiguous(), "BatchNorm NCHW require a contiguous input")
    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1
    val outputData = output.storage().array()
    val outputOffset = output.storageOffset() - 1
    val meanData = mean.storage().array()
    val meanOffset = mean.storageOffset() - 1
    val varData = variance.storage().array()
    val varOffset = variance.storageOffset() - 1
    val nChannels = input.size(2)
    val nBatch = input.size(1)
    val nFrame = input.size(3) * input.size(4)

    if (scale != null) {
      val scaleData = scale.storage().array()
      val scaleOffset = scale.storageOffset() - 1
      val offsetData = offset.storage().array()
      val offsetOffset = offset.storageOffset() - 1
      var i = 0
      var b = 0
      while (b < nBatch) {
        var c = 0
        while (c < nChannels) {
          var k = 0
          while (k < nFrame) {
            val invStd = 1 / Math.sqrt(varData(varOffset + c) + eps).toFloat
            outputData(i + outputOffset) = (inputData(i + inputOffset) - meanData(c + meanOffset)) *
              invStd * scaleData(c + scaleOffset) + offsetData(c + offsetOffset)
            k += 1
            i += 1
          }
          c += 1
        }
        b += 1
      }
    } else {
      var i = 0
      var b = 0
      while (b < nBatch) {
        var c = 0
        while (c < nChannels) {
          var k = 0
          while (k < nFrame) {
            val invStd = 1 / Math.sqrt(varData(varOffset + c) + eps).toFloat
            outputData(i + outputOffset) = (inputData(i + inputOffset) - meanData(c + meanOffset)) *
              invStd
            k += 1
            i += 1
          }
          c += 1
        }
        b += 1
      }
    }
  }

  private[bigdl] def updateOutputNCHWInferDouble(input: Tensor[Double], output: Tensor[Double],
    mean: Tensor[Double], variance: Tensor[Double], scale: Tensor[Double], offset: Tensor[Double],
    eps: Double)
  : Unit = {

    require(input.isContiguous(), "BatchNorm NCHW require a contiguous input")
    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1
    val outputData = output.storage().array()
    val outputOffset = output.storageOffset() - 1
    val meanData = mean.storage().array()
    val meanOffset = mean.storageOffset() - 1
    val varData = variance.storage().array()
    val varOffset = variance.storageOffset() - 1
    val nChannels = input.size(2)
    val nBatch = input.size(1)
    val nFrame = input.size(3) * input.size(4)

    if (scale != null) {
      val scaleData = scale.storage().array()
      val scaleOffset = scale.storageOffset() - 1
      val offsetData = offset.storage().array()
      val offsetOffset = offset.storageOffset() - 1
      var i = 0
      var b = 0
      while (b < nBatch) {
        var c = 0
        while (c < nChannels) {
          var k = 0
          while (k < nFrame) {
            val invStd = 1 / Math.sqrt(varData(varOffset + c) + eps)
            outputData(i + outputOffset) = (inputData(i + inputOffset) - meanData(c + meanOffset)) *
              invStd * scaleData(c + scaleOffset) + offsetData(c + offsetOffset)
            k += 1
            i += 1
          }
          c += 1
        }
        b += 1
      }
    } else {
      var i = 0
      var b = 0
      while (b < nBatch) {
        var c = 0
        while (c < nChannels) {
          var k = 0
          while (k < nFrame) {
            val invStd = 1 / Math.sqrt(varData(varOffset + c) + eps)
            outputData(i + outputOffset) = (inputData(i + inputOffset) - meanData(c + meanOffset)) *
              invStd
            k += 1
            i += 1
          }
          c += 1
        }
        b += 1
      }
    }
  }

  private[bigdl] def updateGradInputNHWCTrainFloat(
    input: Tensor[Float],
    gradOutput: Tensor[Float],
    gradInput: Tensor[Float],
    scale: Tensor[Float],
    saveMean: Tensor[Float],
    saveStd: Tensor[Float],
    gMean: Tensor[Float],
    gxMean: Tensor[Float]
  ): Unit = {
    require(input.nDimension() == 4, "BN require a 4D input")
    require(input.isContiguous(), "input is not contiguous")
    require(gradOutput.nDimension() == 4, "BN require a 4D gradient")
    require(gradOutput.isContiguous(), "gradient is not contiguous")
    val nChannel = gradOutput.size(4)
    require(saveMean.size(1) == nChannel, "saveMean length is not consistent with channel number")
    require(saveStd.size(1) == nChannel, "saveStd length is not consistent with channel number")

    gradInput.resizeAs(gradOutput)
    if (gMean.isEmpty) {
      gMean.resize(nChannel)
      gxMean.resize(nChannel)
    }

    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1
    val gradOutputData = gradOutput.storage().array()
    val gradOutputOffset = gradOutput.storageOffset() - 1
    val gradInputData = gradInput.storage().array()
    val gradInputOffset = gradInput.storageOffset() - 1
    val saveMeanData = saveMean.storage().array()
    val saveMeanOffset = saveMean.storageOffset() - 1
    val saveStdData = saveStd.storage().array()
    val saveStdOffset = saveStd.storageOffset() - 1
    val gMeanData = gMean.storage().array()
    val gxMeanData = gxMean.storage().array()

    val n = gradOutput.nElement()
    var i = 0
    while(i < n) {
      var c = 0
      while(c < nChannel) {
        gMeanData(c) += gradOutputData(i + gradOutputOffset)
        gxMeanData(c) += gradOutputData(i + gradOutputOffset) *
          (inputData(i + inputOffset) - saveMeanData(c + saveMeanOffset))
        c += 1
        i += 1
      }
    }

    var c = 0
    val size = n / nChannel
    while(c < nChannel) {
      gMeanData(c) /= size
      gxMeanData(c) /= size
      c += 1
    }

    if (scale != null) {
      require(scale.size(1) == nChannel, "scale length is not consistent with channel number")

      val scaleData = scale.storage().array()
      val scaleOffset = scale.storageOffset() - 1
      i = 0
      while (i < n) {
        var c = 0
        while (c < nChannel) {
          val invStd = saveStdData(saveStdOffset + c)
          gradInputData(gradInputOffset + i) = scaleData(scaleOffset + c) *
            invStd * (gradOutputData(gradOutputOffset + i) - gMeanData(c) -
            gxMeanData(c) * invStd * invStd * (inputData(inputOffset + i) -
              saveMeanData(saveMeanOffset + c)))
          c += 1
          i += 1
        }
      }
    } else {
      i = 0
      while (i < n) {
        var c = 0
        while (c < nChannel) {
          val invStd = saveStdData(saveStdOffset + c)
          gradInputData(gradInputOffset + i) =
            invStd * (gradOutputData(gradOutputOffset + i) - gMeanData(c) -
            gxMeanData(c) * invStd * invStd * (inputData(inputOffset + i) -
              saveMeanData(saveMeanOffset + c)))
          c += 1
          i += 1
        }
      }
    }
  }

  private[bigdl] def updateGradInputNHWCTrainDouble(
    input: Tensor[Double],
    gradOutput: Tensor[Double],
    gradInput: Tensor[Double],
    scale: Tensor[Double],
    saveMean: Tensor[Double],
    saveStd: Tensor[Double],
    gMean: Tensor[Double],
    gxMean: Tensor[Double]
  ): Unit = {
    require(input.nDimension() == 4, "BN require a 4D input")
    require(input.isContiguous(), "input is not contiguous")
    require(gradOutput.nDimension() == 4, "BN require a 4D gradient")
    require(gradOutput.isContiguous(), "gradient is not contiguous")
    val nChannel = gradOutput.size(4)
    require(saveMean.size(1) == nChannel, "saveMean length is not consistent with channel number")
    require(saveStd.size(1) == nChannel, "saveStd length is not consistent with channel number")

    gradInput.resizeAs(gradOutput)
    if (gMean.isEmpty) {
      gMean.resize(nChannel)
      gxMean.resize(nChannel)
    }

    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1
    val gradOutputData = gradOutput.storage().array()
    val gradOutputOffset = gradOutput.storageOffset() - 1
    val gradInputData = gradInput.storage().array()
    val gradInputOffset = gradInput.storageOffset() - 1
    val saveMeanData = saveMean.storage().array()
    val saveMeanOffset = saveMean.storageOffset() - 1
    val saveStdData = saveStd.storage().array()
    val saveStdOffset = saveStd.storageOffset() - 1
    val gMeanData = gMean.storage().array()
    val gxMeanData = gxMean.storage().array()

    val n = gradOutput.nElement()
    var i = 0
    while(i < n) {
      var c = 0
      while(c < nChannel) {
        gMeanData(c) += gradOutputData(i + gradOutputOffset)
        gxMeanData(c) += gradOutputData(i + gradOutputOffset) *
          (inputData(i + inputOffset) - saveMeanData(c + saveMeanOffset))
        c += 1
        i += 1
      }
    }

    var c = 0
    val size = n / nChannel
    while(c < nChannel) {
      gMeanData(c) /= size
      gxMeanData(c) /= size
      c += 1
    }

    if (scale != null) {
      require(scale.size(1) == nChannel, "scale length is not consistent with channel number")
      val scaleData = scale.storage().array()
      val scaleOffset = scale.storageOffset() - 1
      i = 0
      while (i < n) {
        var c = 0
        while (c < nChannel) {
          val invStd = saveStdData(saveStdOffset + c)
          gradInputData(gradInputOffset + i) = scaleData(scaleOffset + c) *
            invStd * (gradOutputData(gradOutputOffset + i) - gMeanData(c) -
            gxMeanData(c) * invStd * invStd * (inputData(inputOffset + i) -
              saveMeanData(saveMeanOffset + c)))
          c += 1
          i += 1
        }
      }
    } else {
      i = 0
      while (i < n) {
        var c = 0
        while (c < nChannel) {
          val invStd = saveStdData(saveStdOffset + c)
          gradInputData(gradInputOffset + i) =
            invStd * (gradOutputData(gradOutputOffset + i) - gMeanData(c) -
            gxMeanData(c) * invStd * invStd * (inputData(inputOffset + i) -
              saveMeanData(saveMeanOffset + c)))
          c += 1
          i += 1
        }
      }
    }
  }

  private[bigdl] def updateGradInputNHWCInferFloat(
    gradOutput: Tensor[Float],
    gradInput: Tensor[Float],
    scale: Tensor[Float],
    saveStd: Tensor[Float]
  ): Unit = {
    require(gradOutput.nDimension() == 4, "BN require a 4D gradient")
    require(gradOutput.isContiguous(), "gradient is not contiguous")
    val nChannel = gradOutput.size(4)
    require(saveStd.size(1) == nChannel, "saveStd length is not consistent with channel number")

    gradInput.resizeAs(gradOutput)
    val gradOutputData = gradOutput.storage().array()
    val gradOutputOffset = gradOutput.storageOffset() - 1
    val gradInputData = gradInput.storage().array()
    val gradInputOffset = gradInput.storageOffset() - 1
    val saveStdData = saveStd.storage().array()
    val saveStdOffset = saveStd.storageOffset() - 1

    val n = gradOutput.nElement()

    if (scale != null) {
      require(scale.size(1) == nChannel, "scale length is not consistent with channel number")
      val scaleData = scale.storage().array()
      val scaleOffset = scale.storageOffset() - 1
      var i = 0
      while (i < n) {
        var c = 0
        while (c < nChannel) {
          val invStd = saveStdData(saveStdOffset + c)
          gradInputData(gradInputOffset + i) = scaleData(scaleOffset + c) *
            invStd * gradOutputData(gradOutputOffset + i)
          c += 1
          i += 1
        }
      }
    } else {
      var i = 0
      while (i < n) {
        var c = 0
        while (c < nChannel) {
          val invStd = saveStdData(saveStdOffset + c)
          gradInputData(gradInputOffset + i) =
            invStd * gradOutputData(gradOutputOffset + i)
          c += 1
          i += 1
        }
      }
    }
  }

  private[bigdl] def updateGradInputNHWCInferDouble(
    gradOutput: Tensor[Double],
    gradInput: Tensor[Double],
    scale: Tensor[Double],
    saveStd: Tensor[Double]
  ): Unit = {
    require(gradOutput.nDimension() == 4, "BN require a 4D gradient")
    require(gradOutput.isContiguous(), "gradient is not contiguous")
    val nChannel = gradOutput.size(4)
    require(saveStd.size(1) == nChannel, "saveStd length is not consistent with channel number")

    gradInput.resizeAs(gradOutput)
    val gradOutputData = gradOutput.storage().array()
    val gradOutputOffset = gradOutput.storageOffset() - 1
    val gradInputData = gradInput.storage().array()
    val gradInputOffset = gradInput.storageOffset() - 1
    val saveStdData = saveStd.storage().array()
    val saveStdOffset = saveStd.storageOffset() - 1

    val n = gradOutput.nElement()
    var i = 0

    if (scale != null) {
      require(scale.size(1) == nChannel, "scale length is not consistent with channel number")
      val scaleData = scale.storage().array()
      val scaleOffset = scale.storageOffset() - 1
      while (i < n) {
        var c = 0
        while (c < nChannel) {
          val invStd = saveStdData(saveStdOffset + c)
          gradInputData(gradInputOffset + i) = scaleData(scaleOffset + c) *
            invStd * gradOutputData(gradOutputOffset + i)
          c += 1
          i += 1
        }
      }
    } else {
      while (i < n) {
        var c = 0
        while (c < nChannel) {
          val invStd = saveStdData(saveStdOffset + c)
          gradInputData(gradInputOffset + i) =
            invStd * gradOutputData(gradOutputOffset + i)
          c += 1
          i += 1
        }
      }
    }
  }

  private[bigdl] def updateGradInputNCHWTrainFloat(
    input: Tensor[Float],
    gradOutput: Tensor[Float],
    gradInput: Tensor[Float],
    scale: Tensor[Float],
    saveMean: Tensor[Float],
    saveStd: Tensor[Float],
    gMean: Tensor[Float],
    gxMean: Tensor[Float],
    globalGmean: Array[Float],
    globalGxmean: Array[Float],
    gMeanKey: String = null,
    gxMeanKey: String = null,
    needSync: Boolean = false
  ): Unit = {
    require(input.nDimension() == 4, "BN require a 4D input")
    require(input.isContiguous(), "input is not contiguous")
    require(gradOutput.nDimension() == 4, "BN require a 4D gradient")
    require(gradOutput.isContiguous(), "gradient is not contiguous")
    val nChannel = gradOutput.size(2)
    require(saveMean.size(1) == nChannel, "saveMean length is not consistent with channel number")
    require(saveStd.size(1) == nChannel, "saveStd length is not consistent with channel number")

    gradInput.resizeAs(gradOutput)
    if (gMean.isEmpty) {
      gMean.resize(nChannel)
      gxMean.resize(nChannel)
    }

    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1
    val gradOutputData = gradOutput.storage().array()
    val gradOutputOffset = gradOutput.storageOffset() - 1
    val gradInputData = gradInput.storage().array()
    val gradInputOffset = gradInput.storageOffset() - 1
    val saveMeanData = saveMean.storage().array()
    val saveMeanOffset = saveMean.storageOffset() - 1
    val saveStdData = saveStd.storage().array()
    val saveStdOffset = saveStd.storageOffset() - 1
    val gMeanData = gMean.storage().array()
    val gxMeanData = gxMean.storage().array()

    val nBatch = gradOutput.size(1)
    val frameSize = gradOutput.size(3) * gradOutput.size(4)
    val n = gradOutput.nElement()
    var b = 0
    var i = 0
    while(b < nBatch) {
      var c = 0
      while(c < nChannel) {
        var k = 0
        while(k < frameSize) {
          gMeanData(c) += gradOutputData(i + gradOutputOffset)
          gxMeanData(c) += gradOutputData(i + gradOutputOffset) *
            (inputData(i + inputOffset) - saveMeanData(c + saveMeanOffset))
          k += 1
          i += 1
        }
        c += 1
      }
      b += 1
    }

    var gmeanEventLen = 1
    var gmxmeanEventLen = 1

    if (needSync) {

      ParameterSynchronizer.syncData(gMeanKey, gMean)

      val gMeanEventData = ParameterSynchronizer.collect[Float](gMeanKey)
      var c = 0
      while (c < nChannel) {
        globalGmean(c) = 0.0f
        gMeanEventData.keySet.asScala.foreach(threadId => {
          val localGmean = gMeanEventData.get(threadId)
          val localGmeanOffset = localGmean.storageOffset() - 1
          globalGmean(c) += localGmean.storage.array()(c + localGmeanOffset)
        })
        c += 1
      }

      gmeanEventLen = gMeanEventData.size

      ParameterSynchronizer.reset[Float](gMeanKey)

      ParameterSynchronizer.syncData(gxMeanKey, gxMean)

      val gxMeanEventData = ParameterSynchronizer.collect[Float](gxMeanKey)

      c = 0
      while (c < nChannel) {
        globalGxmean(c) = 0.0f
        gxMeanEventData.keySet.asScala.foreach(threadId => {
          val localGxmean = gxMeanEventData.get(threadId)
          val localGxmeanOffset = localGxmean.storageOffset() - 1
          globalGxmean(c) += localGxmean.storage.array()(c + localGxmeanOffset)
        })
        c += 1
      }

      gmxmeanEventLen = gxMeanEventData.size

      ParameterSynchronizer.reset[Float](gxMeanKey)
    }

    var c = 0
    val size = n / nChannel
    while(c < nChannel) {
      if (needSync) {
        gMeanData(c) = globalGmean(c) / (size * gmeanEventLen)
        gxMeanData(c) = globalGxmean(c) / (size * gmxmeanEventLen)
      } else {
        gMeanData(c) = gMeanData(c) / (size * gmeanEventLen)
        gxMeanData(c) = gxMeanData(c) / (size * gmxmeanEventLen)
      }
      c += 1
    }

    i = 0
    b = 0
    if (scale != null) {
      require(scale.size(1) == nChannel, "scale length is not consistent with channel number")
      val scaleData = scale.storage().array()
      val scaleOffset = scale.storageOffset() - 1
      while (b < nBatch) {
        var c = 0
        while (c < nChannel) {
          var k = 0
          while (k < frameSize) {
            val invStd = saveStdData(saveStdOffset + c)
            gradInputData(gradInputOffset + i) = scaleData(scaleOffset + c) *
              invStd * (gradOutputData(gradOutputOffset + i) - gMeanData(c) -
              gxMeanData(c) * invStd * invStd * (inputData(inputOffset + i) -
                saveMeanData(saveMeanOffset + c)))
            k += 1
            i += 1
          }
          c += 1
        }
        b += 1
      }
    } else {
      while (b < nBatch) {
        var c = 0
        while (c < nChannel) {
          var k = 0
          while (k < frameSize) {
            val invStd = saveStdData(saveStdOffset + c)
            gradInputData(gradInputOffset + i) =
              invStd * (gradOutputData(gradOutputOffset + i) - gMeanData(c) -
              gxMeanData(c) * invStd * invStd * (inputData(inputOffset + i) -
                saveMeanData(saveMeanOffset + c)))
            k += 1
            i += 1
          }
          c += 1
        }
        b += 1
      }
    }
  }

  private[bigdl] def updateOutputNCHWTrainFloat(input: Tensor[Float], output: Tensor[Float],
    saveMean: Tensor[Float], saveStd: Tensor[Float], runningMean: Tensor[Float],
    runningVar: Tensor[Float], scale: Tensor[Float], offset: Tensor[Float],
    eps: Float, momentum: Float,
    batchVar: Tensor[Float] = null, saveVar: Tensor[Float] = null, needFix: Boolean = false,
                                                globalMean: Array[Float] = null,
                                                globalStd: Array[Float] = null,
                                                meanKey: String = null,
                                                stdKey: String = null,
                                                needSync: Boolean = false)
  : Unit = {
    require(input.isContiguous(), "BatchNorm NCHW require a contiguous input")
    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1
    val outputData = output.storage().array()
    val outputOffset = output.storageOffset() - 1
    val nChannels = input.size(2)
    val nBatch = input.size(1)
    val nFrame = input.size(3) * input.size(4)
    if(saveMean.size(1) != nChannels) {
      saveMean.resize(nChannels)
      saveStd.resize(nChannels)
      runningMean.resize(nChannels)
      runningVar.resize(nChannels)
    }
    val meanData = saveMean.storage().array()
    val meanOffset = saveMean.storageOffset() - 1
    var i = 0
    var b = 0
    while(b < nBatch) {
      var c = 0
      while (c < nChannels) {
        var k = 0
        var meanSum = 0f
        while(k < nFrame) {
          meanSum += inputData(i + inputOffset)
          k += 1
          i += 1
        }
        meanData(c + meanOffset) += meanSum
        c += 1
      }
      b += 1
    }

    var meanLen = 1
    if (needSync) {
      ParameterSynchronizer.syncData(meanKey, saveMean)
      val meanEventData = ParameterSynchronizer.collect[Float](meanKey)
      meanLen = meanEventData.size
      var c = 0
      while (c < nChannels) {
        globalMean(c) = 0.0f
        meanEventData.keySet.asScala.foreach(threadId => {
          val localMean = meanEventData.get(threadId)
          val localOffset = localMean.storageOffset() - 1
          globalMean(c) += localMean.storage.array()(c + localOffset)
        })
        c += 1
      }

      ParameterSynchronizer.reset[Float](meanKey)
      System.arraycopy(globalMean, 0, meanData, meanOffset, nChannels)
    }

    val n = input.nElement()
    val frameSize = n / nChannels
    var c = 0
    val runningMeanData = runningMean.storage().array()
    val runningMeanOffset = runningMean.storageOffset() - 1
    while(c < nChannels) {
      meanData(c + meanOffset) /= (frameSize * meanLen)
      runningMeanData(c + runningMeanOffset) = meanData(c + meanOffset) * momentum +
        (1 - momentum) * runningMeanData(c + runningMeanOffset)
      c += 1
    }

    val stdData = saveStd.storage().array()
    val stdOffset = saveStd.storageOffset() - 1
    i = 0
    b = 0
    while(b < nBatch) {
      var c = 0
      while(c < nChannels) {
        var k = 0
        var stdSum = 0f
        while(k < nFrame) {
          val diff = (inputData(i + inputOffset) - meanData(c + meanOffset))
          stdSum += diff * diff
          k += 1
          i += 1
        }
        stdData(c + stdOffset) += stdSum
        c += 1
      }
      b += 1
    }

    var stdLen = 1

    if (needSync) {
      ParameterSynchronizer.syncData(stdKey, saveStd)

      val stdEventData = ParameterSynchronizer.collect[Float](stdKey)

      c = 0
      while (c < nChannels) {
        globalStd(c) = 0.0f
        stdEventData.keySet.asScala.foreach(threadId => {
          val localStd = stdEventData.get(threadId)
          val localStdOffSet = localStd.storageOffset() - 1
          globalStd(c) += localStd.storage.array()(c + localStdOffSet)
        })
        c += 1
      }

      stdLen = stdEventData.size

      ParameterSynchronizer.reset[Float](stdKey)

      System.arraycopy(globalStd, 0, stdData, stdOffset, nChannels)
    }

    c = 0
    val runningVarData = runningVar.storage().array()
    val runningVarOffset = runningVar.storageOffset() - 1
    while(c < nChannels) {
      if (stdData(c + stdOffset) == 0 && eps == 0) {
        stdData(c + stdOffset) = 0
        if (saveVar != null) {
          saveVar.setValue(c + 1, 0f)
        }
        if (batchVar != null) {
          batchVar.setValue(c + 1, 0f)
        }
      } else {
        val s = stdData(c + stdOffset)
        val unbiasedVar = s / (frameSize * stdLen - 1)
        if (saveVar != null) {
          saveVar.setValue(c + 1, s / (frameSize * stdLen))
        }
        if (batchVar != null) {
          batchVar.setValue(c + 1, unbiasedVar)
        }
        stdData(c + stdOffset) = 1.0f / Math.sqrt(s / (frameSize * stdLen) + eps).toFloat
        runningVarData(c + runningVarOffset) = momentum * unbiasedVar +
          (1 - momentum) * runningVarData(c + runningVarOffset)
      }
      c += 1
    }

    if (needFix) {
      c = 0
      while(c < nChannels) {
        meanData(c + meanOffset) = 0
        stdData(c + stdOffset) = 0.0001f
        c += 1
      }
    }

    if (scale != null) {
      val scaleData = scale.storage().array()
      val scaleOffset = scale.storageOffset() - 1
      val offsetData = offset.storage().array()
      val offsetOffset = offset.storageOffset() - 1
      i = 0
      b = 0
      while (b < nBatch) {
        var c = 0
        while (c < nChannels) {
          var k = 0
          while (k < nFrame) {
            outputData(i + outputOffset) = (inputData(i + inputOffset) -
              meanData(c + meanOffset)) * stdData(c + stdOffset) *
              scaleData(c + scaleOffset) + offsetData(c + offsetOffset)
            k += 1
            i += 1
          }
          c += 1
        }
        b += 1
      }
    } else {
      i = 0
      b = 0
      while (b < nBatch) {
        var c = 0
        while (c < nChannels) {
          var k = 0
          while (k < nFrame) {
            outputData(i + outputOffset) = (inputData(i + inputOffset) -
              meanData(c + meanOffset)) * stdData(c + stdOffset)
            k += 1
            i += 1
          }
          c += 1
        }
        b += 1
      }
    }
  }

  private[bigdl] def updateOutputNCHWTrainDouble(input: Tensor[Double], output: Tensor[Double],
    saveMean: Tensor[Double], saveStd: Tensor[Double], runningMean: Tensor[Double],
    runningVar: Tensor[Double], scale: Tensor[Double], offset: Tensor[Double],
    eps: Double, momentum: Double,
    batchVar: Tensor[Double] = null, saveVar: Tensor[Double] = null, needFix: Boolean = false,
                                                 globalMean: Array[Double] = null,
                                                 globalStd: Array[Double] = null,
                                                 meanKey: String = null,
                                                 stdKey: String = null,
                                                 needSync: Boolean = false)
  : Unit = {
    require(input.isContiguous(), "BatchNorm NCHW require a contiguous input")
    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1
    val outputData = output.storage().array()
    val outputOffset = output.storageOffset() - 1
    val nChannels = input.size(2)
    val nBatch = input.size(1)
    val nFrame = input.size(3) * input.size(4)
    if(saveMean.size(1) != nChannels) {
      saveMean.resize(nChannels)
      saveStd.resize(nChannels)
      runningMean.resize(nChannels)
      runningVar.resize(nChannels)
    }
    val meanData = saveMean.storage().array()
    val meanOffset = saveMean.storageOffset() - 1
    var i = 0
    var b = 0
    while(b < nBatch) {
      var c = 0
      while (c < nChannels) {
        var k = 0
        var meanSum = 0d
        while(k < nFrame) {
          meanSum += inputData(i + inputOffset)
          k += 1
          i += 1
        }
        meanData(c + meanOffset) += meanSum
        c += 1
      }
      b += 1
    }
    var meanLen = 1
    if (needSync) {
      ParameterSynchronizer.syncData(meanKey, saveMean)

      val meanEventData = ParameterSynchronizer.collect[Double](meanKey)

      meanLen = meanEventData.size

      var c = 0
      while (c < nChannels) {
        globalMean(c) = 0.0
        meanEventData.keySet.asScala.foreach(threadId => {
          val localMean = meanEventData.get(threadId)
          val localOffset = localMean.storageOffset() - 1
          globalMean(c) += localMean.storage.array()(c + localOffset)
        })
        c += 1
      }

      ParameterSynchronizer.reset[Double](meanKey)

      System.arraycopy(globalMean, 0, meanData, meanOffset, nChannels)
    }

    val n = input.nElement()
    val frameSize = n / nChannels
    var c = 0
    val runningMeanData = runningMean.storage().array()
    val runningMeanOffset = runningMean.storageOffset() - 1
    while(c < nChannels) {
      meanData(c + meanOffset) /= (frameSize * meanLen)
      runningMeanData(c + runningMeanOffset) = meanData(c + meanOffset) * momentum +
        (1 - momentum) * runningMeanData(c + runningMeanOffset)
      c += 1
    }

    val stdData = saveStd.storage().array()
    val stdOffset = saveStd.storageOffset() - 1
    i = 0
    b = 0
    while(b < nBatch) {
      var c = 0
      while(c < nChannels) {
        var k = 0
        while(k < nFrame) {
          val diff = (inputData(i + inputOffset) - meanData(c + meanOffset))
          stdData(c + stdOffset) += diff * diff
          k += 1
          i += 1
        }
        c += 1
      }
      b += 1
    }
    var stdLen = 1
    if (needSync) {
      ParameterSynchronizer.syncData(stdKey, saveStd)
      val stdEventData = ParameterSynchronizer.collect[Double](stdKey)
      c = 0
      while (c < nChannels) {
        globalStd(c) = 0.0
        stdEventData.keySet.asScala.foreach(threadId => {
          val localStd = stdEventData.get(threadId)
          val localStdOffSet = localStd.storageOffset() - 1
          globalStd(c) += localStd.storage.array()(c + localStdOffSet)
        })
        c += 1
      }
      stdLen = stdEventData.size
      ParameterSynchronizer.reset[Double](stdKey)
      System.arraycopy(globalStd, 0, stdData, stdOffset, nChannels)
    }

    c = 0
    val runningVarData = runningVar.storage().array()
    val runningVarOffset = runningVar.storageOffset() - 1
    while(c < nChannels) {
      if (stdData(c + stdOffset) == 0 && eps == 0) {
        stdData(c + stdOffset) = 0
        if (saveVar != null) {
          saveVar.setValue(c + 1, 0f)
        }
        if (batchVar != null) {
          batchVar.setValue(c + 1, 0f)
        }
      } else {
        val s = stdData(c + stdOffset)
        val unbiasedVar = s / (frameSize * stdLen - 1)
        if (saveVar != null) {
          saveVar.setValue(c + 1, s / (frameSize * stdLen))
        }
        if (batchVar != null) {
          batchVar.setValue(c + 1, unbiasedVar)
        }
        stdData(c + stdOffset) = 1.0 / Math.sqrt(s / (frameSize * stdLen) + eps)
        runningVarData(c + stdOffset) = momentum * unbiasedVar + (1 - momentum) *
          runningVarData(c + runningVarOffset)
      }
      c += 1
    }

    if (needFix) {
      c = 0
      while(c < nChannels) {
        meanData(c + meanOffset) = 0
        stdData(c + stdOffset) = 0.0001
        c += 1
      }
    }

    if (scale != null) {
      val scaleData = scale.storage().array()
      val scaleOffset = scale.storageOffset() - 1
      val offsetData = offset.storage().array()
      val offsetOffset = offset.storageOffset() - 1
      i = 0
      b = 0
      while (b < nBatch) {
        var c = 0
        while (c < nChannels) {
          var k = 0
          while (k < nFrame) {
            outputData(i + outputOffset) = (inputData(i + inputOffset) -
              meanData(c + meanOffset)) * stdData(c + stdOffset) *
              scaleData(c + scaleOffset) + offsetData(c + offsetOffset)
            k += 1
            i += 1
          }
          c += 1
        }
        b += 1
      }
    } else {
      i = 0
      b = 0
      while (b < nBatch) {
        var c = 0
        while (c < nChannels) {
          var k = 0
          while (k < nFrame) {
            outputData(i + outputOffset) = (inputData(i + inputOffset) -
              meanData(c + meanOffset)) * stdData(c + stdOffset)
            k += 1
            i += 1
          }
          c += 1
        }
        b += 1
      }
    }
  }

  private[bigdl] def updateGradInputNCHWTrainDouble(
    input: Tensor[Double],
    gradOutput: Tensor[Double],
    gradInput: Tensor[Double],
    scale: Tensor[Double],
    saveMean: Tensor[Double],
    saveStd: Tensor[Double],
    gMean: Tensor[Double],
    gxMean: Tensor[Double],
    globalGmean: Array[Double],
    globalGxmean: Array[Double],
    gMeanKey: String = null,
    gxMeanKey: String = null,
    needSync: Boolean = false
  ): Unit = {
    require(input.nDimension() == 4, "BN require a 4D input")
    require(input.isContiguous(), "input is not contiguous")
    require(gradOutput.nDimension() == 4, "BN require a 4D gradient")
    require(gradOutput.isContiguous(), "gradient is not contiguous")
    val nChannel = gradOutput.size(2)
    require(saveMean.size(1) == nChannel, "saveMean length is not consistent with channel number")
    require(saveStd.size(1) == nChannel, "saveStd length is not consistent with channel number")

    gradInput.resizeAs(gradOutput)
    if (gMean.isEmpty) {
      gMean.resize(saveMean.size(1))
      gxMean.resize(saveMean.size(1))
    }

    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1
    val gradOutputData = gradOutput.storage().array()
    val gradOutputOffset = gradOutput.storageOffset() - 1
    val gradInputData = gradInput.storage().array()
    val gradInputOffset = gradInput.storageOffset() - 1
    val saveMeanData = saveMean.storage().array()
    val saveMeanOffset = saveMean.storageOffset() - 1
    val saveStdData = saveStd.storage().array()
    val saveStdOffset = saveStd.storageOffset() - 1
    val gMeanData = gMean.storage().array()
    val gxMeanData = gxMean.storage().array()

    val nBatch = gradOutput.size(1)
    val frameSize = gradOutput.size(3) * gradOutput.size(4)
    val n = gradOutput.nElement()
    var b = 0
    var i = 0
    while(b < nBatch) {
      var c = 0
      while(c < nChannel) {
        var k = 0
        while(k < frameSize) {
          gMeanData(c) += gradOutputData(i + gradOutputOffset)
          gxMeanData(c) += gradOutputData(i + gradOutputOffset) *
            (inputData(i + inputOffset) - saveMeanData(c + saveMeanOffset))
          k += 1
          i += 1
        }
        c += 1
      }
      b += 1
    }
    var gmeanEventLen = 1
    var gmxmeanEventLen = 1
    if (needSync) {
      ParameterSynchronizer.syncData(gMeanKey, gMean)
      val gMeanEventData = ParameterSynchronizer.collect[Double](gMeanKey)
      var c = 0
      while (c < nChannel) {
        globalGmean(c) = 0.0
        gMeanEventData.keySet.asScala.foreach(threadId => {
          val localGmean = gMeanEventData.get(threadId)
          val localGmeanOffset = localGmean.storageOffset() - 1
          globalGmean(c) += localGmean.storage.array()(c + localGmeanOffset)
        })
        c += 1
      }

      gmeanEventLen = gMeanEventData.size

      ParameterSynchronizer.reset[Double](gMeanKey)

      ParameterSynchronizer.syncData(gxMeanKey, gxMean)

      val gxMeanEventData = ParameterSynchronizer.collect[Double](gxMeanKey)

      c = 0
      while (c < nChannel) {
        globalGxmean(c) = 0.0
        gxMeanEventData.keySet.asScala.foreach(threadId => {
          val localGxmean = gxMeanEventData.get(threadId)
          val localGxmeanOffset = localGxmean.storageOffset() - 1
          globalGxmean(c) += localGxmean.storage.array()(c + localGxmeanOffset)
        })
        c += 1
      }

      gmxmeanEventLen = gxMeanEventData.size

      ParameterSynchronizer.reset[Double](gxMeanKey)
    }

    var c = 0
    val size = n / nChannel
    while(c < nChannel) {
      if (needSync) {
        gMeanData(c) = globalGmean(c) / (size * gmeanEventLen)
        val invStd = saveStdData(saveStdOffset + c)
        gxMeanData(c) = globalGxmean(c) * invStd * invStd / (size * gmxmeanEventLen)
      } else {
        gMeanData(c) = gMeanData(c) / (size * gmeanEventLen)
        val invStd = saveStdData(saveStdOffset + c)
        gxMeanData(c) = gxMeanData(c) * invStd * invStd / (size * gmxmeanEventLen)
      }
      c += 1
    }

    i = 0
    b = 0
    if (scale != null) {
      val scaleData = scale.storage().array()
      val scaleOffset = scale.storageOffset() - 1
      require(scale.size(1) == nChannel, "scale length is not consistent with channel number")
      while (b < nBatch) {
        var c = 0
        while (c < nChannel) {
          var k = 0
          while (k < frameSize) {
            val invStd = saveStdData(saveStdOffset + c)
            gradInputData(gradInputOffset + i) = (gradOutputData(gradOutputOffset + i) -
              gMeanData(c) - (inputData(inputOffset + i) - saveMeanData(saveMeanOffset + c)) *
              gxMeanData(c)) * invStd * scaleData(scaleOffset + c)
            k += 1
            i += 1
          }
          c += 1
        }
        b += 1
      }
    } else {
      while (b < nBatch) {
        var c = 0
        while (c < nChannel) {
          var k = 0
          while (k < frameSize) {
            val invStd = saveStdData(saveStdOffset + c)
            gradInputData(gradInputOffset + i) = (gradOutputData(gradOutputOffset + i) -
              gMeanData(c) - (inputData(inputOffset + i) - saveMeanData(saveMeanOffset + c)) *
              gxMeanData(c)) * invStd
            k += 1
            i += 1
          }
          c += 1
        }
        b += 1
      }
    }
  }

  private[bigdl] def updateGradInputNCHWInferFloat(
    gradOutput: Tensor[Float],
    gradInput: Tensor[Float],
    scale: Tensor[Float],
    saveStd: Tensor[Float]
  ): Unit = {
    require(gradOutput.nDimension() == 4, "BN require a 4D gradient")
    require(gradOutput.isContiguous(), "gradient is not contiguous")
    val nChannel = gradOutput.size(2)
    require(saveStd.size(1) == nChannel, "saveStd length is not consistent with channel number")

    gradInput.resizeAs(gradOutput)
    val gradOutputData = gradOutput.storage().array()
    val gradOutputOffset = gradOutput.storageOffset() - 1
    val gradInputData = gradInput.storage().array()
    val gradInputOffset = gradInput.storageOffset() - 1
    val saveStdData = saveStd.storage().array()
    val saveStdOffset = saveStd.storageOffset() - 1

    val nBatch = gradOutput.size(1)
    val frameSize = gradOutput.size(3) * gradOutput.size(4)
    var b = 0
    var i = 0
    if (scale != null) {
      require(scale.size(1) == nChannel, "scale length is not consistent with channel number")
      val scaleData = scale.storage().array()
      val scaleOffset = scale.storageOffset() - 1
      while (b < nBatch) {
        var c = 0
        while (c < nChannel) {
          var k = 0
          while (k < frameSize) {
            val invStd = saveStdData(saveStdOffset + c)
            gradInputData(gradInputOffset + i) = scaleData(scaleOffset + c) *
              invStd * gradOutputData(gradOutputOffset + i)
            k += 1
            i += 1
          }
          c += 1
        }
        b += 1
      }
    } else {
      while (b < nBatch) {
        var c = 0
        while (c < nChannel) {
          var k = 0
          while (k < frameSize) {
            val invStd = saveStdData(saveStdOffset + c)
            gradInputData(gradInputOffset + i) =
              invStd * gradOutputData(gradOutputOffset + i)
            k += 1
            i += 1
          }
          c += 1
        }
        b += 1
      }
    }
  }

  private[bigdl] def updateGradInputNCHWInferDouble(
    gradOutput: Tensor[Double],
    gradInput: Tensor[Double],
    scale: Tensor[Double],
    saveStd: Tensor[Double]
  ): Unit = {
    require(gradOutput.nDimension() == 4, "BN require a 4D gradient")
    require(gradOutput.isContiguous(), "gradient is not contiguous")
    val nChannel = gradOutput.size(2)
    require(saveStd.size(1) == nChannel, "saveStd length is not consistent with channel number")

    gradInput.resizeAs(gradOutput)
    val gradOutputData = gradOutput.storage().array()
    val gradOutputOffset = gradOutput.storageOffset() - 1
    val gradInputData = gradInput.storage().array()
    val gradInputOffset = gradInput.storageOffset() - 1
    val saveStdData = saveStd.storage().array()
    val saveStdOffset = saveStd.storageOffset() - 1

    val nBatch = gradOutput.size(1)
    val frameSize = gradOutput.size(3) * gradOutput.size(4)
    var b = 0
    var i = 0
    if (scale != null) {
      require(scale.size(1) == nChannel, "scale length is not consistent with channel number")
      val scaleData = scale.storage().array()
      val scaleOffset = scale.storageOffset() - 1
      while (b < nBatch) {
        var c = 0
        while (c < nChannel) {
          var k = 0
          while (k < frameSize) {
            val invStd = saveStdData(saveStdOffset + c)
            gradInputData(gradInputOffset + i) = scaleData(scaleOffset + c) *
              invStd * gradOutputData(gradOutputOffset + i)
            k += 1
            i += 1
          }
          c += 1
        }
        b += 1
      }
    } else {
      while (b < nBatch) {
        var c = 0
        while (c < nChannel) {
          var k = 0
          while (k < frameSize) {
            val invStd = saveStdData(saveStdOffset + c)
            gradInputData(gradInputOffset + i) =
              invStd * gradOutputData(gradOutputOffset + i)
            k += 1
            i += 1
          }
          c += 1
        }
        b += 1
      }
    }
  }

  private[bigdl] def accGradientNHWCFloat(gradOutput: Tensor[Float],
    gradWeight: Tensor[Float], gradBias: Tensor[Float],
    input: Tensor[Float], saveMean: Tensor[Float],
    saveStd: Tensor[Float], scaleW: Float, scaleB: Float): Unit = {
    require(gradOutput.isContiguous(), "gradOutput must be contiguous")
    require(gradWeight.isContiguous(), "gradWeight must be contiguous")
    require(gradBias.isContiguous(), "gradBias must be contiguous")
    require(input.isContiguous(), "gradWeight must be contiguous")
    require(saveMean.nDimension() == 1, "saveMean must be 1D")
    require(saveStd.nDimension() == 1, "saveStd must be 1D")
    val nChannel = saveMean.size(1)
    val gradOutputData = gradOutput.storage().array()
    val gradOutputOffset = gradOutput.storageOffset() - 1
    val gradWeightData = gradWeight.storage().array()
    val gradWeightOffset = gradWeight.storageOffset() - 1
    val gradBiasData = gradBias.storage().array()
    val gradBiasOffset = gradBias.storageOffset() - 1
    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1
    val saveMeanData = saveMean.storage().array()
    val saveMeanOffset = saveMean.storageOffset() - 1
    val saveStdData = saveStd.storage().array()
    val saveStdOffset = saveStd.storageOffset() - 1

    var i = 0
    val n = input.nElement()
    while(i < n) {
      var c = 0
      while(c < nChannel) {
        val g = gradOutputData(gradOutputOffset + i)
        gradWeightData(c + gradWeightOffset) += g *
          (inputData(inputOffset + i) - saveMeanData(saveMeanOffset + c)) *
          saveStdData(saveStdOffset + c) * scaleW
        gradBiasData(c + gradBiasOffset) += g * scaleB
        i += 1
        c += 1
      }
    }
  }

  private[bigdl] def accGradientNHWCDouble(gradOutput: Tensor[Double],
    gradWeight: Tensor[Double], gradBias: Tensor[Double],
    input: Tensor[Double], saveMean: Tensor[Double],
    saveStd: Tensor[Double], scaleW: Double, scaleB: Double): Unit = {
    require(gradOutput.isContiguous(), "gradOutput must be contiguous")
    require(gradWeight.isContiguous(), "gradWeight must be contiguous")
    require(gradBias.isContiguous(), "gradBias must be contiguous")
    require(input.isContiguous(), "gradWeight must be contiguous")
    require(saveMean.nDimension() == 1, "saveMean must be 1D")
    require(saveStd.nDimension() == 1, "saveStd must be 1D")
    val nChannel = saveMean.size(1)
    val gradOutputData = gradOutput.storage().array()
    val gradOutputOffset = gradOutput.storageOffset() - 1
    val gradWeightData = gradWeight.storage().array()
    val gradWeightOffset = gradWeight.storageOffset() - 1
    val gradBiasData = gradBias.storage().array()
    val gradBiasOffset = gradBias.storageOffset() - 1
    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1
    val saveMeanData = saveMean.storage().array()
    val saveMeanOffset = saveMean.storageOffset() - 1
    val saveStdData = saveStd.storage().array()
    val saveStdOffset = saveStd.storageOffset() - 1

    var i = 0
    val n = input.nElement()
    while(i < n) {
      var c = 0
      while(c < nChannel) {
        val g = gradOutputData(gradOutputOffset + i)
        gradWeightData(c + gradWeightOffset) += g *
          (inputData(inputOffset + i) - saveMeanData(saveMeanOffset + c)) *
          saveStdData(saveStdOffset + c) * scaleW
        gradBiasData(c + gradBiasOffset) += g * scaleB
        i += 1
        c += 1
      }
    }
  }

  private[bigdl] def accGradientNCHWFloat(gradOutput: Tensor[Float],
    gradWeight: Tensor[Float], gradBias: Tensor[Float],
    input: Tensor[Float], saveMean: Tensor[Float],
    saveStd: Tensor[Float], scaleW: Float, scaleB: Float): Unit = {
    require(gradOutput.isContiguous(), "gradOutput must be contiguous")
    require(gradWeight.isContiguous(), "gradWeight must be contiguous")
    require(gradBias.isContiguous(), "gradBias must be contiguous")
    require(input.isContiguous(), "gradWeight must be contiguous")
    require(saveMean.nDimension() == 1, "saveMean must be 1D")
    require(saveStd.nDimension() == 1, "saveStd must be 1D")
    val nChannel = saveMean.size(1)
    val gradOutputData = gradOutput.storage().array()
    val gradOutputOffset = gradOutput.storageOffset() - 1
    val gradWeightData = gradWeight.storage().array()
    val gradWeightOffset = gradWeight.storageOffset() - 1
    val gradBiasData = gradBias.storage().array()
    val gradBiasOffset = gradBias.storageOffset() - 1
    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1
    val saveMeanData = saveMean.storage().array()
    val saveMeanOffset = saveMean.storageOffset() - 1
    val saveStdData = saveStd.storage().array()
    val saveStdOffset = saveStd.storageOffset() - 1

    val nBatch = gradOutput.size(1)
    val frameSize = gradOutput.size(3) * gradOutput.size(4)
    var i = 0
    var b = 0
    while(b < nBatch) {
      var c = 0
      while (c < nChannel) {
        var k = 0
        while(k < frameSize) {
          val g = gradOutputData(gradOutputOffset + i)
          gradWeightData(c + gradWeightOffset) += g *
            (inputData(inputOffset + i) - saveMeanData(saveMeanOffset + c)) *
            saveStdData(saveStdOffset + c) * scaleW
          gradBiasData(c + gradBiasOffset) += g * scaleB
          k += 1
          i += 1
        }
        c += 1
      }
      b += 1
    }
  }

  private[bigdl] def accGradientNCHWDouble(gradOutput: Tensor[Double],
    gradWeight: Tensor[Double], gradBias: Tensor[Double],
    input: Tensor[Double], saveMean: Tensor[Double],
    saveStd: Tensor[Double], scaleW: Double, scaleB: Double): Unit = {
    require(gradOutput.isContiguous(), "gradOutput must be contiguous")
    require(gradWeight.isContiguous(), "gradWeight must be contiguous")
    require(gradBias.isContiguous(), "gradBias must be contiguous")
    require(input.isContiguous(), "gradWeight must be contiguous")
    require(saveMean.nDimension() == 1, "saveMean must be 1D")
    require(saveStd.nDimension() == 1, "saveStd must be 1D")
    val nChannel = saveMean.size(1)
    val gradOutputData = gradOutput.storage().array()
    val gradOutputOffset = gradOutput.storageOffset() - 1
    val gradWeightData = gradWeight.storage().array()
    val gradWeightOffset = gradWeight.storageOffset() - 1
    val gradBiasData = gradBias.storage().array()
    val gradBiasOffset = gradBias.storageOffset() - 1
    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1
    val saveMeanData = saveMean.storage().array()
    val saveMeanOffset = saveMean.storageOffset() - 1
    val saveStdData = saveStd.storage().array()
    val saveStdOffset = saveStd.storageOffset() - 1

    val nBatch = gradOutput.size(1)
    val frameSize = gradOutput.size(3) * gradOutput.size(4)
    var i = 0
    var b = 0
    while(b < nBatch) {
      var c = 0
      while (c < nChannel) {
        var k = 0
        while(k < frameSize) {
          val g = gradOutputData(gradOutputOffset + i)
          gradWeightData(c + gradWeightOffset) += scaleW * (inputData(inputOffset + i) -
            saveMeanData(saveMeanOffset + c)) * g * saveStdData(saveStdOffset + c)
          gradBiasData(c + gradBiasOffset) += g * scaleB
          k += 1
          i += 1
        }
        c += 1
      }
      b += 1
    }
  }
}
