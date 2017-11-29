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

import scala.reflect.ClassTag
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.mkl.MKL
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, Initializable, TensorModule}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, Tensor}
import com.intel.analytics.bigdl.utils.{Engine, T, Table}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.serializer._
import serialization.Bigdl.{AttrValue, BigDLModule}

import scala.concurrent.Future
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
class SpatialBatchNormalizationWithRelu[T: ClassTag](
        nOutput: Int, // output feature map number
        eps: Double = 1e-5, // avoid divde zero
        momentum: Double = 0.1, // momentum for weight update
        affine: Boolean = true, // affine operation on output or not
        initWeight: Tensor[T] = null,
        initBias: Tensor[T] = null,
        initGradWeight: Tensor[T] = null,
        initGradBias: Tensor[T] = null
      )(implicit ev: TensorNumeric[T])
  extends SpatialBatchNormalization[T](nOutput, eps, momentum, affine,
    initWeight, initBias, initGradWeight, initGradBias)
    with Initializable {

  require(nOutput > 0)

  override val nDim = 4

  @transient
  private var results : Array[Future[_]] = null
  @transient
  // BatchNormalization has internal parameters (saveMean, saveStd)
  // that changes at every forward, so a standard gradcheck won't work with this module.
  // if you want to do a gradcheck, you will need to fix those variables, otherwise not fix.
  private var needFix: Boolean = false

  {
    val wInit = RandomUniform(0, 1)
    val bInit = Zeros
    setInitMethod(wInit, bInit)
  }

  override def reset(): Unit = {
    if (null != weight && initWeight == null) {
      weightInitMethod.init(weight, VariableFormat.ONE_D)
    }

    if (null != bias && initBias == null) {
      biasInitMethod.init(bias, VariableFormat.ONE_D)
    }

    zeroGradParameters()
  }

  @inline
  private def checkInputDim(input: Tensor[T]): Unit = {
    require(input.dim() == nDim || (input.dim() == nDim - 1 && train == false),
      s"only mini-batch supported (${nDim}D tensor), got ${input.dim()}D tensor instead")
  }

  @inline
  private def makeBatch(input: Tensor[T]): Tensor[T] = {
    if (input.dim() == nDim - 1 && train == false) {
      input.addSingletonDimension()
    } else {
      input
    }
  }

  @inline
  private def initializeBuffer(nOutput: Int): Unit = {
    runningMean.resize(nOutput).zero
    runningVar.resize(nOutput).fill(ev.fromType[Int](1))
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    checkInputDim(input)

    output.resizeAs(input)

    val _input = makeBatch(input)
    val nInput = _input.size(2)

    if (runningMean.nElement == 0 || runningMean.nElement < nInput) {
      initializeBuffer(nInput)
    }

    saveMean.resizeAs(runningMean).zero
    saveStd.resizeAs(runningVar).fill(ev.fromType[Int](1))

    if (results == null || results.length > nInput) {
      results = new Array[Future[_]](nInput)
    }
    val n = _input.nElement() / nInput
    val spatialBatchSize = if (nDim == 2) 1 else _input.size(1)
    val frameSize = if (nDim == 2) n else n / spatialBatchSize
    if (ones.nElement() != frameSize) {
      ones.resize(frameSize)
      ones.fill(ev.one)
    }
    ev.getType() match {
      case DoubleType =>
        val inputDouble = _input.asInstanceOf[Tensor[Double]]
        val outputDouble = output.asInstanceOf[Tensor[Double]]
        SpatialBatchNormalizationWithRelu.updateOutputDouble(
          n, nInput, momentum, eps, train,
          spatialBatchSize, nDim, needFix, inputDouble,
          outputDouble, ones.toTensor[Double], saveMean.toTensor[Double], saveStd.toTensor[Double],
          runningMean.toTensor[Double], runningVar.toTensor[Double],
          weight.toTensor[Double], bias.toTensor[Double])

      case FloatType =>
        val inputFloat = _input.asInstanceOf[Tensor[Float]]
        val outputFloat = output.asInstanceOf[Tensor[Float]]
        SpatialBatchNormalizationWithRelu.updateOutputFloat(
          n, nInput, momentum.toFloat, eps.toFloat, train,
          spatialBatchSize, nDim, needFix, inputFloat,
          outputFloat, ones.toTensor[Float], saveMean.toTensor[Float], saveStd.toTensor[Float],
          runningMean.toTensor[Float], runningVar.toTensor[Float],
          weight.toTensor[Float], bias.toTensor[Float])
    }

    output
  }

  private def updateOutputDouble(input: Array[Double], inputOffset: Int, inputStride: Int,
                                 output: Array[Double], outputOffset: Int, outputStride: Int,
                                 nInput: Int, n: Int, stride2: Int
                                ): Unit = {
    var f = 0
    while (f < nInput) {
      val _f = f + 1
      results(f) = Engine.model.invoke(() => {
        var mean = 0.0
        var invstd = 0.0
        if (train) {
          var sum = 0.0
          var i = 0
          while (i < n) {
            sum += input(i % stride2 + (_f - 1) * stride2 + inputOffset +
              (i / stride2) * inputStride)
            i += 1
          }
          mean = sum / n
          saveMean.setValue(_f, ev.fromType[Double](mean))
          sum = 0.0
          i = 0
          while (i < n) {
            sum += (input(i % stride2 + (_f - 1) * stride2 + inputOffset +
              (i / stride2) * inputStride) - mean) * (input(i % stride2 + (_f - 1) * stride2 +
              inputOffset + (i / stride2) * inputStride) - mean)
            i += 1
          }

          invstd = if (sum == 0 && eps == 0.0) {
            0.0
          } else {
            1 / Math.sqrt(sum / n + eps)
          }
          saveStd.setValue(_f, ev.fromType[Double](invstd))

          runningMean.setValue(_f, ev.fromType[Double](momentum * mean + (1 - momentum) *
            ev.toType[Double](runningMean.valueAt(_f))))

          val unbiasedVar = sum / (n - 1)
          runningVar.setValue(_f, ev.fromType[Double](momentum * unbiasedVar + (1 - momentum) *
            ev.toType[Double](runningVar.storage().array()(_f - 1))))
        } else {
          mean = ev.toType[Double](runningMean.valueAt(_f))
          invstd = 1 / Math.sqrt(ev.toType[Double](runningVar.valueAt(_f)) + eps)
        }

        if (needFix) {
          mean = 0
          invstd = 0.0001
          saveMean.zero().fill(ev.fromType(mean))
          saveStd.zero().fill(ev.fromType(invstd))
        }
        val w = if (null != weight) ev.toType[Double](weight.valueAt(_f)) else 1.0
        val b = if (null != bias) ev.toType[Double](bias.valueAt(_f)) else 0.0

        var i = 0
        while (i < n) {
          output(i % stride2 + (_f - 1) * stride2 +
            inputOffset + (i / stride2) * inputStride) = (input(i % stride2 + (_f - 1) * stride2 +
            inputOffset + (i / stride2) * inputStride) - mean) * invstd * w + b
          i += 1
        }
      })
      f += 1
    }
    Engine.model.sync(results)
  }

  var ones = Tensor()


  private def updateOutputFloat(input: Array[Float], inputOffset: Int, inputStride: Int,
                                output: Array[Float], outputOffset: Int, outputStride: Int,
                                nInput: Int, n: Int, stride2: Int
                               ): Unit = {
    var f = 0
    while (f < nInput) {
      val _f = f + 1
      results(f) = Engine.model.invoke(() => {
        var mean = 0.0f
        var invstd = 0.0f
        if (train) {
          var sum = 0.0f
          var i = 0
          while (i < n) {
            sum += input(i % stride2 + (_f - 1) * stride2 + inputOffset +
              (i / stride2) * inputStride)
            i += 1
          }
          mean = sum / n
          saveMean.setValue(_f, ev.fromType(mean))

          sum = 0.0f
          i = 0
          while (i < n) {
            sum += (input(i % stride2 + (_f - 1) * stride2 + inputOffset +
              (i / stride2) * inputStride) - mean) * (input(i % stride2 + (_f - 1) * stride2 +
              inputOffset + (i / stride2) * inputStride) - mean)
            i += 1
          }

          invstd = if (sum == 0 && eps == 0.0) {
            0.0f
          } else {
            1.0f / Math.sqrt(sum / n + eps).toFloat
          }
          saveStd.setValue(_f, ev.fromType(invstd))

          runningMean.setValue(_f, ev.fromType(momentum * mean + (1 - momentum) *
            ev.toType[Double](runningMean.valueAt(_f))))

          val unbiasedVar = sum / (n - 1)
          runningVar.setValue(_f, ev.fromType[Double](momentum * unbiasedVar + (1 - momentum) *
            ev.toType[Double](runningVar.storage().array()(_f - 1))))
        } else {
          mean = ev.toType[Float](runningMean.valueAt(_f))
          invstd = 1 / Math.sqrt(ev.toType[Double](runningVar.valueAt(_f)) + eps).toFloat
        }

        if (needFix) {
          mean = 0
          invstd = 0.0001f
          saveMean.zero().fill(ev.fromType(mean))
          saveStd.zero().fill(ev.fromType(invstd))
        }

        val w = if (null != weight) ev.toType[Float](weight.valueAt(_f)) else 1.0f
        val b = if (null != bias) ev.toType[Float](bias.valueAt(_f)) else 0.0f

        var i = 0
        while (i < n) {
          val out_index = i % stride2 + (_f - 1) * stride2 +
            inputOffset + (i / stride2) * inputStride

          val out_value = (input(out_index) - mean) * invstd * w + b
          if (out_value < 0) {
            output(out_index) = 0
          } else {
            output(out_index) = out_value
          }
          i += 1
        }
      })
      f += 1
    }
    Engine.model.sync(results)
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    backward(input, gradOutput, gradInput, null, null)
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    backward(input, gradOutput, null, gradWeight, gradBias)
  }

  override def backward(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    checkInputDim(input)
    checkInputDim(gradOutput)
    val before = System.nanoTime()
    val result = backward(input, gradOutput, gradInput, gradWeight, gradBias)
    backwardTime += System.nanoTime() - before
    result
  }

  override def backward(input: Tensor[T], gradOutput: Tensor[T],
               theGradInput: Tensor[T] = null, theGradWeight: Tensor[T] = null,
               theGradBias: Tensor[T] = null): Tensor[T] = {
    require(train, "should be in training mode when this.train is true")
    require(null != saveMean && null != saveStd, "must call updateOutput() first")

    if (null != theGradInput) {
      theGradInput.resizeAs(gradOutput)
    }

    val nInput = input.size(2)
    if (results == null || results.length > nInput) {
      results = new Array[Future[_]](nInput)
    }
    val n = input.nElement() / nInput

    ev.getType() match {
      case DoubleType =>
        val inputDouble = input.asInstanceOf[Tensor[Double]]
        val inputData = inputDouble.storage().array()
        val inputOffset = inputDouble.storageOffset() - 1
        val inputStride = input.stride(1)
        val inputStride2 = input.stride(2)
        val gradOutputDouble = gradOutput.asInstanceOf[Tensor[Double]]
        val gradOutputData = gradOutputDouble.storage().array()
        val gradOutputOffset = gradOutputDouble.storageOffset() - 1
        val gradOutputStride = gradOutputDouble.stride(1)
        val gradOutputStride2 = gradOutputDouble.stride(2)
        if (affine) {
          if (theGradInput != null) {
            val gradInputDouble = theGradInput.asInstanceOf[Tensor[Double]]
            val gradInputData = gradInputDouble.storage().array()
            val gradInputOffset = gradInputDouble.storageOffset() - 1
            val gradInputStride = gradInputDouble.stride(1)
            val gradInputStride2 = gradInputDouble.stride(2)
            if (theGradWeight != null && theGradBias != null) {
              val gradWeightDouble = theGradWeight.asInstanceOf[Tensor[Double]]
              val gradWeightData = gradWeightDouble.storage().array()
              val gradWeightOffset = gradWeightDouble.storageOffset() - 1
              val gradBiasDouble = theGradBias.asInstanceOf[Tensor[Double]]
              val gradBiasData = gradBiasDouble.storage().array()
              val gradBiasOffset = gradBiasDouble.storageOffset() - 1
              backwardDouble(inputData, inputOffset, inputStride, inputStride2, gradOutputData,
                gradOutputOffset, gradOutputStride, gradOutputStride2,
                gradInputData, gradInputOffset, gradInputStride, gradInputStride2, nInput, n,
                scaleW, scaleB,
                gradWeightData, gradWeightOffset, gradBiasData,
                gradBiasOffset)
            } else {
              backwardDouble(inputData, inputOffset, inputStride, inputStride2, gradOutputData,
                gradOutputOffset, gradOutputStride, gradOutputStride2,
                gradInputData, gradInputOffset, gradInputStride, gradInputStride2, nInput, n,
                scaleW, scaleB, null, 0, null, 0)
            }
          } else {
            val gradWeightDouble = theGradWeight.asInstanceOf[Tensor[Double]]
            val gradWeightData = gradWeightDouble.storage().array()
            val gradWeightOffset = gradWeightDouble.storageOffset() - 1
            val gradBiasDouble = theGradBias.asInstanceOf[Tensor[Double]]
            val gradBiasData = gradBiasDouble.storage().array()
            val gradBiasOffset = gradBiasDouble.storageOffset() - 1
            backwardDouble(inputData, inputOffset, inputStride, inputStride2, gradOutputData,
              gradOutputOffset, gradOutputStride, gradOutputStride2,
              null, 0, 0, 0, nInput, n, scaleW, scaleB,
              gradWeightData, gradWeightOffset,
              gradBiasData, gradBiasOffset)
          }
        } else if (null != theGradInput) {
          val gradInputDouble = theGradInput.asInstanceOf[Tensor[Double]]
          val gradInputData = gradInputDouble.storage().array()
          val gradInputOffset = gradInputDouble.storageOffset() - 1
          val gradInputStride = gradInputDouble.stride(1)
          val gradInputStride2 = gradInputDouble.stride(2)
          backwardDouble(inputData, inputOffset, inputStride, inputStride2, gradOutputData,
            gradOutputOffset, gradOutputStride, gradOutputStride2,
            gradInputData, gradInputOffset, gradInputStride, gradInputStride2, nInput, n,
            scaleW, scaleB, null, 0, null, 0)
        }

      case FloatType =>
        val inputFloat = input.asInstanceOf[Tensor[Float]]
        val inputData = inputFloat.storage().array()
        val inputOffset = inputFloat.storageOffset() - 1
        val inputStride = input.stride(1)
        val inputStride2 = input.stride(2)
        val gradOutputFloat = gradOutput.asInstanceOf[Tensor[Float]]
        val gradOutputData = gradOutputFloat.storage().array()
        val gradOutputOffset = gradOutputFloat.storageOffset() - 1
        val gradOutputStride = gradOutputFloat.stride(1)
        val gradOutputStride2 = gradOutputFloat.stride(2)
        if (affine) {
          if (theGradInput != null) {
            val gradInputFloat = theGradInput.asInstanceOf[Tensor[Float]]
            val gradInputData = gradInputFloat.storage().array()
            val gradInputOffset = gradInputFloat.storageOffset() - 1
            val gradInputStride = gradInputFloat.stride(1)
            val gradInputStride2 = gradInputFloat.stride(2)
            if (theGradWeight != null && theGradBias != null) {
              val gradWeightFloat = theGradWeight.asInstanceOf[Tensor[Float]]
              val gradWeightData = gradWeightFloat.storage().array()
              val gradWeightOffset = gradWeightFloat.storageOffset() - 1
              val gradBiasFloat = theGradBias.asInstanceOf[Tensor[Float]]
              val gradBiasData = gradBiasFloat.storage().array()
              val gradBiasOffset = gradBiasFloat.storageOffset() - 1
              backwardFloat(inputData, inputOffset, inputStride, inputStride2, gradOutputData,
                gradOutputOffset, gradOutputStride, gradOutputStride2,
                gradInputData, gradInputOffset, gradInputStride, gradInputStride2, nInput, n,
                ev.toType[Float](ev.fromType[Double](scaleW)),
                ev.toType[Float](ev.fromType[Double](scaleB)),
                gradWeightData, gradWeightOffset, gradBiasData,
                gradBiasOffset)
            } else {
              backwardFloat(inputData, inputOffset, inputStride, inputStride2, gradOutputData,
                gradOutputOffset, gradOutputStride, gradOutputStride2,
                gradInputData, gradInputOffset, gradInputStride, gradInputStride2, nInput, n,
                ev.toType[Float](ev.fromType[Double](scaleW)),
                ev.toType[Float](ev.fromType[Double](scaleB)),
                null, 0, null, 0)
            }
          } else {
            val gradWeightFloat = theGradWeight.asInstanceOf[Tensor[Float]]
            val gradWeightData = gradWeightFloat.storage().array()
            val gradWeightOffset = gradWeightFloat.storageOffset() - 1
            val gradBiasFloat = theGradBias.asInstanceOf[Tensor[Float]]
            val gradBiasData = gradBiasFloat.storage().array()
            val gradBiasOffset = gradBiasFloat.storageOffset() - 1
            backwardFloat(inputData, inputOffset, inputStride, inputStride2, gradOutputData,
              gradOutputOffset, gradOutputStride, gradOutputStride2,
              null, 0, 0, 0, nInput, n,
              ev.toType[Float](ev.fromType[Double](scaleW)),
              ev.toType[Float](ev.fromType[Double](scaleB)),
              gradWeightData, gradWeightOffset,
              gradBiasData, gradBiasOffset)
          }
        } else if (null != theGradInput) {
          val gradInputFloat = theGradInput.asInstanceOf[Tensor[Float]]
          val gradInputData = gradInputFloat.storage().array()
          val gradInputOffset = gradInputFloat.storageOffset() - 1
          val gradInputStride = gradInputFloat.stride(1)
          val gradInputStride2 = gradInputFloat.stride(2)
          backwardFloat(inputData, inputOffset, inputStride, inputStride2, gradOutputData,
            gradOutputOffset, gradOutputStride, gradOutputStride2,
            gradInputData, gradInputOffset, gradInputStride, gradInputStride2, nInput, n,
            ev.toType[Float](ev.fromType[Double](scaleW)),
            ev.toType[Float](ev.fromType[Double](scaleB)),
            null, 0, null, 0)
        }
    }

    gradInput
  }

  private def backwardDouble(input: Array[Double], inputOffset: Int, inputStride: Int,
      inputStride2: Int, gradOutput: Array[Double], gradOutputOffset: Int, gradOutputStride: Int,
      gradOutputStride2: Int, gradInput: Array[Double], gradInputOffset: Int, gradInputStride: Int,
      gradInputStride2: Int, nInput: Int, n: Int, scaleW: Double, scaleB: Double,
      gradWeight: Array[Double], gradWeightOffset: Int,
      gradBias: Array[Double], gradBiasOffset: Int): Unit = {
    var f = 0
    while (f < nInput) {
      val _f = f + 1
      results(f) = Engine.model.invoke(() => {
        val w = if (null != weight) ev.toType[Double](weight.valueAt(_f)) else 1.0
        val (mean, invstd) = if (train) {
          (ev.toType[Double](saveMean.valueAt(_f)), ev.toType[Double](saveStd.valueAt(_f)))
        } else {
          (ev.toType[Double](runningMean.valueAt(_f)),
            1 / Math.sqrt(ev.toType[Double](runningVar.valueAt(_f)) + eps))
        }

        var sum = 0.0
        var i = 0
        while (i < n) {
          val index = i % gradOutputStride2 + (_f - 1) * gradOutputStride2 + gradOutputOffset +
            (i / gradOutputStride2) * gradOutputStride
          sum += gradOutput(index)
          i += 1
        }

        var dotp = 0.0
        i = 0
        while (i < n) {
          val inputIndex = i % inputStride2 + (_f - 1) * inputStride2 + inputOffset +
            (i / inputStride2) * inputStride
          val gradOutputIndex = i % gradOutputStride2 + (_f - 1) * gradOutputStride2 +
            gradOutputOffset + (i / gradOutputStride2) * gradOutputStride
          dotp += (input(inputIndex) - mean) * gradOutput(gradOutputIndex)
          i += 1
        }

        if (null != gradInput) {
          if (train) {
            val k = dotp * invstd * invstd / n
            i = 0
            while (i < n) {
              val inputIndex = i % inputStride2 + (_f - 1) * inputStride2 + inputOffset +
                (i / inputStride2) * inputStride
              val gradInputIndex = i % gradInputStride2 + (_f - 1) * gradInputStride2 +
                gradInputOffset + (i / gradInputStride2) * gradInputStride
              gradInput(gradInputIndex) = (input(inputIndex) - mean) * k
              i += 1
            }

            val gradMean = sum / n
            i = 0
            while (i < n) {
              val gradInputIndex = i % gradInputStride2 + (_f - 1) * gradInputStride2 +
                gradInputOffset + (i / gradInputStride2) * gradInputStride
              val gradOutputIndex = i % gradOutputStride2 + (_f - 1) * gradOutputStride2 +
                gradOutputOffset + (i / gradOutputStride2) * gradOutputStride
              gradInput(gradInputIndex) = (gradOutput(gradOutputIndex) - gradMean -
                gradInput(gradInputIndex)) * invstd * w
              i += 1
            }
          } else {
            var i = 0
            while (i < n) {
              val gradInputIndex = i % gradInputStride2 + (_f - 1) * gradInputStride2 +
                gradInputOffset + (i / gradInputStride2) * gradInputStride
              val gradOutputIndex = i % gradOutputStride2 + (_f - 1) * gradOutputStride2 +
                gradOutputOffset + (i / gradOutputStride2) * gradOutputStride
              gradInput(gradInputIndex) = gradOutput(gradOutputIndex) * invstd * w
              i += 1
            }
          }
        }

        if (null != gradWeight && scaleW != 0) {
          gradWeight(_f - 1 + gradWeightOffset) += scaleW * dotp * invstd
        }

        if (null != gradBias && scaleB != 0) {
          gradBias(_f - 1 + gradBiasOffset) += scaleB * sum
        }
      })
      f += 1
    }
    Engine.model.sync(results)
  }

  private def backwardFloat(input: Array[Float], inputOffset: Int, inputStride: Int,
      inputStride2: Int, gradOutput: Array[Float], gradOutputOffset: Int, gradOutputStride: Int,
      gradOutputStride2: Int, gradInput: Array[Float], gradInputOffset: Int, gradInputStride: Int,
      gradInputStride2: Int, nInput: Int, n: Int, scaleW: Float, scaleB: Float,
      gradWeight: Array[Float], gradWeightOffset: Int,
      gradBias: Array[Float], gradBiasOffset: Int): Unit = {
    // do relu first
    if (gradInput != null) {
      val outputArray = this.output.asInstanceOf[Tensor[Float]].storage().array()
      val outputOffset = this.output.storageOffset() - 1
      var i = 0
      while (i < outputArray.length) {
        if (outputArray(i + outputOffset) == 0) {
          gradOutput(i + gradOutputOffset) = 0
        }
        i += 1
      }
    }
    var f = 0
    while (f < nInput) {
      val _f = f + 1
      results(f) = Engine.model.invoke(() => {
        val w = if (null != weight) ev.toType[Float](weight.valueAt(_f)) else 1.0f
        val (mean, invstd) = if (train) {
          (ev.toType[Float](saveMean.valueAt(_f)), ev.toType[Float](saveStd.valueAt(_f)))
        } else {
          (ev.toType[Float](runningMean.valueAt(_f)),
            1 / Math.sqrt(ev.toType[Float](runningVar.valueAt(_f)) + eps).toFloat)
        }

        var sum = 0.0f
        var i = 0
        while (i < n) {
          val index = i % gradOutputStride2 + (_f - 1) * gradOutputStride2 + gradOutputOffset +
            (i / gradOutputStride2) * gradOutputStride
          sum += gradOutput(index)
          i += 1
        }

        var dotp = 0.0f
        i = 0
        while (i < n) {
          val inputIndex = i % inputStride2 + (_f - 1) * inputStride2 + inputOffset +
            (i / inputStride2) * inputStride
          val gradOutputIndex = i % gradOutputStride2 + (_f - 1) * gradOutputStride2 +
            gradOutputOffset + (i / gradOutputStride2) * gradOutputStride
          dotp += (input(inputIndex) - mean) * gradOutput(gradOutputIndex)
          i += 1
        }

        if (null != gradInput) {
          if (train) {
            val k = dotp * invstd * invstd / n
            i = 0
            while (i < n) {
              val inputIndex = i % inputStride2 + (_f - 1) * inputStride2 + inputOffset +
                (i / inputStride2) * inputStride
              val gradInputIndex = i % gradInputStride2 + (_f - 1) * gradInputStride2 +
                gradInputOffset + (i / gradInputStride2) * gradInputStride
              gradInput(gradInputIndex) = (input(inputIndex) - mean) * k
              i += 1
            }

            val gradMean = sum / n
            i = 0
            while (i < n) {
              val gradInputIndex = i % gradInputStride2 + (_f - 1) * gradInputStride2 +
                gradInputOffset + (i / gradInputStride2) * gradInputStride
              val gradOutputIndex = i % gradOutputStride2 + (_f - 1) * gradOutputStride2 +
                gradOutputOffset + (i / gradOutputStride2) * gradOutputStride
              gradInput(gradInputIndex) = (gradOutput(gradOutputIndex) - gradMean -
                gradInput(gradInputIndex)) * invstd * w
              i += 1
            }
          } else {
            var i = 0
            while (i < n) {
              val gradInputIndex = i % gradInputStride2 + (_f - 1) * gradInputStride2 +
                gradInputOffset + (i / gradInputStride2) * gradInputStride
              val gradOutputIndex = i % gradOutputStride2 + (_f - 1) * gradOutputStride2 +
                gradOutputOffset + (i / gradOutputStride2) * gradOutputStride
              gradInput(gradInputIndex) = gradOutput(gradOutputIndex) * invstd * w
              i += 1
            }
          }
        }

        if (null != gradWeight && scaleW != 0) {
          gradWeight(_f - 1 + gradWeightOffset) += scaleW * dotp * invstd
        }

        if (null != gradBias && scaleB != 0) {
          gradBias(_f - 1 + gradBiasOffset) += scaleB * sum
        }
      })
      f += 1
    }
    Engine.model.sync(results)
  }

  override def zeroGradParameters(): Unit = {
    if (affine) {
      gradWeight.zero()
      gradBias.zero()
    }
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    if (affine) {
      (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
    } else {
      null
    }
  }

  override def getParametersTable(): Table = {
    if (affine) {
      T(getName() -> T("weight" -> weight, "bias" -> bias,
        "gradWeight" -> gradWeight, "gradBias" -> gradBias,
        "runningMean" -> runningMean, "runningVar" -> runningVar))
    } else {
      T(getName() -> T("runningMean" -> runningMean, "runningVar" -> runningVar))
    }
  }

  override def toString(): String = {
    s"nn.SpatialBatchNormalizationWithRelu($nOutput, $eps, $momentum, $affine)"
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[BatchNormalization[T]]

  override def equals(other: Any): Boolean = other match {
    case that: BatchNormalization[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        nDim == that.nDim &&
        runningMean == that.runningMean &&
        runningVar == that.runningVar &&
        weight == that.weight &&
        bias == that.bias &&
        nOutput == that.nOutput &&
        eps == that.eps &&
        momentum == that.momentum &&
        affine == that.affine
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), nDim, runningMean, runningVar, weight, bias,
      nOutput, eps, momentum, affine)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object SpatialBatchNormalizationWithRelu extends ModuleSerializable {
  def apply[@specialized(Float, Double) T: ClassTag](
        nOutput: Int,
        eps: Double = 1e-5,
        momentum: Double = 0.1,
        affine: Boolean = true,
        initWeight: Tensor[T] = null,
        initBias: Tensor[T] = null,
        initGradWeight: Tensor[T] = null,
        initGradBias: Tensor[T] = null)(
    implicit ev: TensorNumeric[T]): SpatialBatchNormalizationWithRelu[T] = {
    new SpatialBatchNormalizationWithRelu[T](
      nOutput, eps, momentum, affine, initWeight, initBias, initGradWeight, initGradBias)
  }
  def apply[@specialized(Float, Double) T: ClassTag](
        affine: Option[Int])(
    implicit ev: TensorNumeric[T]): SpatialBatchNormalizationWithRelu[T] = {
    new SpatialBatchNormalizationWithRelu[T](
      nOutput = affine.getOrElse(1), affine = affine.isDefined)
  }

  private[nn] def updateOutputFloat(
                                     n: Int, nInput: Int,
                                     momentum: Float, eps: Float,
                                     train: Boolean, spatialBatchSize: Int,
                                     nDim: Float, needFix: Boolean,
                                     input: Tensor[Float], output: Tensor[Float],
                                     ones: Tensor[Float],
                                     saveMean: Tensor[Float],
                                     saveStd: Tensor[Float],
                                     runningMean: Tensor[Float],
                                     runningVar: Tensor[Float],
                                     weight: Tensor[Float],
                                     bias: Tensor[Float]
                               ): Unit = {
    val inputArray = input.storage().array()
    val inputOffset = input.storageOffset() - 1
    val inputStride = input.stride(1)
    val inputStride2 = input.stride(2)
    val outputArray = output.storage().array()
    val outputOffset = output.storageOffset() - 1
    val outputStride = output.stride(1)
    val onesArray = ones.storage().array()
    val saveMeanArray = saveMean.storage.array
    val saveMeanOffset = saveMean.storageOffset - 1
    val saveStdArray = saveStd.storage.array
    val saveStdOffset = saveStd.storageOffset - 1
    val runningMeanArray = runningMean.storage.array
    val runningMeanOffset = runningMean.storageOffset - 1
    val runningVarArray = runningVar.storage.array
    val runningVarOffset = runningVar.storageOffset - 1
    val weightArray = weight.storage.array
    val weightOffset = weight.storageOffset - 1
    val biasArray = bias.storage.array
    val biasOffset = bias.storageOffset - 1
    val frameSize = if (nDim == 2) n else n / spatialBatchSize

    if (train) {
      var f = 1
      while (f <= nInput) {
        var sum = 0.0f
        var i = 0
        while (i < spatialBatchSize) {
          sum += MKL.vsdot(frameSize, inputArray, inputOffset + (f - 1 + i * nInput)
            * inputStride2, inputStride, onesArray, 0, 1)
          i += 1
        }
        val mean = sum / n
        saveMeanArray(f - 1) = mean

        // output has the same elements with input
        i = 0
        while (i < spatialBatchSize) {
          MKL.vsaxpy(frameSize, -mean, onesArray, 0, 1, outputArray,
            inputOffset + (f - 1 + i * nInput), inputStride)
          i += 1
        }

        f += 1
      }

      MKL.vsPowx(n * nInput, outputArray, inputOffset, 2, outputArray, inputOffset)

      f = 1
      while (f <= nInput) {
        val mean = saveMeanArray(f - 1)
        var sum = 0.0f
        var i = 0
        while (i < spatialBatchSize) {
          sum +=  MKL.vsdot(frameSize, outputArray,
            inputOffset + (f - 1 + i * nInput) * inputStride2,
            inputStride, onesArray, 0, 1)
          i += 1
        }

        val invstd = if (sum == 0 && eps == 0.0) {
          0.0f
        } else {
          1.0f / Math.sqrt(sum / n + eps).toFloat
        }
        saveStdArray(f - 1) = invstd

        runningMeanArray(f - 1) = momentum * mean  + (1- momentum) * runningMeanArray(f - 1)

        val unbiasedVar = sum / (n - 1)
        runningVarArray(f - 1) = momentum * unbiasedVar + (1 - momentum) * runningVarArray(f - 1)
        f += 1
      }
      System.arraycopy(inputArray, inputOffset, outputArray, outputOffset, n * nInput)
    }

    if (needFix) {
      java.util.Arrays.fill(saveMeanArray,
        saveMeanOffset, saveMeanOffset + nInput, 0f)
      java.util.Arrays.fill(saveStdArray,
        saveStdOffset, saveStdOffset + nInput, 0.0001f)
    }

    // update output
    var f = 1
    while (f <= nInput) {
      val (mean, invstd) = if (train) {
        (saveMeanArray(f - 1), saveStdArray(f - 1))
      } else {
        (runningMeanArray(f - 1),
          1f / Math.sqrt(runningVarArray(f - 1) + eps).toFloat)
      }

      val w = if (null != weightArray) weightArray(f - 1 + weightOffset) else 1.0f
      val b = if (null != biasArray) biasArray(f - 1 + biasOffset) else 0.0f

      var i = 0
      while (i < spatialBatchSize) {
        MKL.vsscal(frameSize, w * invstd, outputArray,
          inputOffset + (f - 1 + i * nInput) * inputStride2, inputStride)
        MKL.vsaxpy(frameSize, b - mean * invstd * w, onesArray, 0, 1,
          outputArray, inputOffset + (f - 1 + i * nInput) * inputStride2, inputStride)
        i += 1
      }

      f += 1
    }
  }

  private[nn] def updateOutputDouble(
                                     n: Int, nInput: Int,
                                     momentum: Double, eps: Double,
                                     train: Boolean, spatialBatchSize: Int,
                                     nDim: Double, needFix: Boolean,
                                     input: Tensor[Double], output: Tensor[Double],
                                     ones: Tensor[Double],
                                     saveMean: Tensor[Double],
                                     saveStd: Tensor[Double],
                                     runningMean: Tensor[Double],
                                     runningVar: Tensor[Double],
                                     weight: Tensor[Double],
                                     bias: Tensor[Double]
                                   ): Unit = {
    val inputArray = input.storage().array()
    val inputOffset = input.storageOffset() - 1
    val inputStride = input.stride(1)
    val inputStride2 = input.stride(2)
    val outputArray = output.storage().array()
    val outputOffset = output.storageOffset() - 1
    val outputStride = output.stride(1)
    val onesArray = ones.storage().array()
    val saveMeanArray = saveMean.storage.array
    val saveMeanOffset = saveMean.storageOffset - 1
    val saveStdArray = saveStd.storage.array
    val saveStdOffset = saveStd.storageOffset - 1
    val runningMeanArray = runningMean.storage.array
    val runningMeanOffset = runningMean.storageOffset - 1
    val runningVarArray = runningVar.storage.array
    val runningVarOffset = runningVar.storageOffset - 1
    val weightArray = weight.storage.array
    val weightOffset = weight.storageOffset - 1
    val biasArray = bias.storage.array
    val biasOffset = bias.storageOffset - 1
    val frameSize = if (nDim == 2) n else n / spatialBatchSize

    if (train) {
      var f = 1
      while (f <= nInput) {
        var sum = 0.0
        var i = 0
        while (i < spatialBatchSize) {
          sum += MKL.vddot(frameSize, inputArray, inputOffset + (f - 1 + i * nInput)
            * inputStride2, 1, onesArray, 0, 1)
          i += 1
        }
        val mean = sum / n
        saveMeanArray(f - 1) = mean

        // output has the same elements with input
        i = 0
        while (i < spatialBatchSize) {
          MKL.vdaxpy(frameSize, -mean, onesArray, 0, 1, outputArray,
            inputOffset + (f - 1 + i * nInput), inputStride)
          i += 1
        }

        f += 1
      }

      MKL.vdPowx(n * nInput, outputArray, inputOffset, 2, outputArray, inputOffset)

      f = 1
      while (f <= nInput) {
        val mean = saveMeanArray(f - 1)
        var sum = 0.0
        var i = 0
        while (i < spatialBatchSize) {
          sum +=  MKL.vddot(frameSize, outputArray,
            inputOffset + (f - 1 + i * nInput) * inputStride2,
            inputStride, onesArray, 0, 1)
          i += 1
        }

        val invstd = if (sum == 0 && eps == 0.0) {
          0.0
        } else {
          1.0 / Math.sqrt(sum / n + eps)
        }
        saveStdArray(f - 1) = invstd

        runningMeanArray(f - 1) = momentum * mean  + (1- momentum) * runningMeanArray(f - 1)

        val unbiasedVar = sum / (n - 1)
        runningVarArray(f - 1) = momentum * unbiasedVar + (1 - momentum) * runningVarArray(f - 1)
        f += 1
      }
      System.arraycopy(inputArray, inputOffset, outputArray, outputOffset, n * nInput)
    }

    if (needFix) {
      java.util.Arrays.fill(saveMeanArray,
        saveMeanOffset, saveMeanOffset + nInput, 0)
      java.util.Arrays.fill(saveStdArray,
        saveStdOffset, saveStdOffset + nInput, 0.0001)
    }

    // update output
    var f = 1
    while (f <= nInput) {
      val (mean, invstd) = if (train) {
        (saveMeanArray(f - 1), saveStdArray(f - 1))
      } else {
        (runningMeanArray(f - 1),
          1 / Math.sqrt(runningVarArray(f - 1) + eps))
      }

      val w = if (null != weightArray) weightArray(f - 1 + weightOffset) else 1.0
      val b = if (null != biasArray) biasArray(f - 1 + biasOffset) else 0.0

      var i = 0
      while (i < spatialBatchSize) {
        MKL.vdscal(frameSize, w * invstd, outputArray,
          inputOffset + (f - 1 + i * nInput) * inputStride2, inputStride)
        MKL.vdaxpy(frameSize, b - mean * invstd * w, onesArray, 0, 1,
          outputArray, inputOffset + (f - 1 + i * nInput) * inputStride2, inputStride)
        i += 1
      }


      f += 1
    }
  }
}
