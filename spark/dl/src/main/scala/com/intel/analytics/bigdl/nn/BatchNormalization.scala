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
import com.intel.analytics.bigdl.nn.abstractnn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, Tensor}
import com.intel.analytics.bigdl.utils.{Engine, T, Table}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.serializer._
import serialization.Bigdl.{AttrValue, BigDLModule}

import scala.concurrent.Future
import scala.reflect.ClassTag

/**
 * This layer implements Batch Normalization as described in the paper:
 * "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
 * by Sergey Ioffe, Christian Szegedy https://arxiv.org/abs/1502.03167
 *
 * This implementation is useful for inputs NOT coming from convolution layers.
 * For convolution layers, use nn.SpatialBatchNormalization.
 *
 * The operation implemented is:
 *             ( x - mean(x) )
 *     y =  -------------------- * gamma + beta
 *             standard-deviation(x)
 * where gamma and beta are learnable parameters.The learning of gamma and beta is optional.
 * @param nOutput output feature map number
 * @param eps avoid divide zero
 * @param momentum momentum for weight update
 * @param affine affine operation on output or not
 * @param ev numeric operator
 * @tparam T numeric type
 */
@SerialVersionUID(- 3181824540272906068L)
class BatchNormalization[T: ClassTag](
  val nOutput: Int, // output feature map number
  val eps: Double = 1e-5, // avoid divde zero
  val momentum: Double = 0.1, // momentum for weight update
  val affine: Boolean = true, // affine operation on output or not
  private val initWeight: Tensor[T] = null,
  private val initBias: Tensor[T] = null,
  private val initGradWeight: Tensor[T] = null,
  private val initGradBias: Tensor[T] = null,
  val dataFormat: DataFormat = DataFormat.NCHW
)(implicit ev: TensorNumeric[T]) extends TensorModule[T] with Initializable {

  require(nOutput > 0)

  val nDim = 4
  var runningMean = if (affine) Tensor[T](nOutput) else Tensor[T]()
  var runningVar = if (affine) Tensor[T](nOutput).fill(ev.fromType[Int](1)) else Tensor[T]()
  var saveMean = if (affine) Tensor[T](nOutput) else Tensor[T]()
  var saveStd = if (affine) Tensor[T](nOutput).fill(ev.zero) else Tensor[T]()

  val weight: Tensor[T] =
    if (initWeight != null) initWeight else if (affine) Tensor[T](nOutput) else null
  val bias: Tensor[T] =
    if (initBias != null) initBias else if (affine) Tensor[T](nOutput) else null

  val gradWeight: Tensor[T] =
    if (initGradWeight != null) initGradWeight else if (affine) Tensor[T](nOutput) else null
  val gradBias: Tensor[T] =
    if (initGradBias != null) initGradBias else if (affine) Tensor[T](nOutput) else null

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
  // to fix internal parameters (saveMean, saveStd)
  def setInit(status: Boolean = true): this.type = {
    needFix = status
    this
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
    saveStd.resizeAs(runningVar).fill(ev.zero)

    if (results == null || results.length > nInput) {
      results = new Array[Future[_]](nInput)
    }
    val n = _input.nElement() / nInput
    ev.getType() match {
      case DoubleType =>
        val inputDouble = _input.asInstanceOf[Tensor[Double]]
        val inputData = inputDouble.storage().array()
        val inputOffset = inputDouble.storageOffset() - 1
        val inputStride = _input.stride(1)
        val inputStride2 = _input.stride(2)
        val outputDouble = output.asInstanceOf[Tensor[Double]]
        val outputData = outputDouble.storage().array()
        val outputOffset = outputDouble.storageOffset() - 1
        val outputStride = output.stride(1)
        require(dataFormat == DataFormat.NCHW, "BatchNorm Double only support NCHW")
        updateOutputDouble(inputData, inputOffset, inputStride, outputData, outputOffset,
          outputStride, nInput, n, inputStride2)

      case FloatType =>
        val inputFloat = _input.asInstanceOf[Tensor[Float]]
        val inputData = inputFloat.storage().array()
        val inputOffset = inputFloat.storageOffset() - 1
        val inputStride = _input.stride(1)
        val inputStride2 = _input.stride(2)
        val outputFloat = output.asInstanceOf[Tensor[Float]]
        val outputData = outputFloat.storage().array()
        val outputOffset = outputFloat.storageOffset() - 1
        val outputStride = output.stride(1)
        if (dataFormat == DataFormat.NCHW) {
          val isOld = true
          if (isOld) {
            updateOutputFloat(inputData, inputOffset, inputStride, outputData, outputOffset,
              outputStride, nInput, n, inputStride2)
          } else {
            require(ev.getType() == FloatType, "BatchNorm NCHW only support Float type")
            if (train) {
              BatchNormalization.updateOutputFloatNCHWTrain(inputFloat, outputFloat,
                saveMean.asInstanceOf[Tensor[Float]], saveStd.asInstanceOf[Tensor[Float]],
                runningMean.asInstanceOf[Tensor[Float]], runningVar.asInstanceOf[Tensor[Float]],
                weight.asInstanceOf[Tensor[Float]], bias.asInstanceOf[Tensor[Float]],
                eps.toFloat, momentum.toFloat)
            } else {
              ???
            }
          }
        } else {
          require(ev.getType() == FloatType, "BatchNorm NHWC only support Float type")
          if (train) {
            BatchNormalization.updateOutputFloatNHWCTrain(inputFloat, outputFloat,
              saveMean.asInstanceOf[Tensor[Float]], saveStd.asInstanceOf[Tensor[Float]],
              runningMean.asInstanceOf[Tensor[Float]], runningVar.asInstanceOf[Tensor[Float]],
              weight.asInstanceOf[Tensor[Float]], bias.asInstanceOf[Tensor[Float]],
              eps.toFloat, momentum.toFloat)
          } else {
            BatchNormalization.updateOutputFloatNHWCInfer(inputFloat, outputFloat,
              runningMean.asInstanceOf[Tensor[Float]], runningVar.asInstanceOf[Tensor[Float]],
              weight.asInstanceOf[Tensor[Float]], bias.asInstanceOf[Tensor[Float]])
          }
        }
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

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (dataFormat == DataFormat.NCHW) {
      val isOld = false
      if (isOld) {
        backward(input, gradOutput, gradInput, null, null)
      } else {
        require(ev.getType() == FloatType, "NCHW only support float now")
        BatchNormalization.updateGradInputFloatNCHWTrain(
          input.asInstanceOf[Tensor[Float]], gradOutput.asInstanceOf[Tensor[Float]],
          gradInput.asInstanceOf[Tensor[Float]], weight.asInstanceOf[Tensor[Float]],
          saveMean.asInstanceOf[Tensor[Float]], saveStd.asInstanceOf[Tensor[Float]], gMean, gxMean)
        gradInput
      }
    } else {
      require(ev.getType() == FloatType, "NHWC only support float now")
      BatchNormalization.updateGradInputFloatNHWCTrain(
        input.asInstanceOf[Tensor[Float]], gradOutput.asInstanceOf[Tensor[Float]],
        gradInput.asInstanceOf[Tensor[Float]], weight.asInstanceOf[Tensor[Float]],
        saveMean.asInstanceOf[Tensor[Float]], saveStd.asInstanceOf[Tensor[Float]], gMean, gxMean)
      gradInput
    }
  }

  private val gMean = Tensor[Float]()
  private val gxMean = Tensor[Float]()

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

  def backward(input: Tensor[T], gradOutput: Tensor[T],
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
    gradWeight: Array[Double], gradWeightOffset: Int, gradBias: Array[Double], gradBiasOffset: Int
  ): Unit = {
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
    gradWeight: Array[Float], gradWeightOffset: Int, gradBias: Array[Float], gradBiasOffset: Int
  ): Unit = {
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

  override def copyStatus(src: Module[T]): this.type = {
    require(canEqual(src), s"copyStatus: type mismatch, $src is different from $this")
    val srcModule = src.asInstanceOf[BatchNormalization[T]]
    runningMean.copy(srcModule.runningMean)
    runningVar.copy(srcModule.runningVar)
    this
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
    s"nn.BatchNormalization($nOutput, $eps, $momentum, $affine)"
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

object BatchNormalization extends ModuleSerializable {
  def apply[@specialized(Float, Double) T: ClassTag](
    nOutput: Int,
    eps: Double = 1e-5,
    momentum: Double = 0.1,
    affine: Boolean = true,
    initWeight: Tensor[T] = null,
    initBias: Tensor[T] = null,
    initGradWeight: Tensor[T] = null,
    initGradBias: Tensor[T] = null,
    dataFormat: DataFormat = DataFormat.NCHW)
    (implicit ev: TensorNumeric[T]): BatchNormalization[T] = {

    new BatchNormalization[T](
      nOutput, eps, momentum, affine, initWeight, initBias, initGradWeight, initGradBias,
      dataFormat)
  }
  def apply[@specialized(Float, Double) T: ClassTag](
    affine: Option[Int])(implicit ev: TensorNumeric[T]): BatchNormalization[T] = {
    new BatchNormalization[T](nOutput = affine.getOrElse(1), affine = affine.isDefined)
  }

  override def doLoadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {
    val attrMap = context.bigdlModule.getAttrMap
    val batchNorm = super.doLoadModule(context).asInstanceOf[BatchNormalization[T]]

    batchNorm.runningMean = DataConverter.
      getAttributeValue(context, attrMap.get("runningMean")).
      asInstanceOf[Tensor[T]]

    batchNorm.runningVar = DataConverter.
      getAttributeValue(context, attrMap.get("runningVar")).
      asInstanceOf[Tensor[T]]

    batchNorm.saveMean = DataConverter.
      getAttributeValue(context, attrMap.get("saveMean")).
      asInstanceOf[Tensor[T]]

    batchNorm.saveStd = DataConverter.
      getAttributeValue(context, attrMap.get("saveStd")).
      asInstanceOf[Tensor[T]]

    batchNorm
  }
  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
                                              batchNormBuilder : BigDLModule.Builder)
                                             (implicit ev: TensorNumeric[T]) : Unit = {

    super.doSerializeModule(context, batchNormBuilder)

    val batchNorm = context.moduleData.module.asInstanceOf[BatchNormalization[T]]

    val runningMeanBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, runningMeanBuilder,
      batchNorm.runningMean, ModuleSerializer.tensorType)
    batchNormBuilder.putAttr("runningMean", runningMeanBuilder.build)

    val runningVarBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, runningVarBuilder,
      batchNorm.runningVar, ModuleSerializer.tensorType)
    batchNormBuilder.putAttr("runningVar", runningVarBuilder.build)

    val saveMeanBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, saveMeanBuilder,
      batchNorm.saveMean, ModuleSerializer.tensorType)
    batchNormBuilder.putAttr("saveMean", saveMeanBuilder.build)

    val saveStdBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, saveStdBuilder,
      batchNorm.saveStd, ModuleSerializer.tensorType)
    batchNormBuilder.putAttr("saveStd", saveStdBuilder.build)
  }

  private[bigdl] def updateOutputFloatNHWCInfer(input: Tensor[Float], output: Tensor[Float],
    mean: Tensor[Float], std: Tensor[Float], scale: Tensor[Float], offset: Tensor[Float]): Unit = {

    require(input.isContiguous(), "BatchNorm NHWC require a contiguous input")
    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1
    val outputData = output.storage().array()
    val outputOffset = output.storageOffset() - 1
    val nChannels = input.size(4)
    val n = input.nElement()
    val meanData = mean.storage().array()
    val stdData = std.storage().array()

    val scaleData = scale.storage().array()
    val offsetData = offset.storage().array()
    var i = 0
    while(i < n) {
      var c = 0
      while(c < nChannels) {
        outputData(i + outputOffset + c) = (inputData(i + inputOffset + c) - meanData(c)) *
          stdData(c) * scaleData(c) + offsetData(c)
        c += 1
      }
      i += nChannels
    }
  }

  private[bigdl] def updateOutputFloatNHWCTrain(input: Tensor[Float], output: Tensor[Float],
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
    val mean = saveMean.storage().array()
    var i = 0
    val n = input.nElement()
    val frameSize = n / nChannels
    while(i < n) {
      var c = 0
      while(c < nChannels) {
        mean(c) += inputData(inputOffset + i + c)
        c += 1
      }
      i += nChannels
    }

    var c = 0
    val runningMeanData = runningMean.storage().array()
    while(c < nChannels) {
      mean(c) /= frameSize
      runningMeanData(c) = mean(c) * momentum + (1 - momentum) * runningMeanData(c)
      c += 1
    }

    val std = saveStd.storage().array()
    i = 0
    while(i < n) {
      var c = 0
      while(c < nChannels) {
        val diff = (inputData(inputOffset + i + c) - mean(c))
        std(c) += diff * diff
        c += 1
      }
      i += nChannels
    }

    c = 0
    val runningVarData = runningVar.storage().array()
    while(c < nChannels) {
      if (std(c) == 0 && eps == 0) {
        std(c) = 0
        if (saveVar != null) {
          saveVar.setValue(c + 1, 0f)
        }
        if (batchVar != null) {
          batchVar.setValue(c + 1, 0f)
        }
      } else {
        val s = std(c)
        val unbiasedVar = s / (frameSize - 1)
        if (saveVar != null) {
          saveVar.setValue(c + 1, s / frameSize)
        }
        if (batchVar != null) {
          batchVar.setValue(c + 1, unbiasedVar)
        }
        std(c) = 1.0f / Math.sqrt(s / frameSize + eps).toFloat
        runningVarData(c) = momentum * unbiasedVar + (1 - momentum) * runningVarData(c)
      }
      c += 1
    }

    val scaleData = scale.storage().array()
    val offsetData = offset.storage().array()
    i = 0
    while(i < n) {
      var c = 0
      while(c < nChannels) {
        outputData(i + outputOffset + c) = (inputData(i + inputOffset + c) - mean(c)) * std(c) *
          scaleData(c) + offsetData(c)
        c += 1
      }
      i += nChannels
    }
  }

  private[bigdl] def updateOutputFloatNCHWInfer(input: Tensor[Float], output: Tensor[Float],
    mean: Tensor[Float], std: Tensor[Float], scale: Tensor[Float], offset: Tensor[Float]): Unit = {

    require(input.isContiguous(), "BatchNorm NCHW require a contiguous input")
    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1
    val outputData = output.storage().array()
    val outputOffset = output.storageOffset() - 1
    val meanData = mean.storage().array()
    val meanOffset = mean.storageOffset() - 1
    val stdData = std.storage().array()
    val stdOffset = std.storageOffset() - 1
    val nChannels = input.size(2)
    val nBatch = input.size(1)
    val nFrame = input.size(3) * input.size(4)

    val scaleData = scale.storage().array()
    val offsetData = offset.storage().array()
    var i = inputOffset
    var b = 0
    while(b < nBatch) {
      var c = 0
      while(c < nChannels) {
        var k = 0
        while(k < nFrame) {
          outputData(i + outputOffset) = (inputData(i) - meanData(c + meanOffset)) *
            stdData(c + stdOffset) * scaleData(c) + offsetData(c)
          k += 1
          i += 1
        }
        c += 1
      }
      b += 1
    }
  }

  private[bigdl] def updateGradInputFloatNHWCTrain(
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
    require(scale.size(1) == nChannel, "scale length is not consist with channel number")
    require(saveMean.size(1) == nChannel, "saveMean length is not consist with channel number")
    require(saveStd.size(1) == nChannel, "saveStd length is not consist with channel number")

    gradInput.resizeAs(gradOutput)
    if (gMean.isEmpty) {
      gMean.resize(scale.size(1))
      gxMean.resize(scale.size(1))
    }

    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1
    val gradOutputData = gradOutput.storage().array()
    val gradOutputOffset = gradOutput.storageOffset() - 1
    val gradInputData = gradInput.storage().array()
    val gradInputOffset = gradInput.storageOffset() - 1
    val scaleData = scale.storage().array()
    val scaleOffset = scale.storageOffset() - 1
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

    i = 0
    while(i < n) {
      var c = 0
      while(c < nChannel) {
        val invStd = saveStdData(saveStdOffset + c)
        gradInputData(gradInputOffset + i) = scaleData(scaleOffset + c) *
          invStd * (gradOutputData(gradOutputOffset + i) - gMeanData(c) -
          gxMeanData(c) * invStd * invStd * (inputData(inputOffset + i) -
            saveMeanData(saveMeanOffset + c)))
        c += 1
        i += 1
      }
    }
  }

  private[bigdl] def updateGradInputFloatNHWCInfer(
    gradOutput: Tensor[Float],
    gradInput: Tensor[Float],
    scale: Tensor[Float],
    saveStd: Tensor[Float]
  ): Unit = {
    require(gradOutput.nDimension() == 4, "BN require a 4D gradient")
    require(gradOutput.isContiguous(), "gradient is not contiguous")
    val nChannel = gradOutput.size(4)
    require(scale.size(1) == nChannel, "scale length is not consist with channel number")
    require(saveStd.size(1) == nChannel, "saveStd length is not consist with channel number")

    gradInput.resizeAs(gradOutput)
    val gradOutputData = gradOutput.storage().array()
    val gradOutputOffset = gradOutput.storageOffset() - 1
    val gradInputData = gradInput.storage().array()
    val gradInputOffset = gradInput.storageOffset() - 1
    val scaleData = scale.storage().array()
    val scaleOffset = scale.storageOffset() - 1
    val saveStdData = saveStd.storage().array()
    val saveStdOffset = saveStd.storageOffset() - 1

    val n = gradOutput.nElement()
    var i = 0
    while(i < n) {
      var c = 0
      while(c < nChannel) {
        val invStd = saveStdData(saveStdOffset + c)
        gradInputData(gradInputOffset + i) = scaleData(scaleOffset + c) *
          invStd * gradOutputData(gradOutputOffset + i)
        c += 1
        i += 1
      }
    }
  }

  private[bigdl] def updateOutputFloatNCHWTrain(input: Tensor[Float], output: Tensor[Float],
    saveMean: Tensor[Float], saveStd: Tensor[Float], runningMean: Tensor[Float],
    runningVar: Tensor[Float], scale: Tensor[Float], offset: Tensor[Float],
    eps: Float, momentum: Float,
    batchVar: Tensor[Float] = null, saveVar: Tensor[Float] = null): Unit = {
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
    val mean = saveMean.storage().array()
    var i = inputOffset
    var b = 0
    while(b < nBatch) {
      var c = 0
      while (c < nChannels) {
        var k = 0
        var meanSum = 0f
        while(k < nFrame) {
          meanSum += inputData(i)
          k += 1
          i += 1
        }
        mean(c) += meanSum
        c += 1
      }
      b += 1
    }

    val n = input.nElement()
    val frameSize = n / nChannels
    var c = 0
    val runningMeanData = runningMean.storage().array()
    while(c < nChannels) {
      mean(c) /= frameSize
      runningMeanData(c) = mean(c) * momentum + (1 - momentum) * runningMeanData(c)
      c += 1
    }

    val std = saveStd.storage().array()
    i = inputOffset
    b = 0
    while(b < nBatch) {
      var c = 0
      while(c < nChannels) {
        var k = 0
        var stdSum = 0f
        while(k < nFrame) {
          val diff = (inputData(i) - mean(c))
          stdSum += diff * diff
          k += 1
          i += 1
        }
        std(c) += stdSum
        c += 1
      }
      b += 1
    }

    c = 0
    val runningVarData = runningVar.storage().array()
    while(c < nChannels) {
      if (std(c) == 0 && eps == 0) {
        std(c) = 0
        if (saveVar != null) {
          saveVar.setValue(c + 1, 0f)
        }
        if (batchVar != null) {
          batchVar.setValue(c + 1, 0f)
        }
      } else {
        val s = std(c)
        val unbiasedVar = s / (frameSize - 1)
        if (saveVar != null) {
          saveVar.setValue(c + 1, s / frameSize)
        }
        if (batchVar != null) {
          batchVar.setValue(c + 1, unbiasedVar)
        }
        std(c) = 1.0f / Math.sqrt(s / frameSize + eps).toFloat
        runningVarData(c) = momentum * unbiasedVar + (1 - momentum) * runningVarData(c)
      }
      c += 1
    }

    val scaleData = scale.storage().array()
    val offsetData = offset.storage().array()
    i = inputOffset
    b = 0
    while(b < nBatch) {
      var c = 0
      while(c < nChannels) {
        var k = 0
        while(k < nFrame) {
          outputData(i) = (inputData(i) - mean(c)) * std(c) *
            scaleData(c) + offsetData(c)
          k += 1
          i += 1
        }
        c += 1
      }
      b += 1
    }
  }

  private[bigdl] def updateGradInputFloatNCHWTrain(
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
    val nChannel = gradOutput.size(2)
    require(scale.size(1) == nChannel, "scale length is not consist with channel number")
    require(saveMean.size(1) == nChannel, "saveMean length is not consist with channel number")
    require(saveStd.size(1) == nChannel, "saveStd length is not consist with channel number")

    gradInput.resizeAs(gradOutput)
    if (gMean.isEmpty) {
      gMean.resize(scale.size(1))
      gxMean.resize(scale.size(1))
    }

    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1
    val gradOutputData = gradOutput.storage().array()
    val gradOutputOffset = gradOutput.storageOffset() - 1
    val gradInputData = gradInput.storage().array()
    val gradInputOffset = gradInput.storageOffset() - 1
    val scaleData = scale.storage().array()
    val scaleOffset = scale.storageOffset() - 1
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

    var c = 0
    val size = n / nChannel
    while(c < nChannel) {
      gMeanData(c) /= size
      gxMeanData(c) /= size
      c += 1
    }

    i = 0
    b = 0
    while(b < nBatch) {
      var c = 0
      while(c < nChannel) {
        var k = 0
        while(k < frameSize) {
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
  }

  private[bigdl] def updateGradInputFloatNCHWInfer(
    gradOutput: Tensor[Float],
    gradInput: Tensor[Float],
    scale: Tensor[Float],
    saveStd: Tensor[Float]
  ): Unit = {
    require(gradOutput.nDimension() == 4, "BN require a 4D gradient")
    require(gradOutput.isContiguous(), "gradient is not contiguous")
    val nChannel = gradOutput.size(2)
    require(scale.size(1) == nChannel, "scale length is not consist with channel number")
    require(saveStd.size(1) == nChannel, "saveStd length is not consist with channel number")

    gradInput.resizeAs(gradOutput)
    val gradOutputData = gradOutput.storage().array()
    val gradOutputOffset = gradOutput.storageOffset() - 1
    val gradInputData = gradInput.storage().array()
    val gradInputOffset = gradInput.storageOffset() - 1
    val scaleData = scale.storage().array()
    val scaleOffset = scale.storageOffset() - 1
    val saveStdData = saveStd.storage().array()
    val saveStdOffset = saveStd.storageOffset() - 1

    val nBatch = gradOutput.size(1)
    val frameSize = gradOutput.size(3) * gradOutput.size(4)
    var b = 0
    var i = 0
    while(b < nBatch) {
      var c = 0
      while(c < nChannel) {
        var k = 0
        while(k < frameSize) {
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
  }

  private[bigdl] def accGradientNHWC(gradOutput: Tensor[Float],
    gradWeight: Tensor[Float], gradBias: Tensor[Float],
    input: Tensor[Float], saveMean: Tensor[Float],
    saveStd: Tensor[Float]): Unit = {
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
          saveStdData(saveStdOffset + c)
        gradBiasData(c + gradBiasOffset) += g
        i += 1
        c += 1
      }
    }
  }

  private[bigdl] def accGradientNCHW(gradOutput: Tensor[Float],
    gradWeight: Tensor[Float], gradBias: Tensor[Float],
    input: Tensor[Float], saveMean: Tensor[Float],
    saveStd: Tensor[Float]): Unit = {
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
            saveStdData(saveStdOffset + c)
          gradBiasData(c + gradBiasOffset) += g
          k += 1
          i += 1
        }
        c += 1
      }
      b += 1
    }
  }
}
