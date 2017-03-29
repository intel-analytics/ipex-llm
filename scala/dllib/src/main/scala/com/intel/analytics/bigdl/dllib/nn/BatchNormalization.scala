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
import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, Tensor}
import com.intel.analytics.bigdl.utils.{Engine, T, Table}
import com.intel.analytics.bigdl.utils.RandomGenerator._

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
 *     y =     ( x - mean(x) )
 *          -------------------- * gamma + beta
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
class BatchNormalization[@specialized(Float, Double) T: ClassTag](
  val nOutput: Int, // output feature map number
  val eps: Double = 1e-5, // avoid divde zero
  val momentum: Double = 0.1, // momentum for weight update
  val affine: Boolean = true // affine operation on output or not
)(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  require(nOutput > 0)

  val nDim = 2
  val runningMean = Tensor[T](nOutput)
  val runningVar = Tensor[T](nOutput).fill(ev.fromType[Int](1))
  val saveMean = Tensor[T](nOutput)
  val saveStd = Tensor[T](nOutput).fill(ev.fromType[Int](1))

  val weight: Tensor[T] = if (affine) Tensor[T](nOutput) else null
  val bias: Tensor[T] = if (affine) Tensor[T](nOutput) else null

  val gradWeight: Tensor[T] = if (affine) Tensor[T](nOutput) else null
  val gradBias: Tensor[T] = if (affine) Tensor[T](nOutput) else null

  @transient
  private var results : Array[Future[_]] = null
  @transient
  // BatchNormalization has internal parameters (saveMean, saveStd)
  // that changes at every forward, so a standard gradcheck won't work with this module.
  // if you want to do a gradcheck, you will need to fix those variables, otherwise not fix.
  private var needFix: Boolean = false

  reset()

  override def reset(): Unit = {
    if (null != weight) {
      weight.apply1(_ => ev.fromType[Double](RNG.uniform(0, 1)))
    }

    if (null != bias) {
      bias.fill(ev.fromType[Int](0))
    }

    runningMean.zero()
    runningVar.fill(ev.fromType[Int](1))
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
    val featDim = if (input.dim() == nDim - 1) 1 else 2
    require(input.size(featDim) == runningMean.nElement(),
      s"got ${input.size(featDim)}-feature tensor, expected ${runningMean.nElement()}")
  }

  @inline
  private def makeBatch(input: Tensor[T]): Tensor[T] = {
    if (input.dim() == nDim - 1 && train == false) {
      input.addSingletonDimension()
    } else {
      input
    }
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    checkInputDim(input)

    output.resizeAs(input)
    saveMean.resizeAs(runningMean)
    saveStd.resizeAs(runningVar)

    val _input = makeBatch(input)
    val nInput = _input.size(2)
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
        updateOutputFloat(inputData, inputOffset, inputStride, outputData, outputOffset,
          outputStride, nInput, n, inputStride2)
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
    backward(input, gradOutput, ev.fromType[Int](1), gradInput, null, null)
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T], scale: Double): Unit = {
    backward(input, gradOutput, ev.fromType[Double](scale), null, gradWeight, gradBias)
  }

  override def backward(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    checkInputDim(input)
    checkInputDim(gradOutput)
    val before = System.nanoTime()
    val result = backward(input, gradOutput, ev.fromType[Int](1), gradInput, gradWeight, gradBias)
    backwardTime += System.nanoTime() - before
    result
  }

  def backward(input: Tensor[T], gradOutput: Tensor[T], scale: T = ev.fromType[Int](1),
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
                ev.toType[Double](scale), gradWeightData, gradWeightOffset, gradBiasData,
                gradBiasOffset)
            } else {
              backwardDouble(inputData, inputOffset, inputStride, inputStride2, gradOutputData,
                gradOutputOffset, gradOutputStride, gradOutputStride2,
                gradInputData, gradInputOffset, gradInputStride, gradInputStride2, nInput, n,
                ev.toType[Double](scale), null, 0, null, 0)
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
              null, 0, 0, 0, nInput, n, ev.toType[Double](scale), gradWeightData, gradWeightOffset,
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
            ev.toType[Double](scale), null, 0, null, 0)
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
                ev.toType[Float](scale), gradWeightData, gradWeightOffset, gradBiasData,
                gradBiasOffset)
            } else {
              backwardFloat(inputData, inputOffset, inputStride, inputStride2, gradOutputData,
                gradOutputOffset, gradOutputStride, gradOutputStride2,
                gradInputData, gradInputOffset, gradInputStride, gradInputStride2, nInput, n,
                ev.toType[Float](scale), null, 0, null, 0)
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
              null, 0, 0, 0, nInput, n, ev.toType[Float](scale), gradWeightData, gradWeightOffset,
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
            ev.toType[Float](scale), null, 0, null, 0)
        }
    }

    gradInput
  }

  private def backwardDouble(input: Array[Double], inputOffset: Int, inputStride: Int,
    inputStride2: Int, gradOutput: Array[Double], gradOutputOffset: Int, gradOutputStride: Int,
    gradOutputStride2: Int, gradInput: Array[Double], gradInputOffset: Int, gradInputStride: Int,
    gradInputStride2: Int, nInput: Int, n: Int, scale: Double, gradWeight: Array[Double],
    gradWeightOffset: Int, gradBias: Array[Double], gradBiasOffset: Int
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

        if (null != gradWeight) {
          gradWeight(_f - 1 + gradWeightOffset) += scale * dotp * invstd
        }

        if (null != gradBias) {
          gradBias(_f - 1 + gradBiasOffset) += scale * sum
        }
      })
      f += 1
    }
    Engine.model.sync(results)
  }

  private def backwardFloat(input: Array[Float], inputOffset: Int, inputStride: Int,
    inputStride2: Int, gradOutput: Array[Float], gradOutputOffset: Int, gradOutputStride: Int,
    gradOutputStride2: Int, gradInput: Array[Float], gradInputOffset: Int, gradInputStride: Int,
    gradInputStride2: Int, nInput: Int, n: Int, scale: Float, gradWeight: Array[Float],
    gradWeightOffset: Int, gradBias: Array[Float], gradBiasOffset: Int
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

        if (null != gradWeight) {
          gradWeight(_f - 1 + gradWeightOffset) += scale * dotp * invstd
        }

        if (null != gradBias) {
          gradBias(_f - 1 + gradBiasOffset) += scale * sum
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

object BatchNormalization {
  def apply[@specialized(Float, Double) T: ClassTag](
    nOutput: Int,
    eps: Double = 1e-5,
    momentum: Double = 0.1,
    affine: Boolean = true)(implicit ev: TensorNumeric[T]): BatchNormalization[T] = {
    new BatchNormalization[T](nOutput, eps, momentum, affine)
  }
}
