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

import com.intel.analytics.bigdl.nn.abstractnn.{Initializable, TensorModule}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, T, Table}

import scala.concurrent.Future
import scala.reflect.ClassTag

/**
 * Applies a 1D convolution over an input sequence composed of nInputFrame frames..
 *   The input tensor in `forward(input)` is expected to be a 2D tensor
 *   (`nInputFrame` x `inputFrameSize`) or a 3D tensor
 *   (`nBatchFrame` x `nInputFrame` x `inputFrameSize`).
 *
 * @param inputFrameSize The input frame size expected in sequences given into `forward()`.
 * @param outputFrameSize The output frame size the convolution layer will produce.
 * @param kernelW The kernel width of the convolution
 * @param strideW The step of the convolution in the width dimension.
 * @param propagateBack Whether propagate gradient back, default is true.
 * @param wRegularizer instance of [[Regularizer]]
 *                    (eg. L1 or L2 regularization), applied to the input weights matrices.
 * @param bRegularizer instance of [[Regularizer]]
 *                    applied to the bias.
 * @param initWeight Initial weight
 * @param initBias Initial bias
 * @param initGradWeight Initial gradient weight
 * @param initGradBias Initial gradient bias
 * @tparam T The numeric type in the criterion, usually which are [[Float]] or [[Double]]
 */
class TemporalConvolution[T: ClassTag](
  val inputFrameSize: Int,
  val outputFrameSize: Int,
  val kernelW: Int,
  val strideW: Int = 1,
  val propagateBack: Boolean = true,
  var wRegularizer: Regularizer[T] = null,
  var bRegularizer: Regularizer[T] = null,
  val initWeight: Tensor[T] = null,
  val initBias: Tensor[T] = null,
  val initGradWeight: Tensor[T] = null,
  val initGradBias: Tensor[T] = null
)(implicit ev: TensorNumeric[T]) extends TensorModule[T]  with Initializable {

  val weight: Tensor[T] = if (initWeight != null) {
    initWeight
  } else {
    Tensor[T](outputFrameSize, inputFrameSize * kernelW)
  }

  val bias: Tensor[T] = if (initBias != null) {
    initBias
  } else {
    Tensor[T](outputFrameSize)
  }

  val gradWeight: Tensor[T] = if (initGradWeight != null) {
    initGradWeight
  } else {
    Tensor[T](outputFrameSize, inputFrameSize * kernelW)
  }

  val gradBias: Tensor[T] = if (initBias != null) {
    initGradBias
  } else {
    Tensor[T](outputFrameSize)
  }

  @transient protected var inputWindow: Tensor[T] = _
  @transient protected var outputWindow: Tensor[T] = _
  @transient protected var gradInputWindow: Tensor[T] = _
  @transient protected var gradOutputWindow: Tensor[T] = _

  {
    val stdv = 1.0 / math.sqrt(kernelW * inputFrameSize)
    val wInit: InitializationMethod = RandomUniform(-stdv, stdv)
    val bInit: InitializationMethod = RandomUniform(-stdv, stdv)
    setInitMethod(wInit, bInit)
  }

  @transient
  protected var results: Array[Future[Unit]] = _

  override def reset(): Unit = {
    if (initWeight == null) {
      weightInitMethod.init(weight, VariableFormat.OUT_IN)
    }
    if (initBias == null) {
      biasInitMethod.init(bias, VariableFormat.ONE_D)
    }
    zeroGradParameters()
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    // Require input of 2 dimensions or 3 dimensions
    // 2d input format: time x feature
    // 3d input format: batch x time x feature
    require(input.dim() == 2 || input.dim() == 3,
      "TemporalConvolution: 2D or 3D(batch mode) tensor expected for input, " +
        s"but got ${input.dim()}")
    // Require input to be contiguous
    require(input.isContiguous())

    var dimSeq = 1
    var dimFeat = 2

    if (input.dim() == 3) {
      dimSeq = 2
      dimFeat = 3
    }

    val nInputFrame = input.size(dimSeq)
    var nOutputFrame = (nInputFrame - kernelW) / strideW + 1

    if (inputWindow == null) inputWindow = Tensor[T]()
    if (outputWindow == null) outputWindow = Tensor[T]()

    // Shape check on input with inputFrameSize and kernelW
    require(input.size(dimFeat) == inputFrameSize, "Invalid input frame size. Got: " +
      s"${input.size(dimFeat)}, Expected: $inputFrameSize")
    require(nOutputFrame >= 1, "Input sequence smaller than kernel size. Got: " +
      s"$nInputFrame, Expected: $kernelW")

    val weightT = weight.transpose(1, 2)

    if (input.dim() == 2) {
      output.resize(nOutputFrame, outputFrameSize)
      // Add bias first
      var j = 1
      while (j <= nOutputFrame) {
        outputWindow = output.select(dimSeq, j)
        outputWindow.copy(bias)
        j += 1
      }
      // Add the convolution part
      j = 0
      while (nOutputFrame > 0) {
        val outputFrameStride = (kernelW - 1) / strideW + 1
        val inputFrameStride = outputFrameStride * strideW
        val nFrame = (nInputFrame - j * strideW - kernelW) / inputFrameStride + 1
        nOutputFrame -= nFrame

        inputWindow.set(input.storage(), input.storageOffset() + j * strideW * input.size(dimFeat),
          Array(nFrame, kernelW * input.size(dimFeat)),
          Array(inputFrameStride * input.size(dimFeat), 1))
        outputWindow.set(output.storage(), output.storageOffset() + j * output.size(dimFeat),
          Array(nFrame, output.size(dimFeat)),
          Array(outputFrameStride * output.size(dimFeat), 1))

        outputWindow.addmm(ev.fromType[Int](1), outputWindow,
          ev.fromType[Int](1), inputWindow, weightT)
        j += 1
      }
    } else {
      val batchSize = input.size(1)
      output.resize(batchSize, nOutputFrame, outputFrameSize)
      if (results == null || results.length != batchSize) {
        results = new Array[Future[Unit]](batchSize)
      }
      var i = 0
      while (i < batchSize) {
        results(i) = Engine.model.invoke(() => {
          val inputSample = input.select(1, i + 1)
          val outputSample = output.select(1, i + 1)
          var nOutputSampleFrame = nOutputFrame
          // Add bias first
          var j = 1
          while (j <= nOutputFrame) {
            outputWindow = outputSample.select(dimSeq - 1, j)
            outputWindow.copy(bias)
            j += 1
          }
          // Add the convolution part
          j = 0
          while (nOutputSampleFrame > 0) {
            val outputFrameStride = (kernelW - 1) / strideW + 1
            val inputFrameStride = outputFrameStride * strideW
            val nFrame = (nInputFrame - j * strideW - kernelW) / inputFrameStride + 1
            nOutputSampleFrame -= nFrame

            inputWindow.set(inputSample.storage(), inputSample.storageOffset() +
              j * strideW * inputSample.size(dimFeat - 1),
              Array(nFrame, kernelW * inputSample.size(dimFeat - 1)),
              Array(inputFrameStride * inputSample.size(dimFeat - 1), 1))
            outputWindow.set(outputSample.storage(), outputSample.storageOffset() +
              j * outputSample.size(dimFeat - 1),
              Array(nFrame, outputSample.size(dimFeat - 1)),
              Array(outputFrameStride * outputSample.size(dimFeat - 1), 1))

            outputWindow.addmm(ev.fromType[Int](1), outputWindow,
              ev.fromType[Int](1), inputWindow, weightT)
            j += 1
          }
        })
        i += 1
      }
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    // Require input of 2 dimensions or 3 dimensions
    // 2d input format: time x feature
    // 3d input format: batch x time x feature
    require(input.dim() == 2 || input.dim() == 3,
      "TemporalConvolution: 2D or 3D(batch mode) tensor expected for input, " +
        s"but got ${input.dim()}")
    // Require input to be contiguous
    require(input.isContiguous())

    val dimSeq = if (input.dim() == 2) 1 else 2
    val dimFeat = if (input.dim() == 2) 2 else 3
    val nInputFrame = input.size(dimSeq)
    var nOutputFrame = (nInputFrame - kernelW) / strideW + 1

    if (gradInputWindow == null) gradInputWindow = Tensor[T]()
    if (gradOutputWindow == null) gradOutputWindow = Tensor[T]()

    // Shape check on input with inputFrameSize and kernelW
    require(input.size(dimFeat) == inputFrameSize, "Invalid input frame size. Got: " +
      s"${input.size(dimFeat)}, Expected: $inputFrameSize")
    require(nOutputFrame >= 1, "Input sequence smaller than kernel size. Got: " +
      s"$nInputFrame, Expected: $kernelW")

    gradInput.resizeAs(input)
    gradInput.zero()

    if (gradOutput.dim() == 2) {
      var i = 0
      while (nOutputFrame > 0) {
        val outputFrameStride = (kernelW - 1) / strideW + 1
        val inputFrameStride = outputFrameStride * strideW
        val nFrame = (nInputFrame - i * strideW - kernelW) / inputFrameStride + 1
        nOutputFrame -= nFrame

        gradOutputWindow.set(gradOutput.storage(), gradOutput.storageOffset() +
          i * gradOutput.size(dimFeat), Array(nFrame, gradOutput.size(dimFeat)),
          Array(outputFrameStride * gradOutput.size(dimFeat), 1))
        gradInputWindow.set(gradInput.storage(), gradInput.storageOffset() +
          i * strideW * gradInput.size(dimFeat), Array(nFrame, kernelW * gradInput.size(dimFeat)),
          Array(inputFrameStride * gradInput.size(dimFeat), 1))

        gradInputWindow.addmm(ev.fromType[Int](1), gradInputWindow,
          ev.fromType[Int](1), gradOutputWindow, weight)
        i += 1
      }
    } else {
      val batchSize = input.size(1)
      var gradOutputSample = Tensor[T]()
      var gradInputSample = Tensor[T]()
      var i = 0
      while (i < batchSize) {
        results(i) = Engine.model.invoke(() => {
          gradInputSample = gradInput.select(1, i + 1)
          gradOutputSample = gradOutput.select(1, i + 1)
          var nOutputSampleFrame = nOutputFrame
          var j = 0
          while (nOutputSampleFrame > 0) {
            val outputFrameStride = (kernelW - 1) / strideW + 1
            val inputFrameStride = outputFrameStride * strideW
            val nFrame = (nInputFrame - j * strideW - kernelW) / inputFrameStride + 1
            nOutputSampleFrame -= nFrame

            gradOutputWindow.set(gradOutputSample.storage(), gradOutputSample.storageOffset() +
              j * gradOutputSample.size(dimFeat - 1),
              Array(nFrame, gradOutputSample.size(dimFeat - 1)),
              Array(outputFrameStride * gradOutputSample.size(dimFeat - 1), 1))
            gradInputWindow.set(gradInputSample.storage(), gradInputSample.storageOffset() +
              j * strideW * gradInputSample.size(dimFeat - 1),
              Array(nFrame, kernelW * gradInputSample.size(dimFeat - 1)),
              Array(inputFrameStride * gradInputSample.size(dimFeat - 1), 1))

            gradInputWindow.addmm(ev.fromType[Int](1), gradInputWindow,
              ev.fromType[Int](1), gradOutputWindow, weight)
            j += 1
          }
        })
        i += 1
      }
    }
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    // Require input of 2 dimensions or 3 dimensions
    require(input.nDimension() == 2 || input.nDimension() == 3,
      "Only support 2D or 3D input, " +
        s"input ${input.nDimension()}")
    // Require input to be contiguous
    require(gradOutput.isContiguous())

    val dimSeq = if (input.dim() == 2) 1 else 2
    val dimFeat = if (input.dim() == 2) 2 else 3
    val nInputFrame = input.size(dimSeq)
    var nOutputFrame = (nInputFrame - kernelW) / strideW + 1

    if (gradOutputWindow == null) gradOutputWindow = Tensor[T]()
    if (inputWindow == null) inputWindow = Tensor[T]()

    if (input.nDimension() == 2) {
      var j = 0
      while (j < nOutputFrame) {
        gradOutputWindow.set(gradOutput.select(1, j + 1))
        gradBias.add(gradBias, ev.fromType[Double](scaleB), gradOutputWindow)
        j += 1
      }
      j = 0
      while (nOutputFrame > 0) {
        val outputFrameStride = (kernelW - 1) / strideW + 1
        val inputFrameStride = outputFrameStride * strideW
        val nFrame = (nInputFrame - j * strideW - kernelW) / inputFrameStride + 1
        nOutputFrame -= nFrame

        inputWindow.set(input.storage(), input.storageOffset() + j * strideW * input.size(dimFeat),
          Array(nFrame, kernelW * input.size(dimFeat)),
          Array(inputFrameStride * input.size(dimFeat), 1))
        gradOutputWindow.set(gradOutput.storage(), gradOutput.storageOffset() +
          j * gradOutput.size(dimFeat), Array(nFrame, gradOutput.size(dimFeat)),
          Array(outputFrameStride * gradOutput.size(dimFeat), 1))

        val gradOutputWindowT = gradOutputWindow.transpose(1, 2)
        gradWeight.addmm(ev.fromType[Int](1), gradWeight, ev.fromType[Double](scaleW),
          gradOutputWindowT, inputWindow)
        j += 1
      }
    } else {
      val batchSize = input.size(1)
      var gradOutputSample = Tensor[T]()
      var inputSample = Tensor[T]()
      var i = 0
      while (i < batchSize) {
        results(i) = Engine.model.invoke(() => {
          gradOutputSample = gradOutput.select(1, i + 1)
          inputSample = input.select(1, i + 1)
          var nOutputSampleFrame = nOutputFrame
          var j = 0
          while (j < nOutputFrame) {
            gradOutputWindow.set(gradOutputSample.select(1, j + 1))
            gradBias.add(gradBias, ev.fromType[Double](scaleB), gradOutputWindow)
            j += 1
          }
          j = 0
          while (nOutputSampleFrame > 0) {
            val outputFrameStride = (kernelW - 1) / strideW + 1
            val inputFrameStride = outputFrameStride * strideW
            val nFrame = (nInputFrame - j * strideW - kernelW) / inputFrameStride + 1
            nOutputSampleFrame -= nFrame

            inputWindow.set(inputSample.storage(), inputSample.storageOffset() +
              j * strideW * inputSample.size(dimFeat - 1),
              Array(nFrame, kernelW * inputSample.size(dimFeat - 1)),
              Array(inputFrameStride * inputSample.size(dimFeat - 1), 1))
            gradOutputWindow.set(gradOutputSample.storage(), gradOutputSample.storageOffset() +
              j * gradOutputSample.size(dimFeat - 1),
              Array(nFrame, gradOutputSample.size(dimFeat - 1)),
              Array(outputFrameStride * gradOutputSample.size(dimFeat - 1), 1))

            val gradOutputWindowT = gradOutputWindow.transpose(1, 2)
            gradWeight.addmm(ev.fromType[Int](1), gradWeight, ev.fromType[Double](scaleW),
              gradOutputWindowT, inputWindow)
            j += 1
          }
        })
        i += 1
      }
    }

    if (null != wRegularizer) {
      wRegularizer.accRegularization(weight, gradWeight, scaleW)
    }
    if (null != bRegularizer) {
      bRegularizer.accRegularization(bias, gradBias, scaleB)
    }
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
  }

  override def equals(obj: Any): Boolean = {
    if (!super.equals(obj)) {
      return false
    }
    if (!obj.isInstanceOf[TemporalConvolution[T]]) {
      return false
    }
    val other = obj.asInstanceOf[TemporalConvolution[T]]
    if (this.eq(other)) {
      return true
    }

    inputFrameSize == other.inputFrameSize &&
      outputFrameSize == other.outputFrameSize &&
      kernelW == other.kernelW &&
      strideW == other.strideW &&
      propagateBack == other.propagateBack &&
      weight == other.weight &&
      bias == other.bias &&
      gradWeight == other.gradWeight &&
      gradBias == other.gradBias
  }

  override def hashCode(): Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + inputFrameSize.hashCode()
    hash = hash * seed + outputFrameSize.hashCode()
    hash = hash * seed + kernelW.hashCode()
    hash = hash * seed + strideW.hashCode()
    hash = hash * seed + weight.hashCode()
    hash = hash * seed + bias.hashCode()
    hash = hash * seed + gradWeight.hashCode()
    hash = hash * seed + gradBias.hashCode()

    hash
  }

  override def clearState() : this.type = {
    super.clearState()
    this
  }

  override def toString(): String = {
    s"nn.TemporalConvolution($inputFrameSize -> $outputFrameSize, $kernelW x $strideW)"
  }
}

object TemporalConvolution {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputFrameSize: Int,
    outputFrameSize: Int,
    kernelW: Int,
    strideW: Int = 1,
    propagateBack: Boolean = true,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    initWeight: Tensor[T] = null,
    initBias: Tensor[T] = null,
    initGradWeight: Tensor[T] = null,
    initGradBias: Tensor[T] = null
  )(implicit ev: TensorNumeric[T]): TemporalConvolution[T] = {
    new TemporalConvolution[T](inputFrameSize, outputFrameSize, kernelW,
      strideW, propagateBack,
      wRegularizer, bRegularizer, initWeight, initBias, initGradWeight, initGradBias)
  }
}
