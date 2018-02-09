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

/*
 *  The `LocallyConnected1D` layer works similarly to
 *  the `TemporalConvolution` layer, except that weights are unshared,
 *  that is, a different set of filters is applied at each different patch
 *  of the input.
 * The input tensor in `forward(input)` is expected to be a 2D tensor
 * (`nInputFrame` x `inputFrameSize`) or a 3D tensor
 * (`nBatchFrame` x `nInputFrame` x `inputFrameSize`).
 *
 * @param nInputFrame     the input frame channel
 * @param inputFrameSize  The input frame size expected in sequences given into `forward()`.
 * @param outputFrameSize The output frame size the convolution layer will produce.
 * @param kernelW         The kernel width of the convolution
 * @param strideW         The step of the convolution in the width dimension.
 * @param propagateBack   Whether propagate gradient back, default is true.
 * @param wRegularizer    instance of [[Regularizer]]
 *                        (eg. L1 or L2 regularization), applied to the input weights matrices.
 * @param bRegularizer    instance of [[Regularizer]]
 *                        applied to the bias.
 * @param initWeight      Initial weight
 * @param initBias        Initial bias
 * @param initGradWeight  Initial gradient weight
 * @param initGradBias    Initial gradient bias
 * @tparam T The numeric type in the criterion, usually which are [[Float]] or [[Double]]
 */
class LocallyConnected1D[T: ClassTag](val nInputFrame: Int,
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
                                     )(implicit ev: TensorNumeric[T])
                                        extends TensorModule[T] with Initializable {

  val nOutputFrame = (nInputFrame - kernelW) / strideW + 1

  val weight: Tensor[T] = if (initWeight != null) {
    initWeight
  } else {
    Tensor[T](nOutputFrame, outputFrameSize, inputFrameSize * kernelW)
  }

  val bias: Tensor[T] = if (initBias != null) {
    initBias
  } else {
    Tensor[T](nOutputFrame, outputFrameSize)
  }

  val gradWeight: Tensor[T] = if (initGradWeight != null) {
    initGradWeight
  } else {
    Tensor[T](nOutputFrame, outputFrameSize, inputFrameSize * kernelW)
  }

  val gradBias: Tensor[T] = if (initGradBias != null) {
    initGradBias
  } else {
    Tensor[T](nOutputFrame, outputFrameSize)
  }

  @transient protected var inputWindow: Tensor[T] = _
  @transient protected var outputWindow: Tensor[T] = _
  @transient protected var weightWindow: Tensor[T] = _
  @transient protected var biasWindow: Tensor[T] = _

  @transient protected var gradInputWindow: Tensor[T] = _
  @transient protected var gradOutputWindow: Tensor[T] = _
  @transient protected var gradWeightWindow: Tensor[T] = _

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

  def reshapeInput(input: Tensor[T]): Tensor[T] = {
    if (input.dim() == 2) {
      input.reshape(Array(1, input.size(1), input.size(2)))
    } else {
      input
    }
  }

  def reshapeOutput(input: Tensor[T], output: Tensor[T]): Tensor[T] = {
    if (input.dim() == 2) {
      output.reshape(Array(output.size(2), output.size(3)))
    } else {
      output
    }
  }

  override def updateOutput(_input: Tensor[T]): Tensor[T] = {
    // Require input of 2 dimensions or 3 dimensions
    // 2d input format: time x feature
    // 3d input format: batch x time x feature
    require(_input.dim() == 2 || _input.dim() == 3,
      "LocallyConvolution1D: 2D or 3D(batch mode) tensor expected for input, " +
        s"but got ${_input.dim()}")
    // Require input to be contiguous
    require(_input.isContiguous())

    val input = reshapeInput(_input)

    var dimSeq = input.dim() - 1 // 1
    var dimFeat = dimSeq + 1 // 2

    val nInputFrame = input.size(dimSeq) // 10
    var nOutputFrame = (nInputFrame - kernelW) / strideW + 1 // (10 -3)/1 +1 = 8

    if (inputWindow == null) inputWindow = Tensor[T]()
    if (outputWindow == null) outputWindow = Tensor[T]()
    if (weightWindow == null) weightWindow = Tensor[T]()
    if (biasWindow == null) biasWindow = Tensor[T]()

    // Shape check on input with inputFrameSize and kernelW
    require(input.size(dimFeat) == inputFrameSize, "Invalid input frame size. Got: " +
      s"${input.size(dimFeat)}, Expected: $inputFrameSize")
    require(nOutputFrame >= 1, "Input sequence smaller than kernel size. Got: " +
      s"$nInputFrame, Expected: $kernelW")

    val batchSize = input.size(1)
    val pageSize = weight.size(2) * weight.size(3)

    output.resize(batchSize, nOutputFrame, outputFrameSize)

    if (results == null || results.length != batchSize) {
      results = new Array[Future[Unit]](batchSize)
    }

    var i = 0
    while (i < batchSize) {

      results(i) = Engine.model.invoke(() => {

        val inputSample = input.select(1, i + 1)
        val outputSample = output.select(1, i + 1)
        // Add bias first

        var j = 1
        while (j < nOutputFrame) {
          biasWindow = bias.select(1, j)
          outputWindow = outputSample.select(dimSeq - 1, j) // setup up bias for each ouputframe
          outputWindow.copy(biasWindow)
          j += 1
        }

        // Add the convolution part
        j = 0
        while (j < nOutputFrame) {
          inputWindow.set(inputSample.storage(), inputSample.storageOffset() +
            j * strideW * input.size(dimFeat),
            Array(1, kernelW * input.size(dimFeat)),
            Array(1, 1))

          outputWindow.set(outputSample.storage(), outputSample.storageOffset() +
            j * output.size(dimFeat),
            Array(1, output.size(dimFeat)),
            Array(1, 1))

          val weightT = weightWindow.set(weight.storage(), weight.storageOffset() +
            j * pageSize,
            Array(output.size(dimFeat), kernelW * input.size(dimFeat)),
            Array(kernelW * input.size(dimFeat), 1)
          ).transpose(1, 2)

          outputWindow.addmm(ev.fromType[Int](1), outputWindow,
            ev.fromType[Int](1), inputWindow, weightT)

          j += 1
        }
      })
      i += 1
    }

    output = reshapeOutput(_input, output)
    output
  }

  override def updateGradInput(_input: Tensor[T], _gradOutput: Tensor[T]): Tensor[T] = {

    // Require input of 2 dimensions or 3 dimensions
    // 2d input format: time x feature
    // 3d input format: batch x time x feature
    require(_input.dim() == 2 || _input.dim() == 3,
      "TemporalConvolution: 2D or 3D(batch mode) tensor expected for input, " +
        s"but got ${_input.dim()}")
    // Require input to be contiguous
    require(_input.isContiguous())

    val input = reshapeInput(_input)
    val gradOutput = reshapeInput(_gradOutput)

    val dimSeq = if (input.dim() == 2) 1 else 2
    val dimFeat = if (input.dim() == 2) 2 else 3
    val nInputFrame = input.size(dimSeq)
    var nOutputFrame = (nInputFrame - kernelW) / strideW + 1

    if (gradInputWindow == null) gradInputWindow = Tensor[T]()
    if (gradOutputWindow == null) gradOutputWindow = Tensor[T]()
    if (weightWindow == null) weightWindow = Tensor[T]()

    // Shape check on input with inputFrameSize and kernelW
    require(input.size(dimFeat) == inputFrameSize, "Invalid input frame size. Got: " +
      s"${input.size(dimFeat)}, Expected: $inputFrameSize")
    require(nOutputFrame >= 1, "Input sequence smaller than kernel size. Got: " +
      s"$nInputFrame, Expected: $kernelW")

    gradInput.resizeAs(input)
    gradInput.zero()

    val batchSize = input.size(1)
    val pageSize = weight.size(2) * weight.size(3)

    var gradOutputSample = Tensor[T]()
    var gradInputSample = Tensor[T]()

    var i = 0
    while (i < batchSize) {

      results(i) = Engine.model.invoke(() => {

        gradInputSample = gradInput.select(1, i + 1)
        gradOutputSample = gradOutput.select(1, i + 1)

        var j = 0
        while (j < nOutputFrame) {

          gradOutputWindow.set(gradOutputSample.storage(),
            gradOutputSample.storageOffset() + j * gradOutput.size(dimFeat),
            Array(1, gradOutput.size(dimFeat)),
            Array(1, 1))

          gradInputWindow.set(gradInputSample.storage(),
            gradInputSample.storageOffset() + j * strideW * gradInput.size(dimFeat),
            Array(1, kernelW * gradInput.size(dimFeat)),
            Array(1, 1))

          weightWindow.set(weight.storage(), weight.storageOffset() + j * pageSize,
            Array(output.size(dimFeat), kernelW * input.size(dimFeat)),
            Array(kernelW * input.size(dimFeat), 1)
          )

          gradInputWindow.addmm(ev.fromType[Int](1), gradInputWindow,
            ev.fromType[Int](1), gradOutputWindow, weightWindow)
          j += 1
        }
      })
      i += 1
    }

    gradInput = reshapeOutput(_gradOutput, gradInput)
    gradInput
  }

  override def accGradParameters(_input: Tensor[T], _gradOutput: Tensor[T]): Unit = {

    // Require input of 2 dimensions or 3 dimensions
    require(_input.nDimension() == 2 || _input.nDimension() == 3,
      "Only support 2D or 3D input, " +
        s"input ${_input.nDimension()}")
    // Require input to be contiguous
    require(_gradOutput.isContiguous())

    val input = reshapeInput(_input)
    val gradOutput = reshapeInput(_gradOutput)

    val dimSeq = if (input.dim() == 2) 1 else 2
    val dimFeat = if (input.dim() == 2) 2 else 3
    val nInputFrame = input.size(dimSeq)
    var nOutputFrame = (nInputFrame - kernelW) / strideW + 1

    if (gradOutputWindow == null) gradOutputWindow = Tensor[T]()
    if (inputWindow == null) inputWindow = Tensor[T]()
    if (gradWeightWindow == null) gradWeightWindow = Tensor[T]()
    if (biasWindow == null) biasWindow = Tensor[T]()

    val batchSize = input.size(1)

    var gradOutputSample = Tensor[T]()
    var inputSample = Tensor[T]()

    var i = 0
    while (i < batchSize) {
      results(i) = Engine.model.invoke(() => {
        gradOutputSample = gradOutput.select(1, i + 1)
        inputSample = input.select(1, i + 1)

        var j = 0
        while (j < nOutputFrame) {
          biasWindow.set(gradBias.storage(),
            gradBias.storageOffset() + j * gradOutput.size(dimFeat),
            Array(1, gradOutput.size(dimFeat)),
            Array(1, 1))
          gradOutputWindow.set(gradOutputSample.select(1, j + 1))
          biasWindow.add(biasWindow, ev.fromType[Double](scaleB), gradOutputWindow)
          j += 1
        }

        j = 0
        while (j < nOutputFrame) {
          inputWindow.set(inputSample.storage(), inputSample.storageOffset() +
            j * strideW * input.size(dimFeat),
            Array(1, kernelW * input.size(dimFeat)),
            Array(1, 1))

          gradOutputWindow.set(gradOutputSample.storage(),
            gradOutputSample.storageOffset() + j * gradOutput.size(dimFeat),
            Array(1, gradOutput.size(dimFeat)),
            Array(1, 1))

          val gradOutputWindowT = gradOutputWindow.transpose(1, 2)

          val pageSize = weight.size(2) * weight.size(3)
          gradWeightWindow.set(gradWeight.storage(), gradWeight.storageOffset() +
            j * pageSize,
            Array(gradOutput.size(dimFeat), kernelW * input.size(dimFeat)),
            Array(kernelW * input.size(dimFeat), 1))

          gradWeightWindow.addmm(ev.fromType[Int](1), gradWeightWindow,
            ev.fromType[Double](scaleW), gradOutputWindowT, inputWindow)
          j += 1
        }
      })

      i += 1
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

  override def clearState(): this.type = {
    super.clearState()
    this
  }

  override def toString(): String = {
    s"nn.TemporalConvolution($inputFrameSize -> $outputFrameSize, $kernelW x $strideW)"
  }
}

object LocallyConnected1D {
  def apply[@specialized(Float, Double) T: ClassTag](
                                                      nInputFrame: Int,
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
                                                    )(implicit ev: TensorNumeric[T]):
                                                      LocallyConnected1D[T] = {
    new LocallyConnected1D[T](nInputFrame, inputFrameSize, outputFrameSize, kernelW,
      strideW, propagateBack,
      wRegularizer, bRegularizer, initWeight, initBias, initGradWeight, initGradBias)
  }
}
