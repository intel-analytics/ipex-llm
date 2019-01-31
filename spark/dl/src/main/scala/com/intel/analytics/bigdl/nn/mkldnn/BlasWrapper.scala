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

package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{MultiShape, Shape}
import spire.syntax.module

/**
 * wrap blas module to be dnn module,
 * and the module should have implemented "computeOutputShape" func.
 * @param module
 */
private[bigdl] class BlasWrapper(val module: AbstractModule[Activity, Activity, Float])
  extends MklDnnLayer {

  require(!module.isInstanceOf[MklDnnModule], "Only support wrapper blas layer to dnn layer")

  output = module.output
  gradInput = module.gradInput

  private def inferFormats(inputs: Array[MemoryData]): Int = {
    // reminder: here assume all shapes in inputs should be same
    inputs.foreach(in =>
      require(in.shape.length == 2 || in.shape.length == 4,
      s"only input shape dim 2 and 4 supported, but get ${in.shape.length}"))

    inputs(0).layout match {
      case Memory.Format.nhwc => Memory.Format.nhwc
      case Memory.Format.nc => Memory.Format.nc
      case _ => Memory.Format.nchw
    }
  }

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    // reminder: only support model having implemented computeOutputShape
    val inputShape = inputs.map(in => Shape(in.shape))
    val outputShape = if (inputShape.length == 1) {
      List(module.computeOutputShape(inputShape(0)))
    } else {
      // multi shape
      val out = module.computeOutputShape(MultiShape(inputShape.toList))
      if (out.isInstanceOf[MultiShape]) out.toMulti() else List(out)
    }
    val outDim = outputShape(0).toSingle().length
    require(outDim == 4 || outDim == 2,
      s"only output shape dim 2 and 4 supported, but get ${outDim}")

    val inputFormats = inferFormats(inputs)
    val outputFormats = if (outDim == 4) inputFormats else Memory.Format.nc

    val realInputs = inputShape.map(in => HeapData(in.toSingle().toArray, inputFormats))
    val realOutputs = outputShape.map(in => HeapData(in.toSingle().toArray, outputFormats))

    _inputFormats = realInputs.toArray
    _outputFormats = realOutputs.toArray

    (_inputFormats, _outputFormats)
  }

  override private[mkldnn] def initBwdPrimitives(grad: Array[MemoryData], phase: Phase) = {
    _gradOutputFormats = _outputFormats
    _gradInputFormats = _inputFormats
    (_outputFormats, _gradInputFormats)
  }

  override private[mkldnn] def initGradWPrimitives(grad: Array[MemoryData], phase: Phase) = {
    _gradOutputFormatsForWeight = _outputFormats
    _gradOutputFormatsForWeight
  }

  override def updateOutput(input: Activity): Activity = {
    output = module.forward(input)
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    gradInput = module.updateGradInput(input, gradOutput)
    gradInput
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    module.accGradParameters(input, gradOutput)
  }

  override def clearState() : this.type = {
    super.clearState()
    module.clearState()
    this
  }

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    module.parameters()
  }

  override def equals(obj: Any): Boolean = {
    if (!super.equals(obj) || !obj.isInstanceOf[BlasWrapper]) {
      return false
    }
    val other = obj.asInstanceOf[BlasWrapper]
    if (this.eq(other)) {
      return true
    }
    if (module != other) return false
    true
  }

  override def hashCode() : Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + module.hashCode()
    hash
  }

  override def training(): this.type = {
    train = true
    module.training()
    this
  }

  /**
   * Set the module to evaluate mode
   * @return
   */
  override def evaluate(): this.type = {
    train = false
    module.evaluate()
    this
  }

  override def release(): Unit = module.release()

}


private[bigdl] object BlasWrapper {
  def apply(module: AbstractModule[Activity, Activity, Float]): BlasWrapper =
    new BlasWrapper(module)
}
