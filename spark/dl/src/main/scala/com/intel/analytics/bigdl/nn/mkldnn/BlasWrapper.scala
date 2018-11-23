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
import com.intel.analytics.bigdl.utils.Shape
import spire.syntax.module

// only support wrap bigdl to support nchw or nc
private[bigdl] class BlasWrapper(val module: TensorModule[Float]) extends MklDnnLayer {

  output = Tensor[Float]()
  gradInput = Tensor[Float]()

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    // todo: only support tensor model and implement computeOutputShape
    val inputShape = if (inputs(0).layout == Memory.Format.nhwc) {
      val s = inputs(0).shape
      // from nhwc -> nchw
      Array(s(0), s(3), s(1), s(2))
    } else {
      inputs(0).shape
    }
    val outputShape = module.computeOutputShape(Shape(inputShape)).toSingle().toArray

    require(inputShape.length == 2 || inputShape.length == 4,
      s"just support input shape dim is 2 or 4, but get ${inputShape.length}")
    require(outputShape.length == 2 || outputShape.length == 4,
      s"just support output shape dim is 2 or 4, but get ${outputShape.length}")

    val realInputs = if (inputShape.length == 4) {
      HeapData(inputShape, Memory.Format.nchw)
    } else {
      HeapData(inputShape, Memory.Format.nc)
    }

    val realOutputs = if (outputShape.length == 4) {
      HeapData(outputShape, Memory.Format.nchw)
    } else {
      HeapData(outputShape, Memory.Format.nc)
    }

    _inputFormats = Array(realInputs)
    _outputFormats = Array(realOutputs)

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
    output = module.forward(input.toTensor[Float])
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    gradInput = module.updateGradInput(input.toTensor[Float], gradOutput.toTensor[Float])
    gradInput
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    module.accGradParameters(input.toTensor[Float], gradOutput.toTensor[Float])
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
}


private[bigdl] object BlasWrapper {
  def apply(module: TensorModule[Float]): BlasWrapper = new BlasWrapper(module)
}