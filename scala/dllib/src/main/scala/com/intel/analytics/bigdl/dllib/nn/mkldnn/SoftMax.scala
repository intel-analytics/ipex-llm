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

import com.intel.analytics.bigdl.mkl.{MklDnn, PropKind, Stream => DnnStream}
import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.tensor.{DenseType, Tensor}

import scala.collection.mutable.ArrayBuffer

class SoftMax() extends MklDnnLayer {
  private val nnSoftMax = nn.SoftMax[Float]()

  @transient private var updateOutputTensors: Array[Tensor[Float]] = _
  @transient private var updateOutputMemoryPrimitives: Array[Long] = _

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    phase match {
      case TrainingPhase =>
        _inputFormats = inputs.clone()
        _outputFormats = inputs.clone()
        (_inputFormats, _outputFormats)
      case InferencePhase =>
        val axis = inputs(0).shape.length match {
          case 1 => 0
          case 2 => 1
//          case 3 => 1 // TODO should support this?
          case 4 => 1
          case _ => throw new UnsupportedOperationException("1D, 2D, or 4D tensor expected")
        }

        _inputFormats = singleNativeData(inputs)
        val desc = MklDnn.SoftMaxForwardDescInit(PropKind.ForwardInference,
          inputFormats()(0).getMemoryDescription(), axis)
        val forwardPrimDesc = MklDnn.PrimitiveDescCreate(desc, runtime.engine, 0L)

        _outputFormats = Array(MemoryData.primitiveOutput(forwardPrimDesc))

        val srcs = Array(inputs(0).getPrimitive(runtime))
        val indexes = Array(0)
        val dsts = Array(_outputFormats(0).getPrimitive(runtime))

        val primitive = MklDnn.PrimitiveCreate2(forwardPrimDesc, srcs, indexes, srcs.length, dsts,
          dsts.length)

        updateOutputPrimitives = Array(primitive)
        updateOutputMemoryPrimitives = srcs ++ dsts

        output = initTensor(_outputFormats(0))

        (_inputFormats, _outputFormats)
      case _ => throw new UnsupportedOperationException
    }
  }

  override private[mkldnn] def initBwdPrimitives(grad: Array[MemoryData], phase: Phase) = {
    _gradInputFormats = grad.clone()
    _gradOutputFormats = grad.clone()
    (_gradInputFormats, _gradOutputFormats)
  }

  override def updateOutput(input: Activity): Activity = {
      if (this.isTraining()) {
        nnSoftMax.forward(input)
        output = nnSoftMax.output
      } else {
        if (updateOutputTensors == null) {
          val buffer = new ArrayBuffer[Tensor[Float]]()
          buffer.append(input.asInstanceOf[Tensor[Float]])
          buffer.append(output.asInstanceOf[Tensor[Float]])
          updateOutputTensors = buffer.toArray
        }

        input.toTensor[Float].getTensorType match {
          case DenseType => updateOutputTensors(0) = input.toTensor
          case _ =>
        }

        MklDnnOps.streamSubmit(runtime.stream, 1,
          updateOutputPrimitives,
          updateOutputPrimitives.length,
          updateOutputMemoryPrimitives, updateOutputTensors)
    }

    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    gradInput = nnSoftMax.backward(input, gradOutput)
    gradInput
  }
}

object SoftMax {
  def apply(): SoftMax = {
    new SoftMax()
  }
}
