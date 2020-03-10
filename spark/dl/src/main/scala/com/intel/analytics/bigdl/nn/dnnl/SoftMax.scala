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

import com.intel.analytics.bigdl.dnnl.{ArgType, DataType, Memory, PropKind}
import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{DenseType, Tensor}
import com.intel.analytics.bigdl.utils.Shape

import scala.collection.mutable


class SoftMax(val axis: Int = -1) extends MklDnnLayer {
  private val nnSoftMax = nn.SoftMax[Float]()
  @transient private var modelPhase: Phase = null

  private def initPhase(phase: Phase): Unit = {
    if (phase != null) return modelPhase = phase
    isTraining() match {
      case true =>
        modelPhase = TrainingPhase
      case false =>
        modelPhase = InferencePhase
    }
  }

  private def format(shape: Array[Int]): Int = {
    shape.length match {
      case 2 => Memory.FormatTag.nc
      case 4 => Memory.FormatTag.nchw
      case _ => throw new UnsupportedOperationException(s"${getName()} unsupported input shape")
    }
  }

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    initPhase(phase)
    modelPhase match {
      case TrainingPhase =>
        _inputFormats = inputs.map(x => HeapData(x.shape, format(x.shape)))
        _outputFormats = inputs.map(x => HeapData(x.shape, format(x.shape)))

        (_inputFormats, _outputFormats)
      case InferencePhase =>
        val defaultAxis = inputs(0).shape.length match {
          case 1 => 0
          case 2 => 1
          case 3 => 0
          case 4 => 1
          case _ => throw new UnsupportedOperationException("1D, 2D, 3D or 4D tensor expected")
        }

        _inputFormats = Array(NativeData(inputs(0).shape, inputs(0).layout, DataType.F32))

        val localInputFormat = if (inputs(0).shape.length == 3 &&
          inputs(0).layout == Memory.FormatTag.ntc) {
          // note: here, the format and the true memory layout is not consistent.
          // for ntc input, we should reshape the `shape` and make the format to tnc
          val shape = Array(inputs(0).shape(1), inputs(0).shape(0), inputs(0).shape(2))
          NativeData(shape, Memory.FormatTag.tnc)
        } else {
          _inputFormats(0)
        }

        val desc = DnnlMemory.SoftMaxForwardDescInit(PropKind.ForwardInference,
          localInputFormat.getMemoryDescriptor(), if (axis == -1) defaultAxis else axis)

        val forwardPrimDesc = DnnlMemory.PrimitiveDescCreate(desc, runtime.engine, 0L)

        _outputFormats = if (inputs(0).shape.length ==3 &&
          inputs(0).layout == Memory.FormatTag.ntc) {
          // because set the input format as tnc first, we should set the output to ntc.
          Array(NativeData(inputs(0).shape, Memory.FormatTag.ntc))
        } else {
          Array(MemoryData.primitiveOutput(forwardPrimDesc))
        }

        val primitive = DnnlMemory.PrimitiveCreate(forwardPrimDesc)
        updateOutputPrimitives = Array(primitive)
        fwdExecArgs = mutable.Map(
          ArgType.DNNL_ARG_SRC -> inputs(0).getMemoryObject(runtime),
          ArgType.DNNL_ARG_DST -> _outputFormats(0).getMemoryObject(runtime)
        )

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
        updateOutputTensors = mutable.Map(
          ArgType.DNNL_ARG_SRC -> input.asInstanceOf[Tensor[Float]],
          ArgType.DNNL_ARG_DST -> output.asInstanceOf[Tensor[Float]]
        )
      }
      input.toTensor[Float].getTensorType match {
        case DenseType =>
          updateOutputTensors(ArgType.DNNL_ARG_SRC) = input.toTensor
        case _ =>
      }
      MklDnnOps.streamSubmit(updateOutputPrimitives, runtime.stream,
        fwdExecArgs, updateOutputTensors)
    }

    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    gradInput = nnSoftMax.backward(input, gradOutput)
    gradInput
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    inputShape
  }
}

object SoftMax {
  def apply(axis: Int = -1)(implicit ev: TensorNumeric[Float]): SoftMax = {
    new SoftMax(axis)
  }
}
