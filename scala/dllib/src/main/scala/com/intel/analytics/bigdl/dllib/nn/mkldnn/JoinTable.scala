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

import com.intel.analytics.bigdl.mkl.{DataType, Memory, MklDnn, Query}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor

import scala.collection.mutable.ArrayBuffer

class JoinTable(val dimension: Int) extends MklDnnLayer {
  @transient
  private var memoryPrims: Array[Array[Long]] = _

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    require(inputs.length > 0, s"at least one tensor, but is ${inputs.length}")
    _inputFormats = nativeData(inputs)

    val totalShape = inputs(0).shape.clone()
    val layout = inputs(0).layout
    var i = 1
    while(i < inputs.length) {
      val curShape = inputs(i).shape
      require(totalShape.length == curShape.length, "tensor dimension not match")
      // require(inputs(i).isInstanceOf[NativeData], "memory should be native")
      var j = 0
      while(j < curShape.length) {
        if (j == dimension - 1) {
          totalShape(j) += curShape(j)
        } else {
          require(totalShape(j) == curShape(j), "tensor size not match")
        }
        j += 1
      }

      if (layout != inputs(i).layout || inputs(0).dataType != inputs(i).dataType) {
        _inputFormats(i) = NativeData(inputs(i).shape, layout, inputs(0).dataType)
      }
      i += 1
    }
    val primDesc = MklDnnMemory.ConcatPrimitiveDescCreate(
      MklDnnMemory.MemoryDescInit(totalShape.length, totalShape,
        inputs(0).dataType, Memory.Format.any),
      inputs.length, dimension - 1, _inputFormats.map(_.getPrimitiveDescription(runtime)))

    _outputFormats = Array(MemoryData.primitiveOutput(primDesc))
    updateOutputPrimitives = Array(MklDnnMemory.PrimitiveCreate2(primDesc,
      _inputFormats.map(_.getPrimitive(runtime)),
      new Array[Int](inputs.length), inputs.length,
      _outputFormats.map(_.getPrimitive(runtime)), 1)
    )
    output = initTensor(_outputFormats(0))
    (_inputFormats, _outputFormats)
  }

  override private[mkldnn] def initBwdPrimitives(grads: Array[MemoryData], phase: Phase) = {
    _gradOutputFormats = singleNativeData(grads)
    _gradOutputFormatsForWeight = _gradOutputFormats
    _gradInputFormats = _inputFormats.map(f => {
      NativeData(f.shape, f.layout)
    })
    val prims = new ArrayBuffer[Long]()
    val buffer = new ArrayBuffer[Array[Long]]()
    val offset = new Array[Int](_gradOutputFormats(0).shape.length)
    for(i <- 0 until _gradInputFormats.length) {
      val viewPD = MklDnnMemory.ViewPrimitiveDescCreate(
        _gradOutputFormats(0).getPrimitiveDescription(runtime), _gradInputFormats(i).shape, offset)
      val viewFormat = MemoryData.primitiveOutput(viewPD)
      val reorderPD = MklDnnMemory.ReorderPrimitiveDescCreate(
        viewFormat.getPrimitiveDescription(runtime),
        _gradInputFormats(i).getPrimitiveDescription(runtime))
      val reorderPrim = MklDnnMemory.PrimitiveCreate2(reorderPD,
        Array(viewFormat.getPrimitive(runtime)), Array(0), 1,
        Array(_gradInputFormats(i).getPrimitive(runtime)), 1)
      prims.append(reorderPrim)
      buffer.append(Array(viewFormat.getPrimitive(runtime),
        _gradInputFormats(i).getPrimitive(runtime)))
      offset(dimension - 1) += _gradInputFormats(i).shape(dimension - 1)
    }
    updateGradInputPrimitives = prims.toArray
    gradInput = initActivity(_gradInputFormats)
    memoryPrims = buffer.toArray

    (_gradOutputFormats, _gradInputFormats)
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    require(gradOutput.isTensor, "gradOutput should be tensor")
    require(gradInput.isTable, "gradInput should be table")
    val _gradOutput = gradOutput.asInstanceOf[Tensor[Float]]
    val _gradInput = gradInput.toTable
    val length = _gradInput.length()
    require(length == updateGradInputPrimitives.length, "gradOutput number not match")
    var i = 0
    while(i < length) {
      MklDnnOps.streamSubmit(runtime.stream, 1, Array(updateGradInputPrimitives(i)),
        1, memoryPrims(i), Array(_gradOutput, _gradInput[Tensor[Float]](i + 1)))
      i += 1
    }
    gradInput
  }

  private def isSameDataType(formats: Array[MemoryData]): Boolean = {
    formats.map(_.dataType).toSet.size == 1
  }

}

object JoinTable {
  def apply(dimension: Int): JoinTable = new JoinTable(dimension)
}
