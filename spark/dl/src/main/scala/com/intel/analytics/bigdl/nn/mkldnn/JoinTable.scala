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

import com.intel.analytics.bigdl.mkl.{Memory, MklDnn, Query}

class JoinTable(val dimension: Int) extends MklDnnLayer {
  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    require(inputs.length > 0, "at least one tensor")
    _inputFormats = inputs

    val totalShape = inputs(0).shape.clone()
    val layout = inputs(0).layout
    var i = 1
    while(i < inputs.length) {
      val curShape = inputs(i).shape
      require(totalShape.length == inputs(i).layout, "layout not match")
      require(totalShape.length == curShape.length, "tensor dimension not match")
      require(inputs(i).isInstanceOf[NativeData], "memory should be native")
      var j = 0
      while(j < curShape.length) {
        if (j == dimension - 1) {
          totalShape(j) += curShape(j)
        } else {
          require(totalShape(j) == curShape(j), "tensor size not match")
        }
        j += 1
      }
      i += 1
    }
    _outputFormats = Array(NativeData(totalShape, layout))
    val primDesc = MklDnn.ConcatPrimitiveDescCreate(
      _outputFormats(0).getMemoryDescription(),
      inputs.length, dimension - 1, _inputFormats.map(_.getPrimitiveDescription(runtime)))

    updateOutputPrimitives = Array(MklDnnOps.primitiveCreate2(primDesc,
      _inputFormats.map(_.getPrimitive(runtime)),
      new Array[Int](inputs.length), inputs.length, _outputFormats.map(_.getPrimitive(runtime)), 1))
    output = initTensor(_outputFormats(0))
    (_inputFormats, _outputFormats)
  }

  override private[mkldnn] def initBwdPrimitives(grads: Array[MemoryData], phase: Phase) = {
    null
  }
}