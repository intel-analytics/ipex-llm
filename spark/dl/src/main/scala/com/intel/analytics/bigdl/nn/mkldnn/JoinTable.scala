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

  override private[mkldnn] def inferShape(shapes: Array[Array[Int]]) = {
    require(shapes.length > 0, "at least one tensor")
    _inputFormats = new Array[MemoryData](shapes.length)
    _gradInputFormats = new Array[MemoryData](shapes.length)
    val result = shapes(0).clone()
    _inputFormats(0) = NativeData(shapes(0), Memory.Format.format_undef)
    _gradInputFormats(0) = NativeData(shapes(0), Memory.Format.format_undef)
    var i = 1
    while(i < shapes.length) {
      val curShape = shapes(i)
      require(result.length == curShape.length, "tensor dimension not match")
      var j = 0
      while(j < curShape.length) {
        if (j == dimension - 1) {
          result(j) += curShape(j)
        } else {
          require(result(j) == curShape(j), "tensor size not match")
        }
        j += 1
      }
      _inputFormats(i) = NativeData(curShape, Memory.Format.format_undef)
      _gradInputFormats(i) = NativeData(curShape, Memory.Format.format_undef)
      i += 1
    }
    _outputFormats = Array(NativeData(result, Memory.Format.format_undef))
    _gradOutputFormats = Array(NativeData(result, Memory.Format.format_undef))
    _gradOutputFormatsForWeight = Array(NativeData(result, Memory.Format.format_undef))
    Array(result)
  }

  override private[mkldnn] def initFwdPrimitives(runtime: MklDnnRuntime, phase: Phase) = {
    this.runtime = runtime
    outputFormats()(0).setLayout(inputFormats()(0).layout)
    val inputPrimitiveDescs = initMemPrimDescFromFormat(inputFormats())
    val inputPrimitives = initMemPrimFromPrimDesc(inputPrimitiveDescs)
    val outputMemDesc = initMemDescFromFormat(outputFormats())(0)
    val primDesc = MklDnn.ConcatPrimitiveDescCreate(outputMemDesc,
      inputPrimitiveDescs.length, dimension - 1, inputPrimitiveDescs)
    val outputPrimDesc = MklDnnOps.primitiveDescQueryPd(primDesc, Query.DstPd, 0)
    val outputPrimitives = initMemPrimFromPrimDesc(Array(outputPrimDesc))
    fwdMemPrims = inputPrimitives ++ outputPrimitives

    updateOutputPrimitives = Array(MklDnnOps.primitiveCreate2(primDesc, inputPrimitives,
      new Array[Int](inputPrimitiveDescs.length), inputPrimitiveDescs.length, outputPrimitives, 1))
  }

  override private[mkldnn] def initBwdPrimitives(runtime: MklDnnRuntime, phase: Phase) = {
  }
}