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

import com.intel.analytics.bigdl.mkl.DataType
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

private[mkldnn] class ReorderManager() (implicit owner: MemoryOwner) {
  // (MemoryFormatId, TargetFormat) -> Reorder
  val reorders = mutable.HashMap[(Int, MemoryData), ReorderMemory]()
  // ReorderId -> RefCount
  val refCounts = mutable.HashMap[Int, Int]()
  val useCounts = mutable.HashMap[Int, Int]()

  private var runtime: MklDnnRuntime = _

  def register(from: MemoryData, to: MemoryData): Unit = {
    require(runtime != null, "Please call setRuntime first")
    val mId = System.identityHashCode(from)
    if (needReorder(from, to)) {
      if (reorders.contains((mId, to))) {
        refCounts(System.identityHashCode(reorders((mId, to)))) += 1
      } else {
        val reorder = ReorderMemory(to)
        reorder.setRuntime(runtime)
        reorder.initFwdPrimitives(Array(from), Phase.InferencePhase)
        reorders((mId, to)) = reorder
        val reorderId = System.identityHashCode(reorder)
        refCounts(reorderId) = 1
        useCounts(reorderId) = 0
      }
    }
  }

  def setRuntime(runtime: MklDnnRuntime): Unit = {
    this.runtime = runtime
  }

  def infer(from: Array[MemoryData], to: Array[MemoryData], output: Activity)
  : Activity = {
    if (from.length == 1) {
      require(output.isTensor, "output activity should be a tensor")
      inferTensor(from(0), to(0), output.asInstanceOf[Tensor[Float]])
    } else {
      require(output.toTable.length() == from.length,
        "output activity length doesn't match")
      val outputTable = T()
      var i = 0
      while(i < from.length) {
        outputTable(i + 1) = inferTensor(from(i), to(i), output.toTable(i + 1))
        i += 1
      }
      outputTable
    }
  }

  private def inferTensor(from: MemoryData, to : MemoryData, output: Tensor[Float])
  : Tensor[Float] = {
    val mId = System.identityHashCode(from)
    if (reorders.contains((mId, to))) {
      val reorder = reorders((mId, to))
      val reorderId = System.identityHashCode(reorder)
      val result = if (useCounts(reorderId) == 0) {
        reorder.forward(output).asInstanceOf[Tensor[Float]]
      } else {
        reorder.output.asInstanceOf[Tensor[Float]]
      }
      useCounts(reorderId) += 1
      if (useCounts(reorderId) == refCounts(reorderId)) {
        useCounts(reorderId) = 0
      }
      result
    } else {
      output
    }
  }

  private def needReorder(from: MemoryData, to: MemoryData): Boolean = {
    from match {
      case h: HeapData =>
        to match {
          case hh: HeapData => h.layout != hh.layout
          case nn: NativeData => true
          case _ => throw new UnsupportedOperationException("Not support such memory format")
        }
      case n: NativeData =>
        to match {
          case hh: HeapData => true
          case nn: NativeData =>
            // we will skip the S8 to U8 reorder
            val doNotReorderIt = n.layout == nn.layout && (
              n.dataType == nn.dataType || // the same data type
                (n.dataType == DataType.S8 && nn.dataType == DataType.U8) || // skip the u8 -> s8
                (n.dataType == DataType.U8 && nn.dataType == DataType.S8)) // skip the s8->u8

            !doNotReorderIt
          case _ => throw new UnsupportedOperationException("Not support such memory format")
        }
      case _ => throw new UnsupportedOperationException("Not support such memory format")
    }
  }

  def release(): Unit = { }
}
