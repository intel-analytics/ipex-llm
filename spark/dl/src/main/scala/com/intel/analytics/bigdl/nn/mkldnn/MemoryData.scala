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

import com.intel.analytics.bigdl.mkl.MklDnn

sealed class MemoryData(val shape: Array[Int], val layout: Int)

case class HeapData(_shape: Array[Int], _layout: Int) extends MemoryData(_shape, _layout)

case class NativeData(_shape: Array[Int], _layout: Int) extends MemoryData(_shape, _layout)

private[mkldnn] object MemoryData {
  def isCompatible(actuals: Array[MemoryData], expects: Array[MemoryData]): Boolean = {
    if (actuals.length != expects.length) return false
    actuals.zip(expects).foreach { case (actual, expect) =>
      if (!isSizeCompatible(actual, expect)) return false
      actual match {
        case h: HeapData =>
          expect match {
            case hh: HeapData =>
              if (hh.layout != MklDnn.MemoryFormat.any && hh.layout != h.layout) return false
            case nn: NativeData => return false
            case _ => throw new UnsupportedOperationException("Not support such memory format")
          }
        case n: NativeData =>
          expect match {
            case hh: HeapData => return false
            case nn: NativeData =>
              if (nn.layout != MklDnn.MemoryFormat.any && nn.layout != n.layout) return false
            case _ => throw new UnsupportedOperationException("Not support such memory format")
          }
        case _ => throw new UnsupportedOperationException("Not support such memory format")
      }
    }
    return true
  }

  def isSizeCompatible(actual: MemoryData, expect: MemoryData): Boolean = {
    if (expect == null) return true
    if (actual == null) return false
    if (actual.shape.length != expect.shape.length) return false
    actual.shape.zip(expect.shape).foreach {case (a, e) => if (a != e) return false}
    return true
  }
}