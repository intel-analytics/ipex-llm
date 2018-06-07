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

sealed trait MemoryData {
  def shape: Array[Int]
  def layout: Int
  def setShape(shape: Array[Int]): Unit
  def setLayout(layout: Int): Unit
}

case class HeapData(private var _shape: Array[Int], private var _layout: Int) extends MemoryData {

  override def setShape(shape: Array[Int]): Unit = _shape = shape.clone()

  override def setLayout(layout: Int): Unit = _layout = layout

  override def shape: Array[Int] = _shape.clone()

  override def layout: Int = _layout
}

case class NativeData(private var _shape: Array[Int], private var _layout: Int) extends MemoryData {
  override def shape: Array[Int] = _shape.clone()

  override def layout: Int = _layout

  override def setShape(shape: Array[Int]): Unit = _shape = shape.clone()

  override def setLayout(layout: Int): Unit = _layout = layout
}

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