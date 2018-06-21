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

import com.intel.analytics.bigdl.mkl.Memory

sealed trait MemoryData {
  def shape: Array[Int]
  def layout: Int
  def setShape(shape: Array[Int]): Unit
  def setLayout(layout: Int): Unit

  def isLayoutFixed(): Boolean = {
    layout != Memory.Format.format_undef && layout != Memory.Format.any
  }

  def cloneFormat(): MemoryData
}

case class HeapData(private var _shape: Array[Int], private var _layout: Int) extends MemoryData {

  override def setShape(shape: Array[Int]): Unit = _shape = shape.clone()

  override def setLayout(layout: Int): Unit = _layout = layout

  override def shape: Array[Int] = _shape.clone()

  override def layout: Int = _layout

  override def hashCode(): Int = {
    val seed = 37
    var hash = 1
    hash = hash * seed + this.layout
    var d = 0
    while (d < this.shape.length) {
      hash = hash * seed + this.shape(d)
      d += 1
    }

    hash
  }

  override def equals(obj: scala.Any): Boolean = {
    if (obj == null) {
      return false
    }
    if (!obj.isInstanceOf[HeapData]) {
      return false
    }
    val other = obj.asInstanceOf[HeapData]
    if (this.eq(other)) {
      return true
    }
    if (this.layout != other.layout) {
      return false
    }
    if (this.shape == null && other.shape == null) {
      return true
    }
    if (this.shape != null && other.shape != null) {
      if (this.shape.length != other.shape.length) return false
      var i = 0
      while(i < this.shape.length) {
        if (this.shape(i) != other.shape(i)) return false
        i += 1
      }
      return true
    } else {
      return false
    }
  }

  override def toString: String = {
    s"HeapData([${shape.mkString("x")}], ${layout})"
  }

  override def cloneFormat(): MemoryData = new HeapData(_shape, _layout)
}

case class NativeData(private var _shape: Array[Int], private var _layout: Int) extends MemoryData {
  override def shape: Array[Int] = _shape.clone()

  override def layout: Int = _layout

  override def setShape(shape: Array[Int]): Unit = _shape = shape.clone()

  override def setLayout(layout: Int): Unit = _layout = layout

  override def hashCode(): Int = {
    val seed = 41
    var hash = 1
    hash = hash * seed + this.layout
    var d = 0
    while (d < this.shape.length) {
      hash = hash * seed + this.shape(d)
      d += 1
    }

    hash
  }

  override def equals(obj: scala.Any): Boolean = {
    if (obj == null) {
      return false
    }
    if (!obj.isInstanceOf[NativeData]) {
      return false
    }
    val other = obj.asInstanceOf[NativeData]
    if (this.eq(other)) {
      return true
    }
    if (this.layout != other.layout) {
      return false
    }
    if (this.shape == null && other.shape == null) {
      return true
    }
    if (this.shape != null && other.shape != null) {
      if (this.shape.length != other.shape.length) return false
      var i = 0
      while(i < this.shape.length) {
        if (this.shape(i) != other.shape(i)) return false
        i += 1
      }
      return true
    } else {
      return false
    }
  }

  override def toString: String = {
    s"NativeData([${shape.mkString("x")}], ${layout})"
  }

  override def cloneFormat(): MemoryData = new NativeData(_shape, _layout)
}

private[mkldnn] object MemoryData {

  def noUndef(formats: Array[MemoryData]): Boolean = {
    if (formats == null || formats.length == 0) return true
    formats.foreach(f => if (f.layout == Memory.Format.format_undef) return false)
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