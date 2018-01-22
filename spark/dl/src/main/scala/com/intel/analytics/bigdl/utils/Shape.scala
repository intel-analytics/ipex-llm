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

package com.intel.analytics.bigdl.utils

import scala.reflect.ClassTag

trait Shape {
  /**
   * Use this method if its only a single Shape
   */
  def toSingle(): List[Int] = throw new RuntimeException("Invalid operation")

  /**
   * Use this method if the current Shape consist of multiple value
   */
  def toMulti(): List[Shape] = throw new RuntimeException("Invalid operation")

  /**
   * Update the given dim and return a new copy
   */
  def copyAndUpdate(dim: Int, v: Int): Shape = throw new RuntimeException("Invalid operation")

  /**
   * Update the given dim and return a new copy
   */
  def copyAndUpdate(dim: Int, v: Shape): Shape
    = throw new RuntimeException("Invalid operation")


  protected def getDim(dim: Int, length: Int): Int = {
    val rdim = if (dim < 0) {
      length + dim
    } else {
      dim
    }
    require(rdim < length && rdim >=0, "out of range")
    rdim
  }
}

case class SingleShape(val value: List[Int]) extends Shape {
  override def toSingle(): List[Int] = value

  override def copyAndUpdate(dim: Int, v: Int): Shape = {
    val cValue = value.toArray
    cValue(getDim(dim, value.length)) = v
    Shape(cValue)
  }

  override def canEqual(a: Any): Boolean = a.isInstanceOf[SingleShape]

  override def equals(that: Any): Boolean =
    that match {
      case that: SingleShape => that.canEqual(this) && this.hashCode == that.hashCode
      case _ => false
    }

  override def hashCode: Int = {
    val prime = 31
    var result = 1
    result = prime * value.hashCode()
    return result
  }
}


case class MultiShape(val value: List[Shape]) extends Shape {

  override def toMulti(): List[Shape] = value

  override def copyAndUpdate(dim: Int, v: Shape): Shape = {
    val cValue = value.toArray
    cValue(getDim(dim, value.length)) = v
    MultiShape(cValue.toList)
  }

  override def canEqual(a: Any): Boolean = a.isInstanceOf[MultiShape]

  override def equals(that: Any): Boolean =
    that match {
      case that: MultiShape => that.canEqual(this) && this.hashCode == that.hashCode
      case _ => false
    }

  override def hashCode: Int = {
    val prime = 31
    var result = 1
    result = prime * value.hashCode()
    return result
  }
}

object Shape {

  def apply(item : Array[Int]): Shape = {
    if (item == null) {
      throw new IllegalArgumentException("Empty value")
    }
    new SingleShape(item.toList)
  }

  def apply(item : Int*): Shape = {
    new SingleShape(item.toList)
  }

  def apply[T <: Shape : ClassTag](shapes : List[Shape]): Shape = {
    if (shapes.length > 1) {
      MultiShape(shapes.toList)
    } else if (shapes.length == 1) {
      shapes(0)
    } else {
      throw new IllegalArgumentException("Empty value")
    }
  }
}
