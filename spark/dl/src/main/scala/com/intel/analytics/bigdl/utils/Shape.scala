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

trait Shape {
  def toSingle(): List[Int] = throw new RuntimeException("Invalid operation")
  def toMulti(): List[Shape] = throw new RuntimeException("Invalid operation")
}

class SingleShape(val value: List[Int]) extends Shape {
  override def toSingle(): List[Int] = value
}


case class MultiShape(val value: List[Shape]) extends Shape {

  override def toMulti(): List[Shape] = value
}

object Shape {

  def apply(item : Seq[Int]): Shape = {
    if (item == null) {
      return null
    }
    new SingleShape(item.toList)
  }

  def apply(shapes : List[Shape]): Shape = {
    if (shapes.length > 1) {
      MultiShape(shapes.toList)
    } else if (shapes.length == 1) {
      shapes(0)
    } else {
      null
    }
  }
}