/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Activities

import scala.reflect.ClassTag

class Sequential[A <: Activities : ClassTag, B <: Activities : ClassTag, T: ClassTag]
  (implicit ev: TensorNumeric[T]) extends Container[A, B, T] {

  override def updateOutput(input: A): B = {
    var i = 0
    var result = input.asInstanceOf[Activities]
    while (i < modules.length) {
      result = modules(i).forward(result)
      i += 1
    }

    this.output = result.asInstanceOf[B]
    output
  }

  override def updateGradInput(input: A, nextError: B): A = {
    var i = modules.length - 1
    var error = nextError.asInstanceOf[Activities]
    while (i > 0) {
      if (modules(i).propagateBack) {
        val input = modules(i - 1).output
        error = modules(i).backward(input, error)
        i -= 1
      } else {
        println(s"${modules(i).getName()} does not need backward computation.")
      }
    }
    if (modules(0).propagateBack) {
      error = modules(0).backward(input.asInstanceOf[Activities], error)
    } else {
      println(s"${modules(0).getName()} does not need backward computation.")
    }

    this.gradInput = error.asInstanceOf[A]
    gradInput
  }

  override def equals(obj: Any): Boolean = {
    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[Sequential[A, B, T]]) {
      return false
    }
    val other = obj.asInstanceOf[Sequential[A, B, T]]
    if (this.eq(other)) {
      return true
    }

    if (this.modules.length != other.modules.length) {
      return false
    }

    val moduleLength = modules.length
    var i = 0
    while (i < moduleLength) {
      if (modules(i) != other.modules(i)) {
        return false
      }
      i += 1
    }

    true
  }

  override def hashCode() : Int = {
    val seed = 37
    var hash = super.hashCode()
    val moduleLength = modules.length
    var i = 0
    while (i < moduleLength) {
      hash = hash * seed + modules(i).hashCode()
      i += 1
    }

    hash
  }

  override def toString(): String = {
    val tab = "  "

    s"nn.Sequential {${line + tab}[input -> ${
      modules.zipWithIndex.map {
        case (m: Module[Activities, Activities, T], i: Int) => "(" + (i + 1) + ")"
      }.
        mkString(" -> ")
    } -> output]${line + tab}" +
      s"${
        modules.zipWithIndex.map {
          case (model: Module[Activities, Activities, T], index: Int)
          => s"(${index + 1}): ${model.setLine(line + tab)}"
        }.
          mkString(line + tab)
      }$line}"
  }

}

object Sequential {
  def apply[A <: Activities : ClassTag, B <: Activities : ClassTag,
      @specialized(Float, Double) T: ClassTag]()
      (implicit ev: TensorNumeric[T]) : Sequential[A, B, T] = {
    new Sequential[A, B, T]()
  }
}
