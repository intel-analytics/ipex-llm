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

package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.utils.Activities

import scala.reflect.ClassTag

class Sequential[A <: Activities : ClassTag, B <: Activities : ClassTag, T: ClassTag]
  (implicit ev: TensorNumeric[T]) extends Container[A, B, T] {

  var classPtr = 0L
  override def updateOutput(input: A): B = {
    var i = 0
    var result = input.asInstanceOf[Activities]

    var prev = getPrevPtr()
    while (i < modules.length) {
      if (initForward) {
        modules(i).setPrevPtr(prev)
      }
      result = modules(i).forward(result)
      if (initForward) {
        prev = modules(i).getOutputPtr()
      }
      i += 1
    }

    initForward = false
    this.output = result.asInstanceOf[B]
    output
  }

  override def updateGradInput(input: A, nextError: B): A = {
    var i = modules.length - 1
    var error = nextError.asInstanceOf[Activities]
    var next = getNextPtr()
    while (i > 0) {
      if (initBackward) {
        modules(i).setNextPtr(next)
      }
      val input = modules(i - 1).output
      error = modules(i).backward(input, error)
      if (initBackward) {
        next = modules(i).getInputPtr()
      }
      i -= 1
    }
    if (initBackward) {
      modules(0).setNextPtr(next)
      initBackward = false
    }
    error = modules(0).backward(input.asInstanceOf[Activities], error)

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

  override def initMkl(prevPtr : Long) : Unit = {
    println("I WANT TO SET THE PREV LAYOUT IN SEQUENTIAL")
    if (modules.length > 0) {
//      if (prevPtr != modules(0).getInputPtr())
//        modules(0).initMkl(prevPtr)

      var prev = prevPtr
      for (i <- 0 until modules.length) {
        modules(i).initMkl(prev)
        prev = modules(i).getOutputPtr()
        // println(modules(i))
      }
    }
  }

  override def getClassPtr() : Long = {
    if (modules.length >= 1) {
      modules(0).getClassPtr()
    } else { 0L } // If there isn't a Module in Sequential, it will return 0L.
  }

  override def getInputPtr(): Long = {
    if (modules.length > 0) {
      modules(0).getInputPtr()
    } else { 0L }
  }

  override def getOutputPtr(): Long = {
    if (modules.length > 0) {
      modules(modules.length - 1).getOutputPtr()
    } else { 0L }
  }
}


