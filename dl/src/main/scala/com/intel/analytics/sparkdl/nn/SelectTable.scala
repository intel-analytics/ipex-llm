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
import com.intel.analytics.sparkdl.tensor._
import com.intel.analytics.sparkdl.utils.{T, Table}

import scala.reflect.ClassTag

class SelectTable [T: ClassTag](
  index: Int)
  (implicit ev: TensorNumeric[T]) extends Container[Table, Table, T]  {

  override def updateOutput(input: Table): Table = {
    val index = if (this.index < 0) input.getState().size + index else index

    require(input.contains(index), "index does not exist in the input table")
    output = input(index)

    output
  }

  def zeroTableCopy(t1: Table, t2: Table): Table = {
    for ((k, v) <- t2.getState()) {
      if (k.isInstanceOf[Table]) {
        t1.update(k, zeroTableCopy(if (t1[Table](k).contains(k)) t1(k) else T(), t2(k)))
      } else {
        require(v.isInstanceOf[Tensor[T]], "Input can only consist of Tensor or Table")
        val tensorV = v.asInstanceOf[Tensor[T]]
        if (!t1.contains(k)) {
          t1.update(k, tensorV.clone().zero())
        } else {
          t1[Tensor[T]](k).resizeAs(tensorV)
          t1[Tensor[T]](k).zero()
        }
      }
    }
    for ((k, v) <- t1.getState()) {
      if (!t2.contains(k)) {
        t1.update(k, null)
      }
    }

    t1
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    zeroTableCopy(gradInput, input)
    val index = if (this.index < 0) {
      input.getState().size + this.index + 1
    } else {
      this.index
    }

    require(gradInput.contains(index), "Index exceeds the size of input table")

    gradInput
  }

  override def toString: String = s"SelectTable($index)"
}
