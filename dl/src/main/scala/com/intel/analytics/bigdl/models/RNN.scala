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

package com.intel.analytics.bigdl.models

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

class RnnCell[T : ClassTag] (
  inputSize: Int = 4,
  hiddenSize: Int = 3) (implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[T], T] {

  val i2h = new Linear[T](inputSize, hiddenSize)
  val h2h = new Linear[T](hiddenSize, hiddenSize)
  val cAddTable = new CAddTable[T](false)
  gradInput = T()

  override def updateOutput(input: Table): Tensor[T] = {
    output = cAddTable.updateOutput(
      T(i2h.updateOutput(input(1)), h2h.updateOutput(input(2))))
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    gradInput(1) = i2h.updateGradInput(input(1), gradOutput)
    gradInput(2) = h2h.updateGradInput(input(2), gradOutput)
    gradInput
  }
  override def accGradParameters(input: Table, gradOutput: Tensor[T],
                                 scale: Double = 1.0): Unit = {
    i2h.accGradParameters(input(1), gradOutput)
    h2h.accGradParameters(input(2), gradOutput)
  }
  override def updateParameters(learningRate: T): Unit = {
    i2h.updateParameters(learningRate)
    h2h.updateParameters(learningRate)
  }

  override def zeroGradParameters(): Unit = {
    i2h.zeroGradParameters()
    h2h.zeroGradParameters()
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(i2h.weight, i2h.bias, h2h.weight, h2h.bias),
      Array(i2h.gradWeight, i2h.gradBias, h2h.gradWeight, h2h.gradBias))
  }


  override def toString(): String = {
    var str = "nn.RnnCell"
    str
  }
}

object RnnCell {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputSize: Int = 4,
    hiddenSize: Int = 3)
   (implicit ev: TensorNumeric[T]) : RnnCell[T] = {
    new RnnCell[T](inputSize, hiddenSize)
  }
}
