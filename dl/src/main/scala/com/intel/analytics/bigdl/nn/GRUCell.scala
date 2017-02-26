/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}
import jdk.nashorn.internal.ir.Splittable

import scala.reflect.ClassTag

class GRUCell[T : ClassTag] (
  inputSize: Int = 4,
  outputSize: Int = 3,
  p: Double = 0,
  private var initMethod: InitializationMethod = Default)
  (implicit ev: TensorNumeric[T])
  extends AbstractModule[Table, Table, T] {
  var i2g: AbstractModule[Activity, Activity, T] = _
  var o2g: AbstractModule[Activity, Activity, T] = _
  var GRU: Sequential[T] = buildGRU()

  def buildGRU(): Sequential[T] = {
    if (p != 0) {
      i2g = Sequential()
        .add(ConcatTable()
          .add(Dropout(p))
          .add(Dropout(p)))
        .add(ParallelTable()
          .add(Linear(inputSize, outputSize))
          .add(Linear(inputSize, outputSize)))
        .add(JoinTable(1, 1))

      o2g = Sequential()
        .add(ConcatTable()
          .add(Dropout(p))
          .add(Dropout(p)))
        .add(ParallelTable()
          .add(Linear(inputSize, outputSize, withBias = false))
          .add(Linear(inputSize, outputSize, withBias = false)))
        .add(JoinTable(1, 1))
    } else {
      i2g = Linear(inputSize, 2 * outputSize)
      o2g = Linear(inputSize, 2 * outputSize, withBias = false)
    }

    Sequential()
      .add(ParallelTable()
        .add(i2g)
        .add(o2g))
      .add(CAddTable())
      .add(Reshape(Array(2, outputSize)))
//      .add(SplitTable(1, 2))


    val model = Sequential()
    output = T(Tensor())
    GRU = model
    GRU
  }

  override def updateOutput(input: Table): Table = {
    output = GRU.updateOutput(input).toTable
    output
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    gradInput = GRU.updateGradInput(input, gradOutput).toTable
    gradInput
  }

  override def accGradParameters(input: Table, gradOutput: Table, scale: Double): Unit = {
    GRU.accGradParameters(input, gradOutput, scale)
  }

  override def updateParameters(learningRate: T): Unit = {
    GRU.updateParameters(learningRate)
  }

  override def zeroGradParameters(): Unit = {
    GRU.zeroGradParameters()
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    GRU.parameters()
  }
}

object GRUCell {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputSize: Int = 4,
    outputSize: Int = 3)
    (implicit ev: TensorNumeric[T]): GRUCell[T] = {
    new GRUCell[T](inputSize, outputSize)
  }
}