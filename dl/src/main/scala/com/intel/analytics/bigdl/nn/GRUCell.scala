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

import scala.reflect.ClassTag

class GRUCell[T : ClassTag] (
  inputSize: Int = 4,
  outputSize: Int = 3,
  p: Double = 0,
  private var initMethod: InitializationMethod = Default)
  (implicit ev: TensorNumeric[T])
  extends AbstractModule[Table, Table, T] {
  var i2g: AbstractModule[Activity, Activity, T] = _
  var h2g: AbstractModule[Activity, Activity, T] = _
  var gates: AbstractModule[Activity, Activity, T] = _
  var GRU: AbstractModule[Activity, Activity, T] = buildGRU()

  def buildGates(): AbstractModule[Activity, Activity, T] = {
    if (p != 0) {
      i2g = Sequential()
        .add(ConcatTable()
          .add(Dropout(p))
          .add(Dropout(p)))
        .add(ParallelTable()
          .add(Linear(inputSize, outputSize))
          .add(Linear(inputSize, outputSize)))
        .add(JoinTable(1, 1))

      h2g = Sequential()
        .add(ConcatTable()
          .add(Dropout(p))
          .add(Dropout(p)))
        .add(ParallelTable()
          .add(Linear(outputSize, outputSize, withBias = false))
          .add(Linear(outputSize, outputSize, withBias = false)))
        .add(JoinTable(1, 1))
    } else {
      i2g = Linear(inputSize, 2 * outputSize)
      h2g = Linear(outputSize, 2 * outputSize, withBias = false)
    }

    gates = Sequential()
      .add(ParallelTable()
        .add(i2g)
        .add(h2g))
      .add(CAddTable())
      .add(Reshape(Array(2, outputSize)))
      .add(SplitTable(1, 2))
      .add(ParallelTable()
        .add(Sigmoid())
        .add(Sigmoid()))

    gates
  }

  def buildGRU(): AbstractModule[Activity, Activity, T] = {
    val gru = Sequential()
      .add(ConcatTable()
        .add(Identity())
        .add(gates))
      .add(FlattenTable()) // x(t), h(t - 1), r(t), z(t)

    val h_hat = Sequential()
      .add(ConcatTable()
        .add(Sequential()
          .add(NarrowTable(2, 2))
          .add(CMulTable()))
        .add(SelectTable(1)))
      .add(ParallelTable()
        .add(Sequential()
          .add(Dropout(p))
          .add(Linear(inputSize, outputSize)))
        .add(Sequential()
          .add(Dropout(p))
          .add(Linear(outputSize, outputSize, withBias = false))))
      .add(CAddTable())
      .add(Tanh())

    gru
      .add(ConcatTable()
        .add(Sequential()
          .add(ConcatTable()
            .add(h_hat)
            .add(Sequential()
              .add(SelectTable(4))
              .add(MulConstant(-1))
              .add(AddConstant(1))))
          .add(CMulTable()))
        .add(Sequential()
          .add(ConcatTable()
            .add(SelectTable(2))
            .add(SelectTable(4)))
          .add(CMulTable())))
      .add(CAddTable())

    output = T(Tensor())
    GRU = gru
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