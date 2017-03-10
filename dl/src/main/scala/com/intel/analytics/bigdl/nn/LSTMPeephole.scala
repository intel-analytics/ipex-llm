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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

class LSTMPeephole[T : ClassTag] (
  inputSize: Int = 4,
  hiddenSize: Int = 3,
  p: Double = 0)
  (implicit ev: TensorNumeric[T])
  extends Cell[T] {
  val nHids = 2
  var inputGate: Sequential[T] = _
  var forgetGate: Sequential[T] = _
  var outputGate: Sequential[T] = _
  var hiddenLayer: Sequential[T] = _
  var cellLayer: Sequential[T] = _
  var lstm: Sequential[T] = buildLSTM()

  def buildGate(): Sequential[T] = {
    val gate = Sequential()

    val i2g = Sequential()
      .add(Dropout(p))
      .add(Linear(inputSize, hiddenSize))
    val h2g = Sequential()
      .add(Dropout(p))
      .add(Linear(hiddenSize, hiddenSize, withBias = false))

    gate
      .add(ParallelTable()
        .add(i2g)
        .add(h2g)
        .add(CMul(Array(hiddenSize))))
      .add(CAddTable())
      .add(Sigmoid())
  }

  def buildInputGate(): Sequential[T] = {
    inputGate = buildGate()
    inputGate
  }

  def buildForgetGate(): Sequential[T] = {
    forgetGate = buildGate()
    forgetGate
  }

  def buildOutputGate(): Sequential[T] = {
    outputGate = buildGate()
    outputGate
  }

  def buildHidden(): Sequential[T] = {
    val hidden = Sequential()
      .add(NarrowTable(1, 2))

    val i2h = Sequential()
      .add(Dropout(p))
      .add(Linear(inputSize, hiddenSize))
    val h2h = Sequential()
      .add(Dropout(p))
      .add(Linear(hiddenSize, hiddenSize, withBias = false))

    hidden
      .add(ParallelTable()
        .add(i2h)
        .add(h2h))
      .add(CAddTable())
      .add(Tanh())

    this.hiddenLayer = hidden
    hidden
  }

  def buildCell(): Sequential[T] = {
    buildInputGate()
    buildForgetGate()
    buildHidden()

    val forgetLayer = Sequential()
      .add(ConcatTable()
        .add(forgetGate)
        .add(SelectTable(3)))
      .add(CMulTable())

    val inputLayer = Sequential()
      .add(ConcatTable()
        .add(inputGate)
        .add(hiddenLayer))
      .add(CMulTable())

    val cellLayer = Sequential()
      .add(ConcatTable()
        .add(forgetLayer)
        .add(inputLayer))
      .add(CAddTable())

    this.cellLayer = cellLayer
    cellLayer
  }

  def buildLSTM(): Sequential[T] = {
    buildCell()
    buildOutputGate()

    val LSTM = Sequential()
      .add(FlattenTable())
      .add(ConcatTable()
        .add(NarrowTable(1, 2))
        .add(cellLayer))
      .add(FlattenTable())
      .add(ConcatTable()
        .add(Sequential()
          .add(ConcatTable()
            .add(outputGate)
            .add(Sequential()
              .add(SelectTable(3))
              .add(Tanh())))
          .add(CMulTable()))
        .add(SelectTable(3)))
      .add(ConcatTable()
        .add(SelectTable(1))
        .add(Identity()))

    output = T(Tensor(), T())
    lstm = LSTM
    LSTM
  }

  override def updateOutput(input: Table): Table = {
    output = lstm.updateOutput(input).toTable
    output
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    gradInput = lstm.updateGradInput(input, gradOutput).toTable
    gradInput
  }

  override def accGradParameters(input: Table, gradOutput: Table, scale: Double): Unit = {
    lstm.accGradParameters(input, gradOutput, scale)
  }

  override def updateParameters(learningRate: T): Unit = {
    lstm.updateParameters(learningRate)
  }

  override def zeroGradParameters(): Unit = {
    lstm.zeroGradParameters()
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    lstm.parameters()
  }
}

object LSTMPeephole {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputSize: Int = 4,
    hiddenSize: Int = 3,
    p: Double = 0)
    (implicit ev: TensorNumeric[T]): LSTMPeephole[T] = {
    new LSTMPeephole[T](inputSize, hiddenSize, p)
  }
}

