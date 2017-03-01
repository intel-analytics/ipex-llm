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
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

class LSTMCell[T : ClassTag] (
  inputSize: Int = 4,
  hiddenSize: Int = 3,
  p: Double = 0)
  (implicit ev: TensorNumeric[T])
  extends AbstractModule[Table, Table, T] {
  var inputGate: Sequential[T] = _
  var forgetGate: Sequential[T] = _
  var outputGate: Sequential[T] = _
  var hiddenLayer: Sequential[T] = _
  var cellLayer: Sequential[T] = _
  var lstm: Sequential[T] = buildLSTM()

  def buildGate(): Sequential[T] = {
    val gate = Sequential()
      .add(NarrowTable(2, 2))

    val i2g = Sequential()
      .add(Linear(inputSize, hiddenSize))
    val h2g = Sequential()
      .add(Linear(hiddenSize, hiddenSize, withBias = false))

    gate
      .add(ParallelTable()
        .add(h2g)
        .add(i2g))
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
      .add(NarrowTable(2, 2))

    val i2h = Sequential()
      .add(Linear(inputSize, hiddenSize))
    val h2h = Sequential()
      .add(Linear(hiddenSize, hiddenSize, withBias = false))

    hidden
      .add(ParallelTable()
        .add(h2h)
        .add(i2h))
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
        .add(SelectTable(1)))
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
        .add(cellLayer)
        .add(NarrowTable(2, 2)))
      .add(FlattenTable())
      .add(ConcatTable()
        .add(SelectTable(1))
        .add(Sequential()
          .add(ConcatTable()
            .add(outputGate)
            .add(Sequential()
              .add(SelectTable(1))
              .add(Tanh())))
          .add(CMulTable())))
      .add(ConcatTable()
        .add(NarrowTable(1, 2))
        .add(SelectTable(2)))

    output = T(T(), Tensor())
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

object LSTMCell {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputSize: Int = 4,
    hiddenSize: Int = 3)
    (implicit ev: TensorNumeric[T]): LSTMCell[T] = {
    new LSTMCell[T](inputSize, hiddenSize)
  }
}

class FastLSTMCell[T : ClassTag] (
  inputSize: Int = 4,
  hiddenSize: Int = 3,
  p: Double = 0,
  private var initMethod: InitializationMethod = Default)
  (implicit ev: TensorNumeric[T])
  extends AbstractModule[Table, Table, T] {
  var gates: Sequential[T] = _
  var cellLayer: Sequential[T] = _
  var lstm: Sequential[T] = buildLSTM()

  def buildGates(): Sequential[T] = {
    val gates = Sequential()
      .add(NarrowTable(2, 2))

    var i2g: AbstractModule[_, _, T] = null
    var h2g: AbstractModule[_, _, T] = null

    if (p != 0) {
      i2g = Sequential()
        .add(ConcatTable()
          .add(Dropout(p))
          .add(Dropout(p))
          .add(Dropout(p))
          .add(Dropout(p)))
        .add(ParallelTable()
          .add(Linear(inputSize, hiddenSize))
          .add(Linear(inputSize, hiddenSize))
          .add(Linear(inputSize, hiddenSize))
          .add(Linear(inputSize, hiddenSize)))
        .add(JoinTable(2, 1))

      val o2g = Sequential()
        .add(ConcatTable()
          .add(Dropout(p))
          .add(Dropout(p))
          .add(Dropout(p))
          .add(Dropout(p)))
        .add(ParallelTable()
          .add(Linear(hiddenSize, hiddenSize, withBias = false))
          .add(Linear(hiddenSize, hiddenSize, withBias = false))
          .add(Linear(hiddenSize, hiddenSize, withBias = false))
          .add(Linear(hiddenSize, hiddenSize, withBias = false)))
        .add(JoinTable(2, 1))
    } else {
      i2g = Linear(inputSize, 4 * hiddenSize)
      h2g = Linear(hiddenSize, 4 * hiddenSize, withBias = false)
    }

    gates
      .add(ParallelTable()
        .add(h2g)
        .add(i2g))
      .add(CAddTable())
      .add(Reshape(Array(4, hiddenSize)))
      .add(SplitTable(1, 2))
      .add(ParallelTable()
        .add(Sigmoid())
        .add(Tanh())
        .add(Sigmoid())
        .add(Sigmoid()))

    this.gates = gates
    gates
  }

  def buildLSTM(): Sequential[T] = {
    buildGates()

    val lstm = Sequential()
      .add(FlattenTable())
      .add(ConcatTable()
        .add(SelectTable(1))
        .add(gates))
      .add(FlattenTable()) // cell, input, hidden, forget, output

    val cellLayer = Sequential()
      .add(ConcatTable()
        .add(Sequential()
          .add(NarrowTable(2, 2))
          .add(CMulTable()))
        .add(Sequential()
          .add(ConcatTable()
            .add(SelectTable(1))
            .add(SelectTable(4)))
          .add(CMulTable())))
      .add(CAddTable())

    lstm
      .add(ConcatTable()
        .add(cellLayer)
        .add(SelectTable(5)))
      .add(FlattenTable())


    lstm
      .add(ConcatTable()
        .add(SelectTable(1))
        .add(Sequential()
          .add(ConcatTable()
            .add(Sequential()
              .add(SelectTable(1))
              .add(Tanh()))
            .add(SelectTable(2)))
          .add(CMulTable())))
      .add(ConcatTable()
        .add(Identity())
        .add(SelectTable(2)))

    output = T(T(), Tensor())
    this.lstm = lstm
    lstm
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

object FastLSTMCell {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputSize: Int = 4,
    hiddenSize: Int = 3,
    p: Double = 0)
    (implicit ev: TensorNumeric[T]): FastLSTMCell[T] = {
    new FastLSTMCell[T](inputSize, hiddenSize)
  }
}
