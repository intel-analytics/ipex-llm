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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class LSTMCell[T : ClassTag] (
  inputSize: Int = 4,
  hiddenSize: Int = 3,
  private var initMethod: InitializationMethod = Default)
  (implicit ev: TensorNumeric[T])
  extends AbstractModule[Table, Table, T] {
  var inputGate: Sequential[T] = _
  var forgetGate: Sequential[T] = _
  var outputGate: Sequential[T] = _
  var hiddenLayer: Sequential[T] = _
  var cellLayer: Sequential[T] = _

  def buildGate(): Sequential[T] = {
    val gate = Sequential()
      .add(NarrowTable(1, 2))

    val i2g = Sequential()
      .add(Linear(inputSize, hiddenSize))
    val h2g = Sequential()
      .add(Linear(hiddenSize, hiddenSize))

    gate
      .add(ParallelTable()
        .add(i2g)
        .add(h2g))
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
      .add(Linear(inputSize, hiddenSize))
    val h2h = Sequential()
      .add(Linear(hiddenSize, hiddenSize))

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
      .add(ConcatTable()
        .add(NarrowTable(1, 2))
        .add(cellLayer))
      .add(FlattenTable())
      .add(ConcatTable()
          .add(Sequential())
            .add(ConcatTable()
              .add(outputGate)
              .add(Sequential()
                .add(SelectTable(3))
                .add(Tanh())))
            .add(CMulTable())
          .add(SelectTable(3)))

    LSTM
  }

  /**
   * Computes the output using the current parameter set of the class and input. This function
   * returns the result which is stored in the output field.
   *
   * @param input
   * @return
   */
  override def updateOutput(input: Table): Table = ???

  /**
   * Computing the gradient of the module with respect to its own input. This is returned in
   * gradInput. Also, the gradInput state variable is updated accordingly.
   *
   * @param input
   * @param gradOutput
   * @return
   */
  override def updateGradInput(input: Table, gradOutput: Table): Table = ???
}
