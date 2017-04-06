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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

/**
 * Long Short Term Memory architecture with peephole.
 * Ref. A.: http://arxiv.org/pdf/1303.5778v1 (blueprint for this module)
 * B. http://web.eecs.utk.edu/~itamar/courses/ECE-692/Bobby_paper1.pdf
 * C. http://arxiv.org/pdf/1503.04069v1.pdf
 * D. https://github.com/wojzaremba/lstm
 *
 * @param inputSize the size of each input vector
 * @param hiddenSize Hidden unit size in the LSTM
 */
@SerialVersionUID(- 7566757838561436619L)
class LSTMPeephole[T : ClassTag] (
  val inputSize: Int,
  val hiddenSize: Int,
  val p: Double = 0.0
)
  (implicit ev: TensorNumeric[T])
  extends Cell[T](hiddensShape = Array(hiddenSize, hiddenSize)) {
  var inputGate: Sequential[T] = _
  var forgetGate: Sequential[T] = _
  var outputGate: Sequential[T] = _
  var hiddenLayer: Sequential[T] = _
  var cellLayer: Sequential[T] = _
  override var cell: AbstractModule[Activity, Activity, T] = buildLSTM()

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

    cell = LSTM
    LSTM
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[LSTMPeephole[T]]

  override def equals(other: Any): Boolean = other match {
    case that: LSTMPeephole[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        inputSize == that.inputSize &&
        hiddenSize == that.hiddenSize &&
        p == that.p
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), inputSize, hiddenSize, p)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }

  override def reset(): Unit = {
    super.reset()
    cell.reset()
  }

  override def toString: String = s"LSTMPeephole($inputSize, $hiddenSize, $p)"
}

object LSTMPeephole {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputSize: Int = 4,
    hiddenSize: Int = 3,
    p: Double = 0.0
  )
    (implicit ev: TensorNumeric[T]): LSTMPeephole[T] = {
    new LSTMPeephole[T](inputSize, hiddenSize, p)
  }
}

