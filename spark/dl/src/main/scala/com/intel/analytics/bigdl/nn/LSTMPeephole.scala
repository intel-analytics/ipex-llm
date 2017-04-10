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
import com.intel.analytics.bigdl.optim.Regularizer
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
 * @param  p is used for [[Dropout]] probability. For more details about
 *           RNN dropouts, please refer to
 *           [RnnDrop: A Novel Dropout for RNNs in ASR]
 *           (http://www.stat.berkeley.edu/~tsmoon/files/Conference/asru2015.pdf)
 *           [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks]
 *           (https://arxiv.org/pdf/1512.05287.pdf)
 * @param wRegularizer: instance of [[Regularizer]]
 *                    (eg. L1 or L2 regularization), applied to the input weights matrices.
 * @param uRegularizer: instance [[Regularizer]]
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
 * @param bRegularizer: instance of [[Regularizer]]
            applied to the bias.
 */
@SerialVersionUID(- 7566757838561436619L)
class LSTMPeephole[T : ClassTag] (
  val inputSize: Int,
  val hiddenSize: Int,
  val p: Double = 0.0,
  val wRegularizer: Regularizer[T] = null,
  val uRegularizer: Regularizer[T] = null,
  val bRegularizer: Regularizer[T] = null
)
  (implicit ev: TensorNumeric[T])
  extends Cell[T](
    hiddensShape = Array(hiddenSize, hiddenSize),
    regularizers = Array(wRegularizer, uRegularizer, bRegularizer)
  ) {
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
      .add(Linear(inputSize, hiddenSize, wRegularizer = wRegularizer,
        bRegularizer = bRegularizer))
    val h2g = Sequential()
      .add(Dropout(p))
      .add(Linear(hiddenSize, hiddenSize,
        withBias = false, wRegularizer = uRegularizer))

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
      .add(Linear(inputSize, hiddenSize, wRegularizer = wRegularizer,
        bRegularizer = bRegularizer))
    val h2h = Sequential()
      .add(Dropout(p))
      .add(Linear(hiddenSize, hiddenSize, withBias = false,
        wRegularizer = uRegularizer))

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
    p: Double = 0.0,
    wRegularizer: Regularizer[T] = null,
    uRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null
  )
    (implicit ev: TensorNumeric[T]): LSTMPeephole[T] = {
    new LSTMPeephole[T](inputSize, hiddenSize, p, wRegularizer, uRegularizer, bRegularizer)
  }
}

