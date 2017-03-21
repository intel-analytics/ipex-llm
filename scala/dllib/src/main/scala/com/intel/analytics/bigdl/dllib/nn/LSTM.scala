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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

/**
 * Long Short Term Memory architecture.
 * Ref. A.: http://arxiv.org/pdf/1303.5778v1 (blueprint for this module)
 * B. http://web.eecs.utk.edu/~itamar/courses/ECE-692/Bobby_paper1.pdf
 * C. http://arxiv.org/pdf/1503.04069v1.pdf
 * D. https://github.com/wojzaremba/lstm
 *
 * @param inputSize the size of each input vector
 * @param hiddenSize Hidden unit size in the LSTM
 */
@SerialVersionUID(- 8176191554025511686L)
class LSTM[T : ClassTag] (
  val inputSize: Int,
  val hiddenSize: Int,
  val p: Double = 0.2)
  (implicit ev: TensorNumeric[T])
  extends Cell[T](hiddensShape = Array(hiddenSize, hiddenSize)) {
  var gates: Sequential[T] = _
  var cellLayer: Sequential[T] = _
  override var cell: AbstractModule[Activity, Activity, T] = buildLSTM()

  def buildGates(): Sequential[T] = {
    val gates = Sequential()
      .add(NarrowTable(1, 2))

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
        .add(JoinTable(1, 1))

      h2g = Sequential()
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
        .add(JoinTable(1, 1))
    } else {
      i2g = Linear(inputSize, 4 * hiddenSize)
      h2g = Linear(hiddenSize, 4 * hiddenSize, withBias = false)
    }

    gates
      .add(ParallelTable()
        .add(i2g)
        .add(h2g))
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
        .add(gates)
        .add(SelectTable(3)))
      .add(FlattenTable()) // input, hidden, forget, output, cell

    val cellLayer = Sequential()
      .add(ConcatTable()
        .add(Sequential()
          .add(NarrowTable(1, 2))
          .add(CMulTable()))
        .add(Sequential()
          .add(ConcatTable()
            .add(SelectTable(3))
            .add(SelectTable(5)))
          .add(CMulTable())))
      .add(CAddTable())

    lstm
      .add(ConcatTable()
        .add(cellLayer)
        .add(SelectTable(4)))
      .add(FlattenTable())


    lstm
      .add(ConcatTable()
        .add(Sequential()
          .add(ConcatTable()
            .add(Sequential()
              .add(SelectTable(1))
              .add(Tanh()))
            .add(SelectTable(2)))
          .add(CMulTable()))
        .add(SelectTable(1)))
      .add(ConcatTable()
        .add(SelectTable(1))
        .add(Identity()))

    output = T(Tensor(), T())
    this.cell = lstm
    lstm
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[LSTM[T]]

  override def equals(other: Any): Boolean = other match {
    case that: LSTM[T] =>
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


  override def toString: String = s"LSTM($inputSize, $hiddenSize, $p)"
}

object LSTM {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputSize: Int,
    hiddenSize: Int)
    (implicit ev: TensorNumeric[T]): LSTM[T] = {
    new LSTM[T](inputSize, hiddenSize)
  }
}
