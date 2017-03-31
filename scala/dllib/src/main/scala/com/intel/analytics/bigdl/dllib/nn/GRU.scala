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
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * Gated Recurrent Units architecture.
 * The first input in sequence uses zero value for cell and hidden state
 *
 * Ref.
 * 1. http://www.wildml.com/2015/10/
 * recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/
 *
 * 2. https://github.com/Element-Research/rnn/blob/master/GRU.lua
 *
 * @param inputSize the size of each input vector
 * @param outputSize Hidden unit size in GRU
 */
@SerialVersionUID(6717988395573528459L)
class GRU[T : ClassTag] (
  val inputSize: Int,
  val outputSize: Int)
  (implicit ev: TensorNumeric[T])
  extends Cell[T](hiddensShape = Array(outputSize)) {
  val p: Double = 0 // Dropout threshold
  var i2g: AbstractModule[_, _, T] = _
  var h2g: AbstractModule[_, _, T] = _
  var gates: AbstractModule[_, _, T] = _
  var GRU: AbstractModule[_, _, T] = buildGRU()

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
    buildGates()

    val gru = Sequential()
      .add(ConcatTable()
        .add(Identity())
        .add(gates))
      .add(FlattenTable()) // x(t), h(t - 1), r(t), z(t)

    val h_hat = Sequential()
      .add(ConcatTable()
        .add(SelectTable(1))
        .add(Sequential()
          .add(NarrowTable(2, 2))
          .add(CMulTable())))
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
      .add(ConcatTable()
        .add(Identity[T]())
        .add(Identity[T]()))

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

  override def canEqual(other: Any): Boolean = other.isInstanceOf[GRU[T]]

  override def equals(other: Any): Boolean = other match {
    case that: GRU[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        inputSize == that.inputSize &&
        outputSize == that.outputSize &&
        p == that.p
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), inputSize, outputSize, p)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object GRU {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputSize: Int = 4,
    outputSize: Int = 3)
    (implicit ev: TensorNumeric[T]): GRU[T] = {
    new GRU[T](inputSize, outputSize)
  }
}
