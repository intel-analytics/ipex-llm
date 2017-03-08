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
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

class RnnCell[T : ClassTag] (
  inputSize: Int = 4,
  hiddenSize: Int = 3,
  activation: AbstractModule[_, _, T],
  private var initMethod: InitializationMethod = Default)
  (implicit ev: TensorNumeric[T])
  extends AbstractModule[Table, Table, T] {

  val parallelTable = ParallelTable[T]()
  val i2h = Linear[T](inputSize, hiddenSize)
  val h2h = Linear[T](hiddenSize, hiddenSize)
  parallelTable.add(i2h)
  parallelTable.add(h2h)
  val cAddTable = CAddTable[T]()

  val rnn = Sequential[T]()
    .add(parallelTable)
    .add(cAddTable)
    .add(activation)
    .add(ConcatTable()
      .add(Identity[T]())
      .add(Identity[T]()))

  def setInitMethod(initMethod: InitializationMethod): this.type = {
    this.initMethod = initMethod
    this
  }

  override def reset(): Unit = {
    initMethod match {
      case Default =>
//        parallelTable.modules.foreach( m => {
//          val inputSize = m.asInstanceOf[Linear[T]].weight.size(1).toFloat
//          val outputSize = m.asInstanceOf[Linear[T]].weight.size(2).toFloat
//          val stdv = 6.0 / (inputSize + outputSize)
//          m.asInstanceOf[Linear[T]].weight.apply1( _ =>
//            ev.fromType[Double](RNG.uniform(0, 1) * 2 * stdv - stdv))
//          m.asInstanceOf[Linear[T]].bias.apply1( _ => ev.fromType[Double](0.0))
//        })
      case _ =>
        throw new IllegalArgumentException(s"Unsupported initMethod type ${initMethod}")
    }
    zeroGradParameters()
  }

  override def updateOutput(input: Table): Table = {
    output = rnn.updateOutput(input).toTable
    output
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    gradInput = rnn.updateGradInput(input, gradOutput).toTable
    gradInput
  }

  override def accGradParameters(input: Table, gradOutput: Table,
    scale: Double = 1.0): Unit = {
    rnn.accGradParameters(input, gradOutput, scale)
  }

  override def updateParameters(learningRate: T): Unit = {
    rnn.updateParameters(learningRate)
  }

  override def zeroGradParameters(): Unit = {
    rnn.zeroGradParameters()
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    rnn.parameters()
  }

  override def getParametersTable(): Table = {
    parallelTable.getParametersTable()
  }

  override def toString(): String = {
    var str = "nn.RnnCell"
    str
  }
}

object RnnCell {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputSize: Int = 4,
    hiddenSize: Int = 3,
    activation: AbstractModule[_, _, T])
    (implicit ev: TensorNumeric[T]) : RnnCell[T] = {
    new RnnCell[T](inputSize, hiddenSize, activation)
  }
}
