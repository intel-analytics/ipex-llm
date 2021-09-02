/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.api.keras.layers

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn.{CAddTable, CMulTable, Cell, ConcatTable, Contiguous,
FlattenTable, NarrowTable, ParallelTable, Sigmoid, VolumetricConvolution, Tanh,
SelectTable => BSelectTable, Sequential => BSequential, Identity => BIdentity,
CMul => BCMul}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.pipeline.api.keras.layers.internal.InternalTimeDistributed

import scala.reflect.ClassTag

class InternalConvLSTM3D[T : ClassTag](val inputSize: Int,
  val outputSize: Int,
  val kernel: Int,
  val stride: Int = 1,
  val padding: Int = 0,
  var wRegularizer: Regularizer[T] = null,
  var uRegularizer: Regularizer[T] = null,
  var bRegularizer: Regularizer[T] = null,
  var cRegularizer: Regularizer[T] = null,
  val withPeephole: Boolean = true
  )(implicit ev: TensorNumeric[T]) extends Cell[T](
  hiddensShape = Array(outputSize, outputSize),
  regularizers = Array(wRegularizer, uRegularizer, bRegularizer, cRegularizer)
) {
  var inputGate: BSequential[T] = _
  var forgetGate: BSequential[T] = _
  var outputGate: BSequential[T] = _
  var hiddenLayer: BSequential[T] = _
  var cellLayer: BSequential[T] = _

  override var preTopology: TensorModule[T] = null
  override var cell: AbstractModule[Activity, Activity, T] = buildModel()

  def buildGate(name: String = null): BSequential[T] = {
    val i2g = BSequential()
      .add(Contiguous())
      .add(VolumetricConvolution(inputSize, outputSize, kernel, kernel, kernel,
        stride, stride, stride, padding, padding, padding, wRegularizer = wRegularizer,
        bRegularizer = bRegularizer).setName(name + "_i2g"))
    val h2g = BSequential()
      .add(Contiguous())
      .add(VolumetricConvolution(outputSize, outputSize, kernel, kernel, kernel,
        1, 1, 1, -1, -1, -1, wRegularizer = uRegularizer,
        withBias = false).setName(name + "_h2g"))

    val gate = BSequential()
    if (withPeephole) {
      gate
        .add(ParallelTable()
          .add(i2g)
          .add(h2g)
          .add(BCMul(Array(1, outputSize, 1, 1, 1), cRegularizer)))
    } else {
      gate.add(NarrowTable(1, 2))
      gate
        .add(ParallelTable()
          .add(i2g)
          .add(h2g))
    }

    gate.add(CAddTable())
      .add(Sigmoid())
  }

  def buildInputGate(): BSequential[T] = {
    inputGate = buildGate("InputGate")
    inputGate
  }

  def buildForgetGate(): BSequential[T] = {
    forgetGate = buildGate("ForgetGate")
    forgetGate
  }

  def buildOutputGate(): BSequential[T] = {
    outputGate = buildGate("OutputGate")
    outputGate
  }

  def buildHidden(): BSequential[T] = {
    val hidden = BSequential()
      .add(NarrowTable(1, 2))

    val i2h = BSequential()
      .add(Contiguous())
      .add(VolumetricConvolution(inputSize, outputSize, kernel, kernel, kernel,
        stride, stride, stride, padding, padding, padding, wRegularizer = wRegularizer,
        bRegularizer = bRegularizer).setName("Hidden_i2h"))
    val h2h = BSequential()
      .add(Contiguous())
      .add(VolumetricConvolution(outputSize, outputSize, kernel, kernel, kernel,
        1, 1, 1, -1, -1, -1, withBias = false,
        wRegularizer = uRegularizer).setName("Hidden_h2h"))

    hidden
      .add(ParallelTable()
        .add(i2h)
        .add(h2h))
      .add(CAddTable())
      .add(Tanh())

    this.hiddenLayer = hidden
    hidden
  }

  def buildCell(): BSequential[T] = {
    buildInputGate()
    buildForgetGate()
    buildHidden()

    val forgetLayer = BSequential()
      .add(ConcatTable()
        .add(forgetGate)
        .add(BSelectTable(3)))
      .add(CMulTable())

    val inputLayer = BSequential()
      .add(ConcatTable()
        .add(inputGate)
        .add(hiddenLayer))
      .add(CMulTable())

    val cellLayer = BSequential()
      .add(ConcatTable()
        .add(forgetLayer)
        .add(inputLayer))
      .add(CAddTable())

    this.cellLayer = cellLayer
    cellLayer
  }

  def buildModel(): BSequential[T] = {
    buildCell()
    buildOutputGate()

    val convlstm = BSequential()
      .add(FlattenTable())
      .add(ConcatTable()
        .add(NarrowTable(1, 2))
        .add(cellLayer))
      .add(FlattenTable())

      .add(ConcatTable()
        .add(BSequential()
          .add(ConcatTable()
            .add(outputGate)
            .add(BSequential()
              .add(BSelectTable(3))
              .add(Tanh())))
          .add(CMulTable())
          .add(Contiguous()))
        .add(BSelectTable(3)))

      .add(ConcatTable()
        .add(BSelectTable(1))
        .add(BIdentity()))

    convlstm
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[InternalConvLSTM3D[T]]

  override def equals(other: Any): Boolean = other match {
    case that: InternalConvLSTM3D[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        inputSize == that.inputSize &&
        outputSize == that.outputSize &&
        kernel == that.kernel &&
        stride == that.stride

    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), inputSize, outputSize, kernel, stride)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }

  override def reset(): Unit = {
    super.reset()
    cell.reset()
  }

  override def toString: String = s"InternalConvLSTM3D($inputSize, $outputSize," +
    s"$kernel, $stride)"

  def getOutputSize(sizes: Array[Int]): Array[Int] = {
    val conv = InternalTimeDistributed[T](VolumetricConvolution(inputSize, outputSize,
      kernel, kernel, kernel, stride, stride, stride, padding, padding, padding,
      wRegularizer = wRegularizer, bRegularizer = bRegularizer)
      .asInstanceOf[AbstractModule[Activity, Tensor[T], T]])
    val input = Tensor[T](sizes)
    val output = conv.forward(input)
    output.size()
  }
}
