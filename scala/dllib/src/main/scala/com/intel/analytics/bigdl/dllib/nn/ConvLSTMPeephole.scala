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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

/**
 * Convolution Long Short Term Memory architecture with peephole.
 * Ref. A.: https://arxiv.org/abs/1506.04214 (blueprint for this module)
 * B. https://github.com/viorik/ConvLSTM
 *
 * @param inputSize number of input planes in the image given into forward()
 * @param outputSize number of output planes the convolution layer will produce
 * @param kernelI Convolutional filter size to convolve input
 * @param kernelC Convolutional filter size to convolve cell
 * @param stride The step of the convolution, default is 1
 * @param padding The step of the convolution, default is -1,
 *                behaves same with SAME padding in tensorflow.
 *                Default stride,padding ensure last 2 dim of output shape is the same with input
 * @param activation: activation function, by default to be Tanh if not specified.
 * @param innerActivation: activation function for inner cells,
 *                       by default to be Sigmoid if not specified.
 * @param wRegularizer: instance of [[Regularizer]]
            (eg. L1 or L2 regularization), applied to the input weights matrices.
 * @param uRegularizer: instance [[Regularizer]]
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
 * @param bRegularizer: instance of [[Regularizer]]
            applied to the bias.
 * @param cRegularizer: instance of [[Regularizer]]
            applied to peephole.
 * @param withPeephole: whether use last cell status control a gate.
 */
class ConvLSTMPeephole[T : ClassTag](
  val inputSize: Int,
  val outputSize: Int,
  val kernelI: Int,
  val kernelC: Int,
  val stride: Int = 1,
  val padding: Int = -1,
  var activation: TensorModule[T] = null,
  var innerActivation: TensorModule[T] = null,
  var wRegularizer: Regularizer[T] = null,
  var uRegularizer: Regularizer[T] = null,
  var bRegularizer: Regularizer[T] = null,
  var cRegularizer: Regularizer[T] = null,
  val withPeephole: Boolean = true
)(implicit ev: TensorNumeric[T])
  extends Cell[T](
    hiddensShape = Array(outputSize, outputSize),
    regularizers = Array(wRegularizer, uRegularizer, bRegularizer, cRegularizer)
  ) {
  var inputGate: Sequential[T] = _
  var forgetGate: Sequential[T] = _
  var outputGate: Sequential[T] = _
  var hiddenLayer: Sequential[T] = _
  var cellLayer: Sequential[T] = _
  if (activation == null) activation = Tanh[T]()
  if (innerActivation == null) innerActivation = Sigmoid[T]()

  override var cell: AbstractModule[Activity, Activity, T] = buildModel()
//  val joinDim = 2

  override var preTopology: TensorModule[T] = null
//  override var preTopology: AbstractModule[Activity, Activity, T] = null
//  override def preTopology: AbstractModule[Activity, Activity, T] =
//    Sequential()
//        .add(TimeDistributed(SpatialConvolution(inputSize, outputSize*4, kernelI, kernelI,
//          stride, stride, kernelI/2, kernelI/2, wRegularizer = wRegularizer,
//          bRegularizer = bRegularizer)))

//  def buildGate(offset: Int, length: Int): Sequential[T] = {
//    val i2g = Narrow(joinDim, offset, length)
  def buildGate(name: String = null): Sequential[T] = {
    val i2g = Sequential()
      .add(Contiguous())
      .add(SpatialConvolution(inputSize, outputSize, kernelI, kernelI,
        stride, stride, padding, padding, wRegularizer = wRegularizer,
        bRegularizer = bRegularizer).setName(name + "_i2g"))
    val h2g = Sequential()
      .add(Contiguous())
      .add(SpatialConvolution(outputSize, outputSize, kernelC, kernelC,
      stride, stride, padding, padding, withBias = false,
      wRegularizer = uRegularizer).setName(name + "_h2g"))

    val gate = Sequential()
    if (withPeephole) {
      gate
        .add(ParallelTable()
          .add(i2g)
          .add(h2g)
          .add(CMul(Array(1, outputSize, 1, 1), cRegularizer)))
    } else {
      gate.add(NarrowTable(1, 2))
      gate
        .add(ParallelTable()
          .add(i2g)
          .add(h2g))
    }

    // make a new instance of inner activation for each build gate
    val inner = innerActivation.cloneModule()
    inner.setName(Integer.toHexString(java.util.UUID.randomUUID().hashCode()))

    gate.add(CAddTable())
      .add(inner)
  }

  def buildInputGate(): Sequential[T] = {
//    inputGate = buildGate(1 + outputSize, outputSize)
    inputGate = buildGate("InputGate")
    inputGate
  }

  def buildForgetGate(): Sequential[T] = {
//    forgetGate = buildGate(1, outputSize)
    forgetGate = buildGate("ForgetGate")
    forgetGate
  }

  def buildOutputGate(): Sequential[T] = {
//    outputGate = buildGate(1 + 3 * outputSize, outputSize)
    outputGate = buildGate("OutputGate")
    outputGate
  }

  def buildHidden(): Sequential[T] = {
    val hidden = Sequential()
      .add(NarrowTable(1, 2))

//    val i2h = Narrow(joinDim, 1 + 2 * outputSize, outputSize)
    val i2h = Sequential()
      .add(Contiguous())
      .add(SpatialConvolution(inputSize, outputSize, kernelI, kernelI,
        stride, stride, padding, padding, wRegularizer = wRegularizer,
        bRegularizer = bRegularizer).setName("Hidden_i2h"))
    val h2h = Sequential()
      .add(Contiguous())
      .add(SpatialConvolution(outputSize, outputSize, kernelC, kernelC,
      stride, stride, padding, padding, withBias = false,
      wRegularizer = uRegularizer).setName("Hidden_h2h"))

    hidden
      .add(ParallelTable()
        .add(i2h)
        .add(h2h))
      .add(CAddTable())
      .add(activation)

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

  def buildModel(): Sequential[T] = {
    buildCell()
    buildOutputGate()

    val activation2 = activation.cloneModule()
    activation2.setName(Integer.toHexString(java.util.UUID.randomUUID().hashCode()))

    val convlstm = Sequential()
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
              .add(activation2)))
          .add(CMulTable())
          .add(Contiguous()))
        .add(SelectTable(3)))

      .add(ConcatTable()
        .add(SelectTable(1))
        .add(Identity()))

    convlstm
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[ConvLSTMPeephole[T]]

  override def equals(other: Any): Boolean = other match {
    case that: ConvLSTMPeephole[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        inputSize == that.inputSize &&
        outputSize == that.outputSize &&
        kernelI == that.kernelI &&
        kernelC == that.kernelC &&
        stride == that.stride

    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), inputSize, outputSize, kernelI, kernelC, stride)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }

  override def reset(): Unit = {
    super.reset()
    cell.reset()
  }

  override def toString: String = s"ConvLSTMPeephole2D($inputSize, $outputSize," +
    s"$kernelI, $kernelC, $stride)"
}

object ConvLSTMPeephole {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputSize: Int,
    outputSize: Int,
    kernelI: Int,
    kernelC: Int,
    stride: Int = 1,
    padding: Int = -1,
    activation: TensorModule[T] = null,
    innerActivation: TensorModule[T] = null,
    wRegularizer: Regularizer[T] = null,
    uRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    cRegularizer: Regularizer[T] = null,
    withPeephole: Boolean = true
  )(implicit ev: TensorNumeric[T]): ConvLSTMPeephole[T] = {
    new ConvLSTMPeephole[T](inputSize, outputSize, kernelI, kernelC,
      stride, padding, activation, innerActivation,
      wRegularizer, uRegularizer, bRegularizer, cRegularizer, withPeephole)
  }
}

