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

import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
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
 * @param stride The step of the convolution
 * @param wRegularizer: instance of [[Regularizer]]
            (eg. L1 or L2 regularization), applied to the input weights matrices.
 * @param uRegularizer: instance [[Regularizer]]
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
 * @param bRegularizer: instance of [[Regularizer]]
            applied to the bias.
 * @param withPeephole: whether use last cell status control a gate.
 */
class ConvLSTMPeephole2D[T : ClassTag](
  val inputSize: Int,
  val outputSize: Int,
  val kernelI: Int,
  val kernelC: Int,
  val stride: Int,
  var wRegularizer: Regularizer[T] = null,
  var uRegularizer: Regularizer[T] = null,
  var bRegularizer: Regularizer[T] = null,
  val withPeephole: Boolean = true
)(implicit ev: TensorNumeric[T])
  extends Cell[T](
    hiddensShape = Array(outputSize, outputSize),
    regularizers = Array(wRegularizer, uRegularizer, bRegularizer)
  ) {
  var inputGate: ModuleNode[T] = _
  var forgetGate: ModuleNode[T] = _
  var outputGate: ModuleNode[T] = _
  var hiddenLayer: ModuleNode[T] = _
  var cellLayer: ModuleNode[T] = _

  override var cell: AbstractModule[Activity, Activity, T] =
  Sequential()
    .add(FlattenTable())
    .add(buildConvLSTM())
    .add(ConcatTable()
      .add(SelectTable(1))
      .add(NarrowTable(2, 2)))

  def buildGate(input1: ModuleNode[T], input2: ModuleNode[T], input3: ModuleNode[T])
  : ModuleNode[T] = {

    /**
     * i2g = Contiguous(Conv(input1))
     * h2g = Conv(input2)
     * cMul = CMul(input3)
     * Sigmoid(i2g + h2g + cMul)
     */

    val conti = Contiguous().inputs(input1)
    val i2g = SpatialConvolution(inputSize, outputSize, kernelI, kernelI,
      stride, stride, kernelI/2, kernelI/2, wRegularizer = wRegularizer,
      bRegularizer = bRegularizer).inputs(conti)
    val h2g = SpatialConvolution(outputSize, outputSize, kernelC, kernelC,
      stride, stride, kernelC/2, kernelC/2, withBias = false,
      wRegularizer = uRegularizer).inputs(input2)
    val cadd = if (withPeephole) {
      val cMul = CMul(Array(1, outputSize, 1, 1)).inputs(input3)
      CAddTable().inputs(i2g, h2g, cMul)
    } else {
      CAddTable().inputs(i2g, h2g)
    }
    val sigmoid = Sigmoid().inputs(cadd)
    sigmoid
  }

  def buildInputGate(input1: ModuleNode[T], input2: ModuleNode[T], input3: ModuleNode[T])
  : ModuleNode[T] = {
    inputGate = buildGate(input1, input2, input3)
    inputGate
  }

  def buildForgetGate(input1: ModuleNode[T], input2: ModuleNode[T], input3: ModuleNode[T])
  : ModuleNode[T] = {
    forgetGate = buildGate(input1, input2, input3)
    forgetGate
  }

  def buildOutputGate(input1: ModuleNode[T], input2: ModuleNode[T], input3: ModuleNode[T])
  : ModuleNode[T] = {
    outputGate = buildGate(input1, input2, input3)
    outputGate
  }

  def buildHidden(input1: ModuleNode[T], input2: ModuleNode[T])
  : ModuleNode[T] = {

    /**
     * i2g = Contiguous(Conv(input1))
     * h2g = Conv(input2)
     * Tanh(i2g + h2g)
     */

    val conti = Contiguous().inputs(input1)
    val i2h = SpatialConvolution(inputSize, outputSize, kernelI, kernelI,
      stride, stride, kernelI/2, kernelI/2, wRegularizer = wRegularizer,
      bRegularizer = bRegularizer).inputs(conti)
    val h2h = SpatialConvolution(outputSize, outputSize, kernelC, kernelC,
      stride, stride, kernelC/2, kernelC/2, withBias = false,
      wRegularizer = uRegularizer).inputs(input2)

    val cadd = CAddTable().inputs(i2h, h2h)
    val tanh = Tanh().inputs(cadd)

    this.hiddenLayer = tanh
    tanh
  }

  def buildCell(input1: ModuleNode[T], input2: ModuleNode[T], input3: ModuleNode[T])
  : ModuleNode[T] = {
    buildInputGate(input1, input2, input3)
    buildForgetGate(input1, input2, input3)
    buildHidden(input1, input2)

    val forgetLayer = CMulTable().inputs(forgetGate, input3)

    val inputLayer = CMulTable().inputs(inputGate, hiddenLayer)

    val cellLayer = CAddTable().inputs(forgetLayer, inputLayer)

    this.cellLayer = cellLayer
    cellLayer
  }

  def buildConvLSTM(): Graph[T] = {
    val input1 = Input()
    val input2 = Input()
    val input3 = Input()

    buildCell(input1, input2, input3)
    buildOutputGate(input1, input2, cellLayer)

    val tanh = Tanh().inputs(cellLayer)
    val cMul = CMulTable().inputs(outputGate, tanh)
    val conti = Contiguous().inputs(cMul)

    val out1 = Identity().inputs(conti)
    val out2 = Identity().inputs(conti)
    val out3 = cellLayer

    Graph(Array(input1, input2, input3), Array(out1, out2, out3))
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[ConvLSTMPeephole2D[T]]

  override def equals(other: Any): Boolean = other match {
    case that: ConvLSTMPeephole2D[T] =>
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

object ConvLSTMPeephole2D {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputSize: Int,
    outputSize: Int,
    kernelI: Int,
    kernelC: Int,
    stride: Int = 1,
    wRegularizer: Regularizer[T] = null,
    uRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    withPeephole: Boolean = true
  )(implicit ev: TensorNumeric[T]): ConvLSTMPeephole2D[T] = {
    new ConvLSTMPeephole2D[T](inputSize, outputSize, kernelI, kernelC, stride,
      wRegularizer, uRegularizer, bRegularizer, withPeephole)
  }
}
