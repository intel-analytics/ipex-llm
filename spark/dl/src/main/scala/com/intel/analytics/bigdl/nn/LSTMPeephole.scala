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
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
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
  var wRegularizer: Regularizer[T] = null,
  var uRegularizer: Regularizer[T] = null,
  var bRegularizer: Regularizer[T] = null
)
  (implicit ev: TensorNumeric[T])
  extends Cell[T](
    hiddensShape = Array(hiddenSize, hiddenSize),
    regularizers = Array(wRegularizer, uRegularizer, bRegularizer)
  ) {
  var inputGate: ModuleNode[T] = _
  var forgetGate: ModuleNode[T] = _
  var outputGate: ModuleNode[T] = _
  var hiddenLayer: ModuleNode[T] = _
  var cellLayer: ModuleNode[T] = _
  val featDim = 2

  override var cell: AbstractModule[Activity, Activity, T] =
    Sequential()
      .add(FlattenTable())
      .add(buildLSTM())
      .add(ConcatTable()
        .add(SelectTable(1))
        .add(NarrowTable(2, 2)))

  override var preTopology: TensorModule[T] = if (p != 0) {
    null
  } else {
    Linear(inputSize, 4 * hiddenSize,
      wRegularizer = wRegularizer, bRegularizer = bRegularizer)
  }

  override def hiddenSizeOfPreTopo: Int = hiddenSize * 4

  def buildGate(dimension: Int, offset: Int, length: Int)
               (input1: ModuleNode[T], input2: ModuleNode[T], input3: ModuleNode[T])
  : ModuleNode[T] = {

    /**
     * f(input1 + U * input2)
     */
    var i2g: ModuleNode[T] = null
    var h2g: ModuleNode[T] = null
    if (p != 0) {
      val input1Drop = Dropout(p).inputs(input1)
      i2g = Linear(inputSize, hiddenSize, wRegularizer = wRegularizer,
              bRegularizer = bRegularizer).inputs(input1Drop)
      val input2Drop = Dropout(p).inputs(input2)
      h2g = Linear(hiddenSize, hiddenSize, withBias = false,
          wRegularizer = uRegularizer).inputs(input2Drop)
    } else {
      i2g = Narrow(dimension, offset, length).inputs(input1)
      h2g = Linear(hiddenSize, hiddenSize,
        withBias = false, wRegularizer = uRegularizer).inputs(input2)
    }
    val cMul = CMul(Array(hiddenSize)).inputs(input3)

    val cadd = CAddTable().inputs(i2g, h2g, cMul)
    val sigmoid = Sigmoid().inputs(cadd)

    sigmoid
  }

  def buildInputGate()
                    (input1: ModuleNode[T], input2: ModuleNode[T], input3: ModuleNode[T])
  : ModuleNode[T] = {
    inputGate = buildGate(featDim, 1, hiddenSize)(input1, input2, input3)
    inputGate
  }

  def buildForgetGate()
                     (input1: ModuleNode[T], input2: ModuleNode[T], input3: ModuleNode[T])
  : ModuleNode[T] = {
    forgetGate =
      buildGate(featDim, 1 + hiddenSize, hiddenSize)(input1, input2, input3)
    forgetGate
  }

  def buildOutputGate()
                     (input1: ModuleNode[T], input2: ModuleNode[T], input3: ModuleNode[T])
  : ModuleNode[T] = {
    outputGate =
      buildGate(featDim, 1 + 3 * hiddenSize, hiddenSize)(input1, input2, input3)
    outputGate
  }

  def buildHidden()
                 (input1: ModuleNode[T], input2: ModuleNode[T])
  : ModuleNode[T] = {

    /**
     * f(input1 + W * input2)
     */
    var i2h: ModuleNode[T] = null
    var h2h: ModuleNode[T] = null
    if (p != 0) {
      val input1Drop = Dropout(p).inputs(input1)
      i2h = Linear(inputSize, hiddenSize, wRegularizer = wRegularizer,
          bRegularizer = bRegularizer).inputs(input1Drop)

      val input2Drop = Dropout(p).inputs(input2)
      h2h = Linear(hiddenSize, hiddenSize, withBias = false,
          wRegularizer = uRegularizer).inputs(input2Drop)
    } else {
      i2h = Narrow(featDim, 1 + 2 * hiddenSize, hiddenSize).inputs(input1)
      h2h = Linear(hiddenSize, hiddenSize, withBias = false,
        wRegularizer = uRegularizer).inputs(input2)
    }
    val cadd = CAddTable().inputs(i2h, h2h)

    val tanh = Tanh().inputs(cadd)

    this.hiddenLayer = tanh
    tanh
  }

  def buildCell()
               (input1: ModuleNode[T], input2: ModuleNode[T], input3: ModuleNode[T])
  : ModuleNode[T] = {
    buildInputGate()(input1, input2, input3)
    buildForgetGate()(input1, input2, input3)
    buildHidden()(input1, input2)

    val forgetLayer = CMulTable().inputs(forgetGate, input3)

    val inputLayer = CMulTable().inputs(inputGate, hiddenLayer)

    val cellLayer = CAddTable().inputs(forgetLayer, inputLayer)

    this.cellLayer = cellLayer
    cellLayer
  }

  def buildLSTM(): Graph[T] = {
    val input1 = Input()
    val input2 = Input()
    val input3 = Input()

    /**
     * f: sigmoid
     * g: tanh
     * forgetLayer = input3 * f(input1 + U1 * input2)
     * inputLayer = f(input1 + U2 * input2) * g(input1 + U3 * input2)
     * cellLayer = forgetLayer + inputLayer
     */
    buildCell()(input1, input2, input3)
    buildOutputGate()(input1, input2, cellLayer)

    val tanh = Tanh().inputs(cellLayer)
    val cMul = CMulTable().inputs(outputGate, tanh)

    val out1 = Identity().inputs(cMul)
    val out2 = Identity().inputs(cMul)
    val out3 = cellLayer

    /**
     * out1 = outputGate * g(cellLayer)
     * out2 = out1
     * out3 = cellLayer
     */
    Graph(Array(input1, input2, input3), Array(out1, out2, out3))
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
    new LSTMPeephole[T](inputSize, hiddenSize, p, wRegularizer, uRegularizer,
      bRegularizer)
  }
}

