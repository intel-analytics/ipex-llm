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
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.optim.Regularizer
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
 * @param  p is used for [[Dropout]] probability. For more details about
 *           RNN dropouts, please refer to
 *           [RnnDrop: A Novel Dropout for RNNs in ASR]
 *           (http://www.stat.berkeley.edu/~tsmoon/files/Conference/asru2015.pdf)
 *           [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks]
 *           (https://arxiv.org/pdf/1512.05287.pdf)
 * @param activation: activation function, by default to be Tanh if not specified.
 * @param innerActivation: activation function for inner cells,
 *                       by default to be Sigmoid if not specified.
 * @param wRegularizer: instance of [[Regularizer]]
 *                    (eg. L1 or L2 regularization), applied to the input weights matrices.
 * @param uRegularizer: instance [[Regularizer]]
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
 * @param bRegularizer: instance of [[Regularizer]]
            applied to the bias.
 */
@SerialVersionUID(6717988395573528459L)
class GRU[T : ClassTag] (
  val inputSize: Int,
  val outputSize: Int,
  val p: Double = 0,
  var activation: TensorModule[T] = null,
  var innerActivation: TensorModule[T] = null,
  var wRegularizer: Regularizer[T] = null,
  var uRegularizer: Regularizer[T] = null,
  var bRegularizer: Regularizer[T] = null)(implicit ev: TensorNumeric[T])
  extends Cell[T](
    hiddensShape = Array(outputSize),
    regularizers = Array(wRegularizer, uRegularizer, bRegularizer)
  ) {

  if (activation == null) activation = Tanh[T]()
  if (innerActivation == null) innerActivation = Sigmoid[T]()
  var i2g: ModuleNode[T] = _
  var h2g: ModuleNode[T] = _
  val featDim = 2

  override var cell: AbstractModule[Activity, Activity, T] = buildModel()

  override var preTopology: TensorModule[T] =
  if (p != 0) { null } else {
    Linear(inputSize, 3 * outputSize,
      wRegularizer = wRegularizer, bRegularizer = bRegularizer)
  }

  override def hiddenSizeOfPreTopo: Int = 3 * outputSize

  def buildGates()(input1: ModuleNode[T], input2: ModuleNode[T])
  : (ModuleNode[T], ModuleNode[T]) = {
    if (p != 0) {
      val dropi2g1 = Dropout(p).inputs(input1)
      val dropi2g2 = Dropout(p).inputs(input1)
      val lineari2g1 = Linear(inputSize, outputSize,
        wRegularizer = wRegularizer, bRegularizer = bRegularizer).inputs(dropi2g1)
      val lineari2g2 = Linear(inputSize, outputSize,
        wRegularizer = wRegularizer, bRegularizer = bRegularizer).inputs(dropi2g2)
      i2g = JoinTable(2, 0).inputs(lineari2g1, lineari2g2)

      val droph2g1 = Dropout(p).inputs(input2)
      val droph2g2 = Dropout(p).inputs(input2)
      val linearh2g1 = Linear(outputSize, outputSize,
        wRegularizer = wRegularizer, bRegularizer = bRegularizer, withBias = false).inputs(droph2g1)
      val linearh2g2 = Linear(outputSize, outputSize,
        wRegularizer = wRegularizer, bRegularizer = bRegularizer, withBias = false).inputs(droph2g2)
      h2g = JoinTable(2, 0).inputs(linearh2g1, linearh2g2)
    } else {
      i2g = Narrow[T](featDim, 1, 2 * outputSize).inputs(input1)
      h2g = Linear(outputSize, 2 * outputSize, withBias = false,
        wRegularizer = uRegularizer).inputs(input2)
    }

    val cadd = CAddTable(true).inputs(i2g, h2g)

    val narrow1 = Narrow[T](featDim, 1, outputSize).inputs(cadd)
    val narrow2 = Narrow[T](featDim, 1 + outputSize, outputSize).inputs(cadd)

    val innerActivation2 = innerActivation.cloneModule()
    innerActivation2.setName(Integer.toHexString(java.util.UUID.randomUUID().hashCode()))

    val inner1 = innerActivation.inputs(narrow1)
    val inner2 = innerActivation2.inputs(narrow2)

    (inner1, inner2)
  }

  def buildModel(): Graph[T] = {
    val x = Input()
    val h = Input()
    val (r, z) = buildGates()(x, h) // x(t), h(t - 1), r(t), z(t)

    val f2g = if (p != 0) {
      val drop1 = Dropout(p).inputs(x)
      val linear1 = Linear(inputSize, outputSize,
        wRegularizer = wRegularizer, bRegularizer = bRegularizer).inputs(drop1)
      linear1
    } else {
      Narrow(featDim, 1 + 2 * outputSize, outputSize).inputs(x)
    }

    // h_hat = tanh(f2g + Linear(r*h))
    val cMult = CMulTable().inputs(h, r)
    val drop2 = Dropout(p).inputs(cMult)
    val linear2 = Linear(outputSize, outputSize, withBias = false,
               wRegularizer = uRegularizer).inputs(drop2)

    val cadd2 = CAddTable(true).inputs(f2g, linear2)
    val h_hat = activation.inputs(cadd2)

    // h_t (1 - z) * h + z * h_hat
    val mulConst = MulConstant(-1).inputs(z)
    val addConst = AddConstant(1).inputs(mulConst)
    val cMult2 = CMulTable().inputs(h_hat, addConst)

    val cMult3 = CMulTable().inputs(h, z)

    val cadd3 = CAddTable(false).inputs(cMult2, cMult3)

    val out1 = Identity().inputs(cadd3)
    val out2 = Identity().inputs(cadd3)

    Graph(Array(x, h), Array(out1, out2))
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
    outputSize: Int = 3,
    p: Double = 0,
    activation: TensorModule[T] = null,
    innerActivation: TensorModule[T] = null,
    wRegularizer: Regularizer[T] = null,
    uRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null)(implicit ev: TensorNumeric[T]): GRU[T] = {
    new GRU[T](inputSize, outputSize, p, activation, innerActivation,
      wRegularizer, uRegularizer, bRegularizer)
  }
}
