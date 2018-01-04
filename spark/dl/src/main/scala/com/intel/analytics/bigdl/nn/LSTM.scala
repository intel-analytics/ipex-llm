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
 * @param p is used for [[Dropout]] probability. For more details about
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
@SerialVersionUID(- 8176191554025511686L)
class LSTM[T : ClassTag] (
  val inputSize: Int,
  val hiddenSize: Int,
  val p: Double = 0,
  var activation: TensorModule[T] = null,
  var innerActivation: TensorModule[T] = null,
  var wRegularizer: Regularizer[T] = null,
  var uRegularizer: Regularizer[T] = null,
  var bRegularizer: Regularizer[T] = null
)
  (implicit ev: TensorNumeric[T])
  extends Cell[T](
    hiddensShape = Array(hiddenSize, hiddenSize),
    regularizers = Array(wRegularizer, uRegularizer, bRegularizer)
  ) {

  if (activation == null) activation = Tanh[T]()
  if (innerActivation == null) innerActivation = Sigmoid[T]()
  var gates: Sequential[T] = _
  var cellLayer: Sequential[T] = _

  override var cell: AbstractModule[Activity, Activity, T] = buildModel()

  override var preTopology: TensorModule[T] = if (p != 0) {
    null
  } else {
    Linear(inputSize, 4 * hiddenSize,
      wRegularizer = wRegularizer, bRegularizer = bRegularizer)
  }

  override def hiddenSizeOfPreTopo: Int = 4 * hiddenSize

  def buildGates()(input1: ModuleNode[T], input2: ModuleNode[T])
  : (ModuleNode[T], ModuleNode[T], ModuleNode[T], ModuleNode[T]) = {

    var i2g: ModuleNode[T] = null
    var h2g: ModuleNode[T] = null

    if (p != 0) {
      val dropi2g1 = Dropout(p).inputs(input1)
      val dropi2g2 = Dropout(p).inputs(input1)
      val dropi2g3 = Dropout(p).inputs(input1)
      val dropi2g4 = Dropout(p).inputs(input1)

      val lineari2g1 = Linear(inputSize, hiddenSize,
        wRegularizer = wRegularizer, bRegularizer = bRegularizer).inputs(dropi2g1)
      val lineari2g2 = Linear(inputSize, hiddenSize,
        wRegularizer = wRegularizer, bRegularizer = bRegularizer).inputs(dropi2g2)
      val lineari2g3 = Linear(inputSize, hiddenSize,
        wRegularizer = wRegularizer, bRegularizer = bRegularizer).inputs(dropi2g3)
      val lineari2g4 = Linear(inputSize, hiddenSize,
        wRegularizer = wRegularizer, bRegularizer = bRegularizer).inputs(dropi2g4)

      i2g = JoinTable(1, 1).inputs(lineari2g1, lineari2g2, lineari2g3, lineari2g4)

      val droph2g1 = Dropout(p).inputs(input2)
      val droph2g2 = Dropout(p).inputs(input2)
      val droph2g3 = Dropout(p).inputs(input2)
      val droph2g4 = Dropout(p).inputs(input2)

      val linearh2g1 = Linear(hiddenSize, hiddenSize,
        wRegularizer = wRegularizer, bRegularizer = bRegularizer).inputs(droph2g1)
      val linearh2g2 = Linear(hiddenSize, hiddenSize,
        wRegularizer = wRegularizer, bRegularizer = bRegularizer).inputs(droph2g2)
      val linearh2g3 = Linear(hiddenSize, hiddenSize,
        wRegularizer = wRegularizer, bRegularizer = bRegularizer).inputs(droph2g3)
      val linearh2g4 = Linear(hiddenSize, hiddenSize,
        wRegularizer = wRegularizer, bRegularizer = bRegularizer).inputs(droph2g4)

      h2g = JoinTable(1, 1).inputs(linearh2g1, linearh2g2, linearh2g3, linearh2g4)
    } else {
      i2g = input1
      h2g = Linear(hiddenSize, 4 * hiddenSize,
        withBias = false, wRegularizer = uRegularizer).inputs(input2)
    }

    val caddTable = CAddTable(false).inputs(i2g, h2g)
    val reshape = Reshape(Array(4, hiddenSize)).inputs(caddTable)
    val split1 = Select(2, 1).inputs(reshape)
    val split2 = Select(2, 2).inputs(reshape)
    val split3 = Select(2, 3).inputs(reshape)
    val split4 = Select(2, 4).inputs(reshape)

    // make different instances of inner activation
    val innerActivation2 = innerActivation.cloneModule()
    innerActivation2.setName(Integer.toHexString(java.util.UUID.randomUUID().hashCode()))
    val innerActivation3 = innerActivation.cloneModule()
    innerActivation3.setName(Integer.toHexString(java.util.UUID.randomUUID().hashCode()))

    (innerActivation.inputs(split1),
      activation.inputs(split2),
      innerActivation2.inputs(split3),
      innerActivation3.inputs(split4))
  }

  def buildModel(): Sequential[T] = {
    Sequential()
      .add(FlattenTable())
      .add(buildLSTM())
      .add(ConcatTable()
        .add(SelectTable(1))
        .add(NarrowTable(2, 2)))
  }

  def buildLSTM(): Graph[T] = {
    val input1 = Input()
    val input2 = Input()
    val input3 = Input()
    val (in, hid, forg, out) = buildGates()(input1, input2)

    /**
     * g: activation
     * cMult1 = in * hid
     * cMult2 = forg * input3
     * cMult3 = out * g(cMult1 + cMult2)
     */
    val cMult1 = CMulTable().inputs(in, hid)
    val cMult2 = CMulTable().inputs(forg, input3)
    val cadd = CAddTable(true).inputs(cMult1, cMult2)
    val activation2 = activation.cloneModule()
    activation2.setName(Integer.toHexString(java.util.UUID.randomUUID().hashCode()))
    val activate = activation2.inputs(cadd)
    val cMult3 = CMulTable().inputs(activate, out)

    val out1 = Identity().inputs(cMult3)
    val out2 = Identity().inputs(cMult3)
    val out3 = cadd

    /**
     * out1 = cMult3
     * out2 = out1
     * out3 = cMult1 + cMult2
     */
    Graph(Array(input1, input2, input3), Array(out1, out2, out3))
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
    hiddenSize: Int,
    p: Double = 0,
    activation: TensorModule[T] = null,
    innerActivation: TensorModule[T] = null,
    wRegularizer: Regularizer[T] = null,
    uRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null
  )
    (implicit ev: TensorNumeric[T]): LSTM[T] = {
    new LSTM[T](inputSize, hiddenSize, p, activation, innerActivation,
      wRegularizer, uRegularizer, bRegularizer)
  }
}
