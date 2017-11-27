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
import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * Implementation of vanilla recurrent neural network cell
 * i2h: weight matrix of input to hidden units
 * h2h: weight matrix of hidden units to themselves through time
 * The updating is defined as:
 * h_t = f(i2h * x_t + h2h * h_{t-1})
 *
 * @param inputSize input size
 * @param hiddenSize hidden layer size
 * @param activation activation function f for non-linearity
 * @param isInputWithBias boolean
 * @param isHiddenWithBias boolean
 * @param wRegularizer: instance of [[Regularizer]]
 *                    (eg. L1 or L2 regularization), applied to the input weights matrices.
 * @param uRegularizer: instance [[Regularizer]]
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
 * @param bRegularizer: instance of [[Regularizer]](../regularizers.md),
            applied to the bias.
 */
class RnnCell[T : ClassTag] (
  val inputSize: Int = 4,
  val hiddenSize: Int = 3,
  activation: TensorModule[T],
  val isInputWithBias: Boolean = true,
  val isHiddenWithBias: Boolean = true,
  var wRegularizer: Regularizer[T] = null,
  var uRegularizer: Regularizer[T] = null,
  var bRegularizer: Regularizer[T] = null)
  (implicit ev: TensorNumeric[T])
  extends Cell[T](Array(hiddenSize),
    regularizers = Array(wRegularizer, uRegularizer, bRegularizer)) {

  override var preTopology: TensorModule[T] =
    Linear[T](inputSize,
      hiddenSize,
      wRegularizer = wRegularizer,
      bRegularizer = bRegularizer,
      withBias = isInputWithBias)

  override var cell: AbstractModule[Activity, Activity, T] = buildModel()

  def buildModel(): Graph[T] = {
    val i2h = Input()
    val h2h = Linear[T](hiddenSize, hiddenSize,
      wRegularizer = uRegularizer, withBias = isHiddenWithBias).inputs()
    val add = CAddTable[T](false).inputs(i2h, h2h)
    val activate = activation.inputs(add)
    val out1 = Identity[T].inputs(activate)
    val out2 = Identity[T].inputs(activate)
    Graph(Array(i2h, h2h), Array(out1, out2))
  }

  /**
   * Clear cached activities to save storage space or network bandwidth. Note that we use
   * Tensor.set to keep some information like tensor share
   *
   * The subclass should override this method if it allocate some extra resource, and call the
   * super.clearState in the override method
   *
   * @return
   */
  override def clearState(): RnnCell.this.type = {
    super.clearState()
    cell.clearState()
    this
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[RnnCell[T]]

  override def equals(other: Any): Boolean = other match {
    case that: RnnCell[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        cell == that.cell
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), cell)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object RnnCell {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputSize: Int = 4,
    hiddenSize: Int = 3,
    activation: TensorModule[T],
    isInputWithBias: Boolean = true,
    isHiddenWithBias: Boolean = true,
    wRegularizer: Regularizer[T] = null,
    uRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null)
   (implicit ev: TensorNumeric[T]) : RnnCell[T] = {
    new RnnCell[T](
      inputSize,
      hiddenSize,
      activation,
      isInputWithBias,
      isHiddenWithBias,
      wRegularizer,
      uRegularizer,
      bRegularizer)
  }
}
