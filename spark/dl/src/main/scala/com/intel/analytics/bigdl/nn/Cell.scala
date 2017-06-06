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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

/**
 * The Cell class is a super class of any recurrent kernels, such as
 * [[RnnCell]], [[LSTM]] and [[GRU]]. All the kernels in a recurrent
 * network should extend the [[Cell]] abstract class.
 *
 * @param hiddensShape represents the shape of hiddens which would be
 *                     transferred to the next recurrent time step.
 *                     E.g. For RnnCell, it should be Array(hiddenSize)
 *                     For LSTM, it should be Array(hiddenSize, hiddenSize)
 *                     (because each time step a LSTM return two hiddens `h` and `c` in order,
 *                     which have the same size.)
 *
 *@param regularizers If the subclass has regularizers, it need to put the regularizers into
 *                     an array and pass the array into the [[Cell]] constructor as an argument. See
 *                     [[LSTM]] as a concrete example.
 */
abstract class Cell[T : ClassTag](
  val hiddensShape: Array[Int],
  var regularizers: Array[Regularizer[T]] = null
)(implicit ev: TensorNumeric[T])
  extends AbstractModule[Table, Table, T] {

  /**
   * Any recurrent kernels should have a cell member variable which
   * represents the module in the kernel.
   *
   * The `cell` receive an input with a format of T(`input`, `preHiddens`), and
   * the output should be a format of T(`output`, `hiddens`).
   * The `hiddens` represents the kernel's output hiddens at the current time step, which will
   * be transferred to next time step. For instance, a simple [[RnnCell]], `hiddens` is h,
   * for LSTM, `hiddens` is T(h, c), and for both of them, the `output` variable represents h.
   * Similarly the `preHiddens` is the kernel's output hiddens at the previous time step.
   *
   */
  var cell: AbstractModule[Activity, Activity, T]

  /**
   * resize the hidden parameters wrt the batch size, hiddens shapes.
   *
   * e.g. RnnCell contains 1 hidden parameter (H), thus it will return Tensor(size)
   *      LSTM contains 2 hidden parameters (C and H) and will return T(Tensor(), Tensor())\
   *      and recursively intialize all the tensors in the Table.
   *
   * @param hidden
   * @param size batchSize
   * @return
   */
  def hidResize(hidden: Activity, size: Int): Activity = {
    if (hidden == null) {
      if (hiddensShape.length == 1) {
        hidResize(Tensor[T](), size)
      } else {
        val _hidden = T()
        var i = 1
        while (i <= hiddensShape.length) {
          _hidden(i) = Tensor[T]()
          i += 1
        }
        hidResize(_hidden, size)
      }
    } else {
      if (hidden.isInstanceOf[Tensor[T]]) {
        require(hidden.isInstanceOf[Tensor[T]],
          "Cell: hidden should be a Tensor")
        hidden.toTensor.resize(size, hiddensShape(0))
      } else {
        require(hidden.isInstanceOf[Table],
          "Cell: hidden should be a Table")
        var i = 1
        while (i <= hidden.toTable.length()) {
          hidden.toTable[Tensor[T]](i).resize(size, hiddensShape(i - 1))
          i += 1
        }
        hidden
      }
    }
  }

  override def updateOutput(input: Table): Table = {
    output = cell.updateOutput(input).toTable
    output
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    gradInput = cell.updateGradInput(input, gradOutput).toTable
    gradInput
  }

  override def accGradParameters(input: Table, gradOutput: Table): Unit = {
    cell.accGradParameters(input, gradOutput)
  }

  override def updateParameters(learningRate: T): Unit = {
    cell.updateParameters(learningRate)
  }

  override def zeroGradParameters(): Unit = {
    cell.zeroGradParameters()
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    cell.parameters()
  }

  override def getParametersTable(): Table = {
    cell.getParametersTable()
  }

  override def reset(): Unit = {
    cell.reset()
  }

  /**
   * Use this method to set the whether the recurrent cell
   * is regularized
   *
   * @param isRegularized whether to be regularized or not
   */
  def regluarized(
    isRegularized: Boolean
  ): Unit = {
    if (null != regularizers) {
      regularizers.foreach(x =>
        if (null != x) {
          if (isRegularized) x.enable()
          else x.disable()
        }
      )
    }
  }
}
