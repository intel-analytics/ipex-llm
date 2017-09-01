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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer.{DataConverter, ModuleData, ModuleSerializable, ModuleSerializer}
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * Enable user stack multiple simple cells.
 */
class MultiCell[T : ClassTag](val cells: Array[Cell[T]])(implicit ev: TensorNumeric[T])
  extends Cell[T](
    hiddensShape = cells.last.hiddensShape,
    regularizers = cells.flatMap(_.regularizers)) {
  // inputDim and hidDim must be the same with Recurrent
  private val inputDim = 1
  private val hidDim = 2

  override var cell: AbstractModule[Activity, Activity, T] = _
  
  var states: Array[Activity] = null

  /**
   * resize the hidden parameters wrt the batch size, hiddens shapes.
   *
   * e.g. RnnCell contains 1 hidden parameter (H), thus it will return Tensor(size)
   *      LSTM contains 2 hidden parameters (C and H) and will return T(Tensor(), Tensor())
   *      and recursively intialize all the tensors in the Table.
   * @param batchSize batchSize
   * @return
   */
  def hidResize2(batchSize: Int, imageSize: Array[Int] = null): Array[Activity] = {
    var i = 0
    while(i < cells.length) {
      states(i) = cells(i).hidResize(states(i), batchSize, imageSize)
      i += 1
    }
    states
  }
  
  override def updateOutput(input: Table): Table = {
    require(states != null, "state of multicell cannot be null")
    var i = 0
    var result = T()
    result(inputDim) = input(inputDim)
    while (i < cells.length) {
      result(hidDim) = states(i)
      result = cells(i).forward(result).toTable
      states(i) = result(hidDim)
      i += 1
    }

    this.output = result
    output
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    var i = cells.length - 1
    var error = T()
    val nextInput = T()
    error(inputDim) = gradOutput(inputDim)
    while (i > 0) {
      nextInput(inputDim) = cells(i - 1).output.toTable(inputDim)
      nextInput(hidDim) = states(i)
      error(hidDim) = states(i)
      error = cells(i).updateGradInput(nextInput, error)
      i -= 1
    }
    nextInput(inputDim) = input(inputDim)
    nextInput(hidDim) = states(0)
    error(hidDim) = states(0)
    error = cells(0).updateGradInput(nextInput, error)

    this.gradInput = error
    gradInput
  }

  override def accGradParameters(input: Table, gradOutput: Table): Unit = {
    var i = cells.length - 1
    var currentGradOutput = T()
    currentGradOutput(inputDim) = gradOutput(inputDim)
    val nextInput = T()
    while (i > 0) {
      val previousModule = cells(i - 1)
      nextInput(inputDim) = previousModule.output.toTable(inputDim)
      nextInput(hidDim) = states(i)
      currentGradOutput(hidDim) = states(i)
      cells(i).accGradParameters(nextInput, currentGradOutput)
      currentGradOutput = cells(i).gradInput.toTable(inputDim)
      i -= 1
    }
    nextInput(inputDim) = input(inputDim)
    nextInput(hidDim) = states(0)
    currentGradOutput(hidDim) = states(0)
    cells(0).accGradParameters(nextInput, currentGradOutput)
  }

  override def backward(input: Table, gradOutput: Table): Table = {
    var i = cells.length - 1
    var error = T()
    error(inputDim) = gradOutput(inputDim)
    val nextInput = T()
    while (i > 0) {
      nextInput(inputDim) = cells(i - 1).output.toTable(inputDim)
      nextInput(hidDim) = states(i)
      error(hidDim) = states(i)
      error = cells(i).backward(nextInput, error)
      i -= 1
    }
    nextInput(inputDim) = input(inputDim)
    nextInput(hidDim) = states(0)
    error(hidDim) = states(0)
    error = cells(0).backward(nextInput, error)

    this.gradInput = error
    gradInput
  }

  override def zeroGradParameters(): Unit = {
    cells.foreach(_.zeroGradParameters())
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    val weights = new ArrayBuffer[Tensor[T]]()
    val gradWeights = new ArrayBuffer[Tensor[T]]()
    cells.foreach(m => {
      val params = m.parameters()
      if (params != null) {
        params._1.foreach(weights += _)
        params._2.foreach(gradWeights += _)
      }
    })
    (weights.toArray, gradWeights.toArray)
  }

  override def getParametersTable(): Table = {
    val pt = T()
    cells.foreach(m => {
      val params = m.getParametersTable()
      if (params != null) {
        params.keySet.foreach(key => pt(key) = params(key))
      }
    })
    pt
  }

  override def reset(): Unit = {
    cells.foreach(_.reset())
  }
}

object MultiCell {
  def apply[@specialized(Float, Double) T: ClassTag](cells: Array[Cell[T]]
    )(implicit ev: TensorNumeric[T]): MultiCell[T] = {
    new MultiCell[T](cells)
  }
}
