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
  private val inputDim = Recurrent.inputDim
  private val hidDim = Recurrent.hidDim

  override var preTopology: AbstractModule[Activity, Activity, T] = null
  
  override var cell: AbstractModule[Activity, Activity, T] = buildModel()

  var states: Array[Activity] = null
  var gradStates: Array[Activity] = null
  
  var state0: Activity = null
  
  def buildModel(): Sequential[T] = {
    val seq = Sequential()
    cells.foreach{ cell =>
      if (cell.preTopology != null) {
        seq.add(cell.preTopology)
      }
      seq.add(cell)
    }
    seq
  }
  
  override def updateOutput(input: Table): Table = {
    require(states != null, "state of multicell cannot be null")
    var i = 0
    val result = T()
    result(inputDim) = input(inputDim)
    state0 = states.head
    
    while (i < cells.length) {
      result(hidDim) = states(i)
      
      if (cells(i).preTopology != null) {
//        val inputTmp = result(inputDim).asInstanceOf[Tensor[T]].clone()
//        val sizes = 1 +: inputTmp.size()
//        inputTmp.resize(sizes)
//        val outputTmp = cells(i).preTopology.forward(inputTmp).toTensor[T]
//        result(inputDim) = outputTmp.select(1, 1)
        result(inputDim) = cells(i).preTopology.forward(result(inputDim)).toTensor[T]
      }
      cells(i).forward(result).toTable
      // propogate state for next time step
      states(i) = cells(i).output.toTable(hidDim)
      result(inputDim) = cells(i).output.toTable(inputDim)
      i += 1
    }

    this.output = result
    output
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    throw new Exception("Should not enter MultiCell updateGradInput since backward is override")
    gradInput
  }

  override def accGradParameters(input: Table, gradOutput: Table): Unit = {
    throw new Exception("Should not enter MultiCell accGradParameters since backward is override")
  }

  override def backward(input: Table, gradOutput: Table): Table = {
    var i = cells.length
    var error = T()
    error(inputDim) = gradOutput(inputDim)
    
    val nextInput = T()
    while (i >= 1) {
      val input0: Tensor[T] = if (i > 1) {
        cells(i - 2).output.toTable(inputDim)
      } else input(inputDim)
      nextInput(inputDim) = if (cells(i - 1).preTopology != null) {
        cells(i - 1).preTopology.forward(input0)
      } else input0
        
      nextInput(hidDim) = if (i == 1) state0 else states(i - 2)
      error(hidDim) = gradStates(i - 1)
      error = cells(i - 1).backward(nextInput, error)
      gradStates(i - 1) = error(hidDim)

      if (cells(i - 1).preTopology != null) {
        error(inputDim) = cells(i - 1).preTopology.backward(input0,
          cells(i - 1).gradInput.toTable[Tensor[T]](inputDim))
      }
      i -= 1
    }

    this.gradInput = error
    gradInput
  }

  override def zeroGradParameters(): Unit = {
    cells.foreach(_.zeroGradParameters())
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
