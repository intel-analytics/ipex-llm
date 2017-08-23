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
class MultiCell[T : ClassTag](cells: Array[Cell[T]])(implicit ev: TensorNumeric[T])
  extends Cell[T](hiddensShape = cells.last.hiddensShape) {
  // inputDim and hidDim must be the same with Recurrent
  private val inputDim = 1
  private val hidDim = 2

  override var cell: AbstractModule[Activity, Activity, T] = Sequential[T]()
  cells.foreach(x => cell.asInstanceOf[Sequential[T]].add(x))
  
  var states: Array[Activity] = null

  override def updateOutput(input: Activity): Activity = {
    var i = 0
    var result = T()
    result = input.toTable
    while (i < cells.length) {
      result(hidDim) = states(i)
      result = cells(i).forward(result).toTable
      i += 1
    }

    this.output = result.toTable[Tensor[T]](inputDim)
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    var i = cells.length - 1
    var error = gradOutput
    var nextInput = T()
    while (i > 0) {
      nextInput = cells(i - 1).output.toTable
      nextInput(hidDim) = states(i)
      error = cells(i).updateGradInput(nextInput, error)
      i -= 1
    }
    nextInput = input.toTable
    error = cells(0).updateGradInput(nextInput, error)

    this.gradInput = error
    gradInput
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    var i = cells.length - 1
    var currentModule = cells(i)
    var currentGradOutput = gradOutput
    var nextInput = T()
    while (i > 0) {
      val previousModule = cells(i - 1)
      nextInput = previousModule.output.toTable
      nextInput(hidDim) = states(i)
      currentModule.accGradParameters(nextInput, currentGradOutput)
      currentGradOutput = currentModule.gradInput
      currentModule = previousModule
      i -= 1
    }
    nextInput = input.toTable
    currentModule.accGradParameters(nextInput, currentGradOutput)
  }

  override def backward(input: Activity, gradOutput: Activity): Activity = {
    var i = cells.length - 1
    var error = gradOutput
    while (i > 0) {
      val input = cells(i - 1).output
      error = cells(i).backward(input, error)
      i -= 1
    }
    error = cells(0).backward(input, error)

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
