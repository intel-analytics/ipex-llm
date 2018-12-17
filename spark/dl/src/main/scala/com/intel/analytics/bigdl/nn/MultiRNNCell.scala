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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer._
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter.ArrayConverter
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * Enable user stack multiple simple cells.
 */
class MultiRNNCell[T : ClassTag](val cells: Array[Cell[T]])(implicit ev: TensorNumeric[T])
  extends Cell[T](
    hiddensShape = cells.last.hiddensShape,
    regularizers = cells.flatMap(_.regularizers)) {
  // inputDim and hidDim must be the same with Recurrent
  private val inputDim = Recurrent.inputDim
  private val hidDim = Recurrent.hidDim

  override var preTopology: TensorModule[T] = null

  override var cell: AbstractModule[Activity, Activity, T] = buildModel()

  override def hidResize(hidden: Activity, batchSize: Int, stepShape: Array[Int]): Activity = {
    if (hidden == null) {
      hidResize(T(), batchSize, stepShape)
    } else {
      var i = 0
      while (i < cells.size) {
        hidden.toTable.insert(cells(i).hidResize(null, batchSize, stepShape))
        i += 1
      }
      hidden
    }
  }

  def buildModel(): Sequential[T] = {
    val seq = Sequential()
    cells.foreach{ cell =>
      if (cell.preTopology != null) {
        cell.includePreTopology = true
      }
      seq.add(cell)
    }
    seq
  }

  override def updateOutput(input: Table): Table = {
    val result = T()
    result(inputDim) = input(inputDim)
    // states and outputStates is 1 based
    val states = input(hidDim).asInstanceOf[Table]
    val outputStates = T()

    var i = 0
    while (i < cells.length) {
      result(hidDim) = states(i + 1)
      cells(i).forward(result).toTable
      result(inputDim) = cells(i).output.toTable(inputDim)
      outputStates.insert(cells(i).output.toTable(hidDim))
      i += 1
    }

    result(hidDim) = outputStates
    this.output = result
    output
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    var i = cells.length - 1
    var error = T()
    error(inputDim) = gradOutput(inputDim)
    val states = input(hidDim).asInstanceOf[Table]
    val gradStates = gradOutput(hidDim).asInstanceOf[Table]
    val outputGradStates = T()

    val nextInput = T()
    while (i >= 0) {
      val input0: Tensor[T] = if (i > 0) {
        cells(i - 1).output.toTable(inputDim)
      } else input(inputDim)
      nextInput(inputDim) = input0

      nextInput(hidDim) = states(i + 1)
      error(hidDim) = gradStates(i + 1)
      error = cells(i).updateGradInput(nextInput, error)
      outputGradStates(i + 1) = error(hidDim)
      i -= 1
    }

    this.gradInput = error
    gradInput(hidDim) = outputGradStates
    gradInput
  }

  override def accGradParameters(input: Table, gradOutput: Table): Unit = {
    var i = cells.length - 1
    val error = T()
    error(inputDim) = gradOutput(inputDim)
    val states = input(hidDim).asInstanceOf[Table]
    val gradStates = gradOutput(hidDim).asInstanceOf[Table]

    val nextInput = T()
    while (i >= 0) {
      val input0: Tensor[T] = if (i > 0) {
        cells(i - 1).output.toTable(inputDim)
      } else input(inputDim)
      nextInput(inputDim) = input0

      nextInput(hidDim) = states(i + 1)
      error(hidDim) = gradStates(i + 1)
      cells(i).accGradParameters(nextInput, error)
      error(inputDim) = cells(i).gradInput.toTable(inputDim)
      i -= 1
    }
  }

  override def backward(input: Table, gradOutput: Table): Table = {
    val before = System.nanoTime()
    var i = cells.length - 1
    var error = T()
    error(inputDim) = gradOutput(inputDim)
    val states = input(hidDim).asInstanceOf[Table]
    val gradStates = gradOutput(hidDim).asInstanceOf[Table]
    val outputGradStates = T()

    val nextInput = T()
    while (i >= 0) {
      val input0: Tensor[T] = if (i > 0) {
        cells(i - 1).output.toTable(inputDim)
      } else input(inputDim)
      nextInput(inputDim) = input0

      nextInput(hidDim) = states(i + 1)
      error(hidDim) = gradStates(i + 1)
      error = cells(i).backward(nextInput, error)
      outputGradStates(i + 1) = error(hidDim)
      i -= 1
    }

    this.gradInput = error
    gradInput(hidDim) = outputGradStates
    backwardTime += System.nanoTime() - before
    gradInput
  }

  override def reset(): Unit = {
    cells.foreach(_.reset())
  }
}

object MultiRNNCell extends ModuleSerializable {
  def apply[@specialized(Float, Double) T: ClassTag](cells: Array[Cell[T]]
    )(implicit ev: TensorNumeric[T]): MultiRNNCell[T] = {
    new MultiRNNCell[T](cells)
  }

  override def doLoadModule[T: ClassTag](context : DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {

    val attrMap = context.bigdlModule.getAttrMap

    val cells = DataConverter.getAttributeValue(context, attrMap.get("cells")).
      asInstanceOf[Array[AbstractModule[_, _, T]]].map(_.asInstanceOf[Cell[T]])

    val multiRNNCell = MultiRNNCell[T](cells)

    CellSerializer.populateCellAttributes(context, multiRNNCell)
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
                                              cellModuleBuilder : BigDLModule.Builder)
                                             (implicit ev: TensorNumeric[T]) : Unit = {

    CellSerializer.saveCellAttributes(context, cellModuleBuilder)

    val cellsBuilder = AttrValue.newBuilder
    ArrayConverter.setAttributeValue(context, cellsBuilder,
      context.moduleData.module.asInstanceOf[MultiRNNCell[T]].cells,
      scala.reflect.runtime.universe.typeOf[Array[_ <:
        AbstractModule[_ <: Activity, _ <:  Activity, _ <: Any]]])
    cellModuleBuilder.putAttr("cells", cellsBuilder.build)
  }

}
