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
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer.{ContainerSerializable, DataConverter, ModuleData, ModuleSerializer}
import serialization.Bigdl.{AttrValue, BigDLModule}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * [[RecurrentDecoder]] module is a container of rnn cells that used to make
 * a prediction of the next timestep based on the prediction we made from
 * the previous timestep. Input for RecurrentDecoder is dynamically composed
 * during training. input at t(i) is output at t(i-1), input at t(0) is
 * user input, and user input has to be batch x ???(depends on cell type)
 * without time information.

 * Different types of rnn cells can be added using add() function. Currently
 * only support lstmpeephole, convlstm, convlstm3D cell.
 */
class RecurrentDecoder[T : ClassTag](outputLength: Int)
  (implicit ev: TensorNumeric[T]) extends Recurrent[T] {

  times = outputLength

  /**
   *
   *  modules: topology (cell)
   *
   * The topology (or cell) will be cloned for N times w.r.t the time dimension.
   *
   * @param module module to be add
   * @return this container
   */
  override def add(module: AbstractModule[_ <: Activity, _ <: Activity, T]):
    RecurrentDecoder.this.type = {
    require(module.isInstanceOf[Cell[T]],
      "Recurrent: contained module should be Cell type")
    topology = module.asInstanceOf[Cell[T]]
    preTopology = null
    topology.ignorePreTopology = true
    topology.cell = topology.buildModel()
    modules += topology
    this
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim == 2 || input.dim == 4 || input.dim == 5,
      "Recurrent: input should be a 2D/4D/5D Tensor, e.g [batch, nDim], " +
        s"current input.dim = ${input.dim}")

    batchSize = input.size(batchDim)
    outputCell = input

    val hiddenSize = topology.hiddensShape(0)
    val outputSize = input.size()
    outputSize(1) = hiddenSize
    require(hiddenSize == input.size()(1), "hiddenSize is " +
      "not the same with input size!! Please update cell settings or use Recurrent instead!")
    val featureSizes = outputSize.drop(1)
    output.resize(Array(batchSize, times) ++ featureSizes)
    // Clone N modules along the sequence dimension.
    extend(featureSizes)

    /**
     * currentInput forms a T() type. It contains two elements, hidden and input.
     * Each time it will feed the cell with T(hidden, input) (or T(input, hidden) depends on
     * your hidDim and inputDim), and the cell will give a table output containing two
     * identical elements T(output, output). One of the elements from the cell output is
     * the updated hidden. Thus the currentInput will update its hidden element with this output.
     */
    var i = 1
    // init state
    currentInput(hidDim) = if (initState != null) initState
     else hidden

    while (i <= times) {
      if (i == 1) {
        // input at t(0) is last time step of user input
        currentInput(inputDim) = input
        cells(i - 1).forward(currentInput)
      } else {
        // input at t(i) is output at t(i-1)
        cells(i - 1).forward(cells(i - 2).output)
      }
      i += 1
    }

    copy(cells.map(x => x.output.toTable[Tensor[T]](inputDim)),
        output, 0)
    output
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    currentGradOutput(hidDim) = gradHidden
    /**
     * Since we clone module along the time dimension, the output of each
     * iteration have been recorded by the cloned modules. Thus, we can
     * reuse these outputs during the backward operations by copying the
     * outputs to _input variable.
     *
     * The output of Cell(i-1) should be one of the elements fed to the inputs
     * of Cell(i)
     * The first module in the cells array accepts zero hidden parameter.
     */

    var i = times
    while (i >= 1) {
      currentGradOutput(inputDim) = gradOutput.select(timeDim, i)

      if (i > 1) {
        cells(i - 1).regluarized(false)
        _input = cells(i - 2).output
      } else {
        cells(i - 1).regluarized(true)
        _input(hidDim) = hidden
        _input(inputDim) = input
      }

      cells(i - 1).accGradParameters(_input, currentGradOutput)
      currentGradOutput(hidDim) = cells(i - 1).gradInput.toTable(hidDim)
      i -= 1
    }
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput = gradInputCell
    gradInputCell.resizeAs(output)
    currentGradOutput(hidDim) = gradHidden
    var i = times
    while (i >= 1) {
      currentGradOutput(inputDim) = gradOutput.select(timeDim, i)

      if (i > 1) {
        _input = cells(i - 2).output
      } else {
        _input(hidDim) = hidden
        _input(inputDim) = input
      }

      cells(i - 1).updateGradInput(_input, currentGradOutput)
      currentGradOutput(hidDim) = cells(i - 1).gradInput.toTable(hidDim)
      i -= 1
    }
    copy(cells.map(x => x.gradInput.toTable[Tensor[T]](inputDim)),
        gradInputCell, 0)
    gradInput
  }

  override def backward(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val st = System.nanoTime
    currentGradOutput(hidDim) = gradHidden
    var i = times
    while (i >= 1) {
      currentGradOutput(inputDim) = gradOutput.select(timeDim, i)
      if (i > 1) {
        cells(i - 1).regluarized(false)
        _input = cells(i - 2).output
      } else {
        cells(i - 1).regluarized(true)
        _input(hidDim) = hidden
        _input(inputDim) = input
      }
      cells(i - 1).backward(_input, currentGradOutput)
      currentGradOutput(hidDim) = cells(i - 1).gradInput.toTable(hidDim)
      i -= 1
    }

    gradInput = gradInputCell
    gradInputCell.resizeAs(output)
    copy(cells.map(x => x.gradInput.toTable[Tensor[T]](inputDim)),
      gradInputCell, 0)

    this.backwardTime = System.nanoTime - st
    gradInput
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[RecurrentDecoder[T]]

  override def equals(other: Any): Boolean = other match {
    case that: RecurrentDecoder[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        cells == that.cells
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), cells)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object RecurrentDecoder extends ContainerSerializable {
  def apply[@specialized(Float, Double) T: ClassTag](outputLength: Int)
    (implicit ev: TensorNumeric[T]) : RecurrentDecoder[T] = {
    new RecurrentDecoder[T](outputLength)
  }

  override def loadModule[T: ClassTag](model : BigDLModule)
    (implicit ev: TensorNumeric[T]) : ModuleData[T] = {
    val moduleData = super.loadModule(model)
    val recurrentDecoder = moduleData.module.asInstanceOf[RecurrentDecoder[T]]
    val attrMap = model.getAttrMap

    val topologyAttr = attrMap.get("topology")
    recurrentDecoder.topology = DataConverter.getAttributeValue(topologyAttr).
      asInstanceOf[Cell[T]]

    moduleData
  }

  override def serializeModule[T: ClassTag](module : ModuleData[T])
    (implicit ev: TensorNumeric[T]) : BigDLModule = {
    val containerBuilder = BigDLModule.newBuilder(super.serializeModule(module))

    val recurrentDecoder = module.module.asInstanceOf[RecurrentDecoder[T]]

    val topologyBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(topologyBuilder, recurrentDecoder.topology,
      ModuleSerializer.abstractModuleType)
    containerBuilder.putAttr("topology", topologyBuilder.build)

    containerBuilder.build
  }
}
