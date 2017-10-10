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
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.bigdl.utils.serializer.{ContainerSerializable, DataConverter, ModuleData, ModuleSerializer}
import serialization.Bigdl.{AttrValue, BigDLModule}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * [[RecurrentDecoder]] module is a container of rnn cells that used to make
 * a prediction of the next timestep based on the prediction we made from
 * the previous timestep. Input for RecurrentDecoder is dynamically composed
 * during training. input at t(i) is output at t(i-1), input at t(0) is
 * user input, and user input has to be batch x stepShape(shape of the input
 * at a single time step).

 * Different types of rnn cells can be added using add() function.
 * @param seqLength sequence length of the output
 */
class RecurrentDecoder[T : ClassTag](seqLength: Int)
  (implicit ev: TensorNumeric[T]) extends Recurrent[T] {

  times = seqLength

  /**
   *
   *  modules: -- preTopology
   *           |- topology (cell)
   *
   * The topology (or cell) will be cloned for N times w.r.t the time dimension.
   * The preTopology will be execute only once before the recurrence.
   *
   * @param module module to be add
   * @return this container
   */
  override def add(module: AbstractModule[_ <: Activity, _ <: Activity, T]):
    RecurrentDecoder.this.type = {
    require(module.isInstanceOf[Cell[T]],
      "Recurrent: contained module should be Cell type")

    topology = module.asInstanceOf[Cell[T]]
    preTopology = topology.preTopology

    if (preTopology != null) {
      modules += preTopology
    }
    modules += topology

    require((preTopology == null && modules.length == 1) ||
      (topology != null && preTopology != null && modules.length == 2),
      "Recurrent extend: should contain only one cell or plus a pre-topology" +
        " to process input")
    this
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim == 2 || input.dim == 4 || input.dim == 5,
      "Recurrent: input should be a 2D/4D/5D Tensor, e.g [batch, nDim], " +
        s"current input.dim = ${input.dim}")
    val hiddenSize = topology.hiddensShape(0)
    val outputSize = input.size()
    require(hiddenSize == input.size()(1), "hiddenSize is " +
      "not the same with input size!! Please update cell settings or use Recurrent instead!")
    val featureSizes = outputSize.drop(1)
    output.resize(Array(batchSize, times) ++ featureSizes)
    // Clone N modules along the sequence dimension.
    initHidden(featureSizes)
    if (preTopology == null) {
      input2Cell.resize(output.size())
    }

    /**
     * currentInput forms a T() type. It contains two elements, hidden and input.
     * Each time it will feed the cell with T(hidden, input) (or T(input, hidden) depends on
     * your hidDim and inputDim), and the cell will give a table output containing two
     * identical elements T(output, output). One of the elements from the cell output is
     * the updated hidden. Thus the currentInput will update its hidden element with this output.
     */
    // Clone N modules along the sequence dimension.
    cloneCells()
    currentInput(hidDim) = if (initHiddenState != null) initHiddenState
    else hidden

    var i = 1
    while (i <= times) {
      // input at t(0) is user input
      val inputTmp = if (i == 1) {
        input
      } else {
        // input at t(i) is output at t(i-1)
        cells(i - 2).output.toTable[Tensor[T]](inputDim)
      }

      currentInput(inputDim) = if (preTopology != null) {
        preTopology.updateOutput(inputTmp).toTensor[T]
      } else {
        inputTmp
      }
      input2Cell.select(2, i).copy(currentInput(inputDim))
      cells(i - 1).updateOutput(currentInput)
      currentInput(hidDim) = cells(i - 1).output.toTable(hidDim)
      i += 1
    }

    Recurrent.copy(cells.map(x => x.output.toTable[Tensor[T]](inputDim)), output)
    output
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    throw new Exception("Should not enter RecurrentDecoder accGradParameters" +
      "as it has override backward")
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    throw new Exception("Should not enter RecurrentDecoder updateGradInput" +
      "as it has override backward")
    gradInput
  }

  override def backward(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val st = System.nanoTime
    gradInput.resizeAs(output)
    currentGradOutput(hidDim) = gradHidden
    var i = times
    while (i >= 1) {
      currentGradOutput(inputDim) = if (i == times) {
      Recurrent.selectCopy(gradOutput, i, stepGradBuffer)
      }
      else {
        val _gradInput = if (preTopology != null) {
          val input0 = if (i == 1) input else cells(i - 2).output.toTable[Tensor[T]](inputDim)
          preTopology.backward(input0,
            cells(i).gradInput.toTable[Tensor[T]](inputDim)).toTensor[T]
        } else {
          cells(i).gradInput.toTable[Tensor[T]](inputDim)
        }
        gradInput.select(timeDim, i + 1).copy(_gradInput)
        gradOutput.select(timeDim, i).clone().add(_gradInput)
//        gradOutput.select(timeDim, i)
//        gradOutput.select(timeDim, i).clone().add(cells(i).gradInput.toTable[Tensor[T]](inputDim))
      }

      _input(hidDim) = if (i > 1) cells(i - 2).output.toTable(hidDim)
      else if (initHiddenState == null) hidden else initHiddenState

      _input(inputDim) = Recurrent.selectCopy(input2Cell, i, stepInput2CellBuf)

      if (i == 1) {
        cells(i - 1).regluarized(true)
      } else {
        cells(i - 1).regluarized(false)
      }
      cells(i - 1).backward(_input, currentGradOutput)
      currentGradOutput(hidDim) = cells(i - 1).gradInput.toTable(hidDim)
      i -= 1
    }

    val gradInput0 = if (preTopology == null) cells(0).gradInput.toTable[Tensor[T]](inputDim)
      else preTopology.backward(input, cells(0).gradInput.toTable[Tensor[T]](inputDim)).toTensor
    gradInput.select(timeDim, 1).copy(gradInput0)
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
