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
import com.intel.analytics.bigdl.utils.serializer._
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
class RecurrentDecoder[T : ClassTag](seqLength: Int)
  (implicit ev: TensorNumeric[T]) extends Recurrent[T] {

  times = seqLength
  private val newInput = Tensor[T]()

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim == 2 || input.dim == 4 || input.dim == 5,
      "Recurrent: input should be a 2D/4D/5D Tensor, e.g [batch, nDim], " +
        s"current input.dim = ${input.dim}")

    batchSize = input.size(batchDim)

    val hiddenSize = topology.hiddensShape(0)
    val outputSize = input.size()
    outputSize(1) = hiddenSize
    require(hiddenSize == input.size()(1), "hiddenSize is " +
      "not the same with input size!! Please update cell settings or use Recurrent instead!")
    val featureSizes = outputSize.drop(1)
    output.resize(Array(batchSize, times) ++ featureSizes)
    // Clone N modules along the sequence dimension.
    extend(featureSizes)
    if (preTopology != null) newInput.resize(output.size())
    else outputCell.resize(output.size())

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
      // input at t(0) is user input
      val inputTmp = if (i == 1) {
        input
      } else {
        // input at t(i) is output at t(i-1)
        cells(i - 2).output.toTable[Tensor[T]](inputDim)
      }

      currentInput(inputDim) = if (preTopology != null) {
        newInput.narrow(2, i, 1).copy(inputTmp)
        val sizes = 1 +: inputTmp.size()
        inputTmp.resize(sizes)
        val _input = preTopology.updateOutput(inputTmp).toTensor[T]
        inputTmp.resize(sizes.takeRight(sizes.length - 1))
        _input.select(1, 1)
      } else {
        outputCell.narrow(2, i, 1).copy(inputTmp)
        inputTmp
      }
      cells(i - 1).updateOutput(currentInput)
      currentInput(hidDim) = cells(i - 1).output.toTable(hidDim)
      i += 1
    }

    if (preTopology != null) {
      // For backward preTopology use
      outputCell = preTopology.updateOutput(newInput).toTensor[T]
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
    currentGradOutput(hidDim) = gradHidden
    var i = times
    while (i >= 1) {
      currentGradOutput(inputDim) = if (i == times) gradOutput.select(timeDim, i)
      else {
        gradOutput.select(timeDim, i).clone()
          .add(cells(i).gradInput.toTable[Tensor[T]](inputDim).clone())
      }

      _input(hidDim) = if (i > 1) cells(i - 2).output.toTable(hidDim)
      else if (initState == null) hidden else initState
      _input(inputDim) = Recurrent.selectCopy(outputCell, i, outputBuffer)

      if (i == 1) {
        cells(i - 1).regluarized(true)
      } else {
        cells(i - 1).regluarized(false)
      }
      cells(i - 1).backward(_input, currentGradOutput)
      currentGradOutput(hidDim) = cells(i - 1).gradInput.toTable(hidDim)
      i -= 1
    }

    gradInput = if (preTopology != null) {
      /**
       * if preTopology is Sequential, it has not created gradInput.
       * Thus, it needs to create a new Tensor.
       */
      if (preTopology.gradInput == null) {
        preTopology.gradInput = Tensor[T]()
      }
      preTopology.gradInput.toTensor[T]
    } else {
      gradInputCell
    }
    gradInputCell.resizeAs(outputCell)
    Recurrent.copy(cells.map(x => x.gradInput.toTable[Tensor[T]](inputDim)), gradInputCell)
    if (preTopology != null) {
      gradInput = preTopology.backward(newInput, gradInputCell).toTensor[T]
    }

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

  override def loadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumeric[T]) : ModuleData[T] = {
    val moduleData = super.loadModule(context)
    val recurrentDecoder = moduleData.module.asInstanceOf[RecurrentDecoder[T]]
    val attrMap = context.bigdlModule.getAttrMap

    val topologyAttr = attrMap.get("topology")
    recurrentDecoder.topology = DataConverter.
      getAttributeValue(context, topologyAttr).
      asInstanceOf[Cell[T]]

    moduleData
  }

  override def serializeModule[T: ClassTag](context: SerializeContext[T])
    (implicit ev: TensorNumeric[T]) : SerializeResult = {
    val containerBuilder = (super.serializeModule(context).bigDLModule)

    val recurrentDecoder = context.moduleData.module.asInstanceOf[RecurrentDecoder[T]]

    val topologyBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context,
      topologyBuilder, recurrentDecoder.topology,
      ModuleSerializer.abstractModuleType)
    containerBuilder.putAttr("topology", topologyBuilder.build)

    SerializeResult(containerBuilder, context.storages)
  }
}
