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
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer._
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
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

  var subModules: Array[AbstractModule[_ <: Activity, _ <: Activity, T]] = null

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
   * The preTopology defines operations to pre-process the input when it is not dependent
   * on the time dimension. For example, the i2h in SimpleRNN Cell can be calculated before
   * the recurrence since all the input slices are independent.
   *
   * This is particular useful to boost the performance of the recurrent layer.
   *
   * Please define your own preTopology according to your Cell structure.
   * Please refer to SimpleRNN or LSTM for reference.
   * @return
   */
  var preTopology: TensorModule[T]

  private[nn] var includePreTopology: Boolean = false

  private var gradOutput2PreTopology = Tensor[T]()

  def hiddenSizeOfPreTopo: Int = hiddensShape(0)

  /**
   * resize the hidden parameters wrt the batch size, hiddens shapes.
   *
   * e.g. RnnCell contains 1 hidden parameter (H), thus it will return Tensor(size)
   *      LSTM contains 2 hidden parameters (C and H) and will return T(Tensor(), Tensor())\
   *      and recursively intialize all the tensors in the Table.
   *
   * @param hidden
   * @param batchSize batchSize
   * @param stepShape For rnn/lstm/gru, it's embedding size. For convlstm/
   *                     convlstm3D, it's a list of outputPlane, length, width, height
   * @return
   */
  def hidResize(hidden: Activity, batchSize: Int, stepShape: Array[Int]): Activity = {
    if (hidden == null) {
      if (hiddensShape.length == 1) {
        hidResize(Tensor[T](), batchSize, stepShape)
      } else {
        val _hidden = T()
        var i = 1
        while (i <= hiddensShape.length) {
          _hidden(i) = Tensor[T]()
          i += 1
        }
        hidResize(_hidden, batchSize, stepShape)
      }
    } else {
      if (hidden.isInstanceOf[Tensor[T]]) {
        require(hidden.isInstanceOf[Tensor[T]],
          "Cell: hidden should be a Tensor")
        hidden.toTensor.resize(batchSize, hiddensShape(0))
      } else {
        require(hidden.isInstanceOf[Table],
          "Cell: hidden should be a Table")
        var i = 1
        val sizes = new Array[Int](stepShape.length + 1)
        sizes(0) = batchSize
        Array.copy(stepShape, 0, sizes, 1, stepShape.size)
        while (i <= hidden.toTable.length()) {
          sizes(1) = hiddensShape(i - 1)
          hidden.toTable[Tensor[T]](i).resize(sizes)
          i += 1
        }
        hidden
      }
    }
  }

  override def updateOutput(input: Table): Table = {
    if (includePreTopology) {
      assert(preTopology != null, "preTopology cannot be null if includePreTopology is true")
      val inputTensor = input.toTable[Tensor[T]](Recurrent.inputDim)
      input(Recurrent.inputDim) = preTopology.forward(inputTensor)
      output = cell.forward(input).toTable
      input(Recurrent.inputDim) = inputTensor
    } else output = cell.forward(input).toTable
    output
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    if (includePreTopology) {
      val inputTensor = input.toTable[Tensor[T]](Recurrent.inputDim)
      input(Recurrent.inputDim) = preTopology.output
      gradInput = cell.updateGradInput(input, gradOutput).toTable
      gradOutput2PreTopology = gradInput.toTable[Tensor[T]](Recurrent.inputDim)
      gradInput(Recurrent.inputDim) =
        preTopology.updateGradInput(inputTensor, gradInput.toTable[Tensor[T]](Recurrent.inputDim))
      input(Recurrent.inputDim) = inputTensor
    } else {
      gradInput = cell.updateGradInput(input, gradOutput).toTable
    }
    gradInput
  }

  override def accGradParameters(input: Table, gradOutput: Table): Unit = {
    if (includePreTopology) {
      val inputTensor = input.toTable[Tensor[T]](Recurrent.inputDim)
      input(Recurrent.inputDim) = preTopology.output
      cell.accGradParameters(input, gradOutput)
      preTopology.accGradParameters(inputTensor, gradOutput2PreTopology)
      input(Recurrent.inputDim) = inputTensor
    } else {
      cell.accGradParameters(input, gradOutput)
    }
  }

  override def backward(input: Table, gradOutput: Table): Table = {
    val before = System.nanoTime()
    if (includePreTopology) {
      val inputTensor = input.toTable[Tensor[T]](Recurrent.inputDim)
      input(Recurrent.inputDim) = preTopology.output
      gradInput = cell.backward(input, gradOutput)
      gradInput(Recurrent.inputDim) =
        preTopology.backward(inputTensor, gradInput.toTable[Tensor[T]](Recurrent.inputDim))
      input(Recurrent.inputDim) = inputTensor
    } else {
      gradInput = cell.backward(input, gradOutput).toTable
    }
    backwardTime += System.nanoTime() - before

    gradInput
  }

  override def getTimes(): Array[(AbstractModule[_ <: Activity, _ <: Activity, T], Long, Long)] = {
    val cellTimes = cell.getTimes
    val (cellFwdTime, cellBwdTime) = Utils.calculateFwdBwdTime(cellTimes)
    cellTimes ++ Array((this, forwardTime - cellFwdTime, backwardTime - cellBwdTime))
  }

  override def resetTimes(): Unit = {
    super.resetTimes()
    cell.resetTimes
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    val _cell = if (includePreTopology) {
      Sequential().add(preTopology).add(cell)
    } else cell
    _cell.parameters()
  }

  override def getParametersTable(): Table = {
    val _cell = if (includePreTopology) {
      Sequential().add(preTopology).add(cell)
    } else cell
    _cell.getParametersTable()
  }

  override def reset(): Unit = {
    cell.reset()
    if (includePreTopology) preTopology.reset()
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

object CellSerializer extends ModuleSerializable {

  private[nn] def populateCellAttributes[T: ClassTag](context : DeserializeContext,
                                                   cell : Cell[T])
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {
    val attrMap = context.bigdlModule.getAttrMap
    cell.cell = DataConverter.getAttributeValue(context, attrMap.get("cell")).
      asInstanceOf[AbstractModule[Activity, Activity, T]]

    val preTopologyAttr = attrMap.get("preTopology")
    cell.preTopology = DataConverter.getAttributeValue(context, preTopologyAttr).
      asInstanceOf[TensorModule[T]]

    val includePreTopologyAttr = attrMap.get("includePreTopology")
    cell.includePreTopology = DataConverter.getAttributeValue(context,
      includePreTopologyAttr).asInstanceOf[Boolean]
    cell
  }

  override def doLoadModule[T: ClassTag](context : DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {
    val cell = super.doLoadModule(context).asInstanceOf[Cell[T]]
    populateCellAttributes(context, cell)
  }

  private[nn] def saveCellAttributes[T: ClassTag](context: SerializeContext[T],
    cellModuleBuilder : BigDLModule.Builder)
    (implicit ev: TensorNumeric[T]) : Unit = {
    val cellModule = context.moduleData.module.asInstanceOf[Cell[T]]

    val cellSerializerFlagBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, cellSerializerFlagBuilder, true,
      scala.reflect.runtime.universe.typeOf[Boolean])
    cellModuleBuilder.putAttr("is_cell_module", cellSerializerFlagBuilder.build)

    val cellBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, cellBuilder, cellModule.cell,
      ModuleSerializer.abstractModuleType)
    cellModuleBuilder.putAttr("cell", cellBuilder.build)

    val preTopologyBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, preTopologyBuilder,
      cellModule.preTopology, ModuleSerializer.tensorModuleType)
    cellModuleBuilder.putAttr("preTopology", preTopologyBuilder.build)

    val includePreTopologyBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, includePreTopologyBuilder,
      cellModule.includePreTopology, scala.reflect.runtime.universe.typeOf[Boolean])
    cellModuleBuilder.putAttr("includePreTopology", includePreTopologyBuilder.build)
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
                                              cellModuleBuilder : BigDLModule.Builder)
                                           (implicit ev: TensorNumeric[T]) : Unit = {

    super.doSerializeModule(context, cellModuleBuilder)
    saveCellAttributes(context, cellModuleBuilder)
  }
}
