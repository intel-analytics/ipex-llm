/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.api.keras.layers.internal

import com.intel.analytics.bigdl.nn.{BatchNormParams, BigDLWrapperUtils, Cell, Recurrent}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.utils.serializer.{ContainerSerializable, DeserializeContext, ModuleSerializer, SerializeContext}
import com.intel.analytics.zoo.pipeline.api.keras.layers.{InternalConvLSTM2D, InternalConvLSTM3D}

import scala.reflect.ClassTag
import scala.reflect.runtime._

class InternalRecurrent[T: ClassTag](
    batchNormParams: BatchNormParams[T] = null
)(implicit ev: TensorNumeric[T]) extends Recurrent[T](batchNormParams, false) {

  override def add(module: AbstractModule[_ <: Activity, _ <: Activity, T]): this.type = {
    super.add(module)
    if (this.preTopology != null) {
      module.asInstanceOf[Cell[T]].preTopology = null
    }
    this
  }

  def getHiddenShape(): Array[Int] = {
    this.topology.hiddensShape
  }

  // get gradient hidden state at the first time step
  def getGradHiddenState(): Activity = {
    require(cells != null && cells(0).gradInput != null,
      "getGradHiddenState need to be called after backward")
    cells(0).gradInput.toTable(hidDim)
  }

  protected var initGradHiddenState: Activity = null
  // set gradient hiddent state at the last time step
  def setGradHiddenState(gradHiddenState: Activity): Unit = {
    initGradHiddenState = gradHiddenState
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    if (initGradHiddenState != null) gradHidden = initGradHiddenState
    super.accGradParameters(input, gradOutput)
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (initGradHiddenState != null) gradHidden = initGradHiddenState
    super.updateGradInput(input, gradOutput)
  }

  override def backward(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val st = System.nanoTime
    gradInput = updateGradInput(input, gradOutput)
    accGradParameters(input, gradOutput)
    this.backwardTime = System.nanoTime - st
    gradInput
  }

  // fix not support "valid" padding issue for convlstm, it would be better fix in BigDL
  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim == 3 || input.dim == 5 || input.dim == 6,
      "Recurrent: input should be a 3D/5D/6D Tensor, e.g [batch, times, nDim], " +
        s"current input.dim = ${input.dim}")

    batchSize = input.size(batchDim)
    times = input.size(timeDim)

    input2Cell = if (preTopology != null) {
      preTopology.forward(input).toTensor[T]
    } else {
      input
    }

    val hiddenSize = topology.hiddensShape(0)
    val outputSize = if (topology.isInstanceOf[InternalConvLSTM2D[T]]) {
      topology.asInstanceOf[InternalConvLSTM2D[T]].getOutputSize(input.size())
    } else if (topology.isInstanceOf[InternalConvLSTM3D[T]]) {
      topology.asInstanceOf[InternalConvLSTM3D[T]].getOutputSize(input.size())
    } else {
      val sizes = input.size()
      sizes(2) = hiddenSize
      sizes
    }
    output.resize(outputSize)

    /**
     * currentInput forms a T() type. It contains two elements, hidden and input.
     * Each time it will feed the cell with T(hidden, input) (or T(input, hidden) depends on
     * your hidDim and inputDim), and the cell will give a table output containing two
     * identical elements T(output, output). One of the elements from the cell output is
     * the updated hidden. Thus the currentInput will update its hidden element with this output.
     */
    var i = 1
    // Clone N modules along the sequence dimension.
    initHidden(outputSize.drop(2))
    cloneCells()

    currentInput(hidDim) = if (initHiddenState != null) initHiddenState
    else hidden

    while (i <= times) {
      currentInput(inputDim) = BigDLWrapperUtils.selectCopy(input2Cell, i, stepInput2CellBuf)
      cells(i - 1).forward(currentInput)
      val curOutput = cells(i - 1).output
      currentInput(hidDim) = curOutput[Table](hidDim)
      i += 1
    }

    BigDLWrapperUtils.copy(cells.map(x => x.output.toTable[Tensor[T]](inputDim)),
      output)
    output
  }
}

object InternalRecurrent extends ContainerSerializable {

  ModuleSerializer.registerModule(
    "com.intel.analytics.zoo.pipeline.api.keras.layers.internal.InternalRecurrent",
    InternalRecurrent)

  override def doLoadModule[T: ClassTag](context : DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {

    val attrMap = context.bigdlModule.getAttrMap

    val flag = DataConverter
      .getAttributeValue(context, attrMap.get("bnorm"))
      .asInstanceOf[Boolean]
    val recurrent = if (flag) {
      new InternalRecurrent[T](BatchNormParams[T]())
    } else {
      new InternalRecurrent[T]()
    }

    val topologyAttr = attrMap.get("topology")
    recurrent.topology = DataConverter.getAttributeValue(context, topologyAttr).
      asInstanceOf[Cell[T]]

    val preTopologyAttr = attrMap.get("preTopology")
    recurrent.preTopology = DataConverter.getAttributeValue(context, preTopologyAttr).
      asInstanceOf[AbstractModule[Activity, Activity, T]]

    if (recurrent.preTopology != null) {
      recurrent.modules.append(recurrent.preTopology)
    }
    recurrent.modules.append(recurrent.topology)

    if (flag) {
      val bnormEpsAttr = attrMap.get("bnormEps")
      recurrent.batchNormParams.eps =
        DataConverter.getAttributeValue(context, bnormEpsAttr)
          .asInstanceOf[Double]

      val bnormMomentumAttr = attrMap.get("bnormMomentum")
      recurrent.batchNormParams.momentum =
        DataConverter.getAttributeValue(context, bnormMomentumAttr)
          .asInstanceOf[Double]

      val bnormInitWeightAttr = attrMap.get("bnormInitWeight")
      recurrent.batchNormParams.initWeight =
        DataConverter.getAttributeValue(context, bnormInitWeightAttr)
          .asInstanceOf[Tensor[T]]

      val bnormInitBiasAttr = attrMap.get("bnormInitBias")
      recurrent.batchNormParams.initBias =
        DataConverter.getAttributeValue(context, bnormInitBiasAttr)
          .asInstanceOf[Tensor[T]]

      val bnormInitGradWeightAttr = attrMap.get("bnormInitGradWeight")
      recurrent.batchNormParams.initGradWeight =
        DataConverter.getAttributeValue(context, bnormInitGradWeightAttr)
          .asInstanceOf[Tensor[T]]

      val bnormInitGradBiasAttr = attrMap.get("bnormInitGradBias")
      recurrent.batchNormParams.initGradBias =
        DataConverter.getAttributeValue(context, bnormInitGradBiasAttr)
          .asInstanceOf[Tensor[T]]

      val bnormAffineAttr = attrMap.get("bnormAffine")
      recurrent.batchNormParams.affine =
        DataConverter.getAttributeValue(context, bnormAffineAttr)
          .asInstanceOf[Boolean]
    }

    recurrent
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
      recurrentBuilder : BigDLModule.Builder)
    (implicit ev: TensorNumeric[T]) : Unit = {

    val recurrent = context.moduleData.module.asInstanceOf[InternalRecurrent[T]]

    val topologyBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, topologyBuilder, recurrent.topology,
      ModuleSerializer.abstractModuleType)
    recurrentBuilder.putAttr("topology", topologyBuilder.build)

    val preTopologyBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, preTopologyBuilder,
      recurrent.preTopology, ModuleSerializer.abstractModuleType)
    recurrentBuilder.putAttr("preTopology", preTopologyBuilder.build)

    val flag = if (recurrent.batchNormParams != null) {

      val bnormEpsBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context, bnormEpsBuilder,
        recurrent.batchNormParams.eps, universe.typeOf[Double])
      recurrentBuilder.putAttr("bnormEps", bnormEpsBuilder.build)

      val bnormMomentumBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context, bnormMomentumBuilder,
        recurrent.batchNormParams.momentum, universe.typeOf[Double])
      recurrentBuilder.putAttr("bnormMomentum", bnormMomentumBuilder.build)

      val bnormInitWeightBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context, bnormInitWeightBuilder,
        recurrent.batchNormParams.initWeight, ModuleSerializer.tensorType)
      recurrentBuilder.putAttr("bnormInitWeight", bnormInitWeightBuilder.build)

      val bnormInitBiasBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context, bnormInitBiasBuilder,
        recurrent.batchNormParams.initBias, ModuleSerializer.tensorType)
      recurrentBuilder.putAttr("bnormInitBias", bnormInitBiasBuilder.build)

      val bnormInitGradWeightBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context, bnormInitGradWeightBuilder,
        recurrent.batchNormParams.initGradWeight, ModuleSerializer.tensorType)
      recurrentBuilder.putAttr("bnormInitGradWeight", bnormInitGradWeightBuilder.build)

      val bnormInitGradBiasBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context, bnormInitGradBiasBuilder,
        recurrent.batchNormParams.initGradBias, ModuleSerializer.tensorType)
      recurrentBuilder.putAttr("bnormInitGradBias", bnormInitGradBiasBuilder.build)

      val bnormAffineBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context, bnormAffineBuilder,
        recurrent.batchNormParams.affine, universe.typeOf[Boolean])
      recurrentBuilder.putAttr("bnormAffine", bnormAffineBuilder.build)

      true
    } else {
      false
    }

    val bNormBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, bNormBuilder,
      flag, universe.typeOf[Boolean])
    recurrentBuilder.putAttr("bnorm", bNormBuilder.build)

  }
}
