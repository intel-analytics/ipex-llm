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

package com.intel.analytics.bigdl.utils.intermediate

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.mkl.{AlgKind, Direction}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat, TensorModule}
import com.intel.analytics.bigdl.nn.{Module => _, _}
import com.intel.analytics.bigdl.nn.mkldnn._
import com.intel.analytics.bigdl.optim.DistriOptimizer._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{DirectedGraph, Node, ReflectionUtils, T}

import scala.collection.mutable
import scala.reflect.ClassTag

private[bigdl] class IRToDnn extends ConvertBase[IRElement[Float], Module[Float]] {

  private val prefix = "com.intel.analytics.bigdl.nn.mkldnn."
  // converter function mappings
  private val IR2DnnMap = new mutable.HashMap[String, (IRElement[Float]) => Module[Float]]

  mapInit()

  private def mapInit(): Unit = {
    IR2DnnMap("IRSpatialConvolution") = fromSpatialConvolution
    IR2DnnMap("IRSpatialShareConvolution") = fromSpatialShareConvolution
    IR2DnnMap("IRSpatialMaxPooling") = fromSpatialMaxPooling
    IR2DnnMap("IRSpatialAveragePooling") = fromSpatialAveragePooling
    IR2DnnMap("IRSpatialBatchNormalization") = fromSpatialBatchNormalization
    IR2DnnMap("IRSpatialCrossMapLRN") = fromSpatialCrossMapLRN
    IR2DnnMap("IRReLU") = fromReLU
    IR2DnnMap("IRJoinTable") = fromJoinTable
    IR2DnnMap("IRGeneralModule") = fromBlasModule
    IR2DnnMap("IRInput") = fromInput
  }

  override def convertLayerCheck(layer: IRElement[Float]): Boolean = {
    val name = layer.getOp().name
    if (IR2DnnMap.contains(name) && checkRequirement(layer)) return true
    return ReflectionUtils.findClass(prefix + name.substring(2)) != null
  }

  override def convertLayer(layer: IRElement[Float]) : Module[Float] = {
    val name = layer.getOp().name
    if (IR2DnnMap.contains(name)) {
      val dnn = IR2DnnMap(name)(layer)
      if (layer.getName != "") dnn.setName(layer.name)
      dnn
    } else {
      ReflectionUtils.reflectFromIR(layer, Class.forName(prefix + name.substring(2)))
    }
  }

  override def convertingCheck(allNodes: Array[Node[IRElement[Float]]]) : Boolean = {
    var convert = true
    allNodes.foreach(node => {
      val op = node.element.getOp()
      if (!convertLayerCheck(node.element)) {
        logger.info(s"${node.element.getOp()} convertion failed")
        convert = false
      }
    })
    convert
  }

  override def convert(allNodes: Array[Node[IRElement[Float]]])
    : mutable.HashMap[Node[IRElement[Float]], Node[Module[Float]]] = {
    val nodeMap = new mutable.HashMap[Node[IRElement[Float]], Node[Module[Float]]]()
    allNodes.foreach(node => {
      val op = node.element.getOp()
      var dnn = if (convertLayerCheck(node.element)) {
        new Node(convertLayer(node.element))
      } else {
        throw new UnsupportedOperationException(s"can not find ${node.element.getOp()} ")
      }
      // special treat for reshape -> linear and view -> linear
      if (op.isInstanceOf[IRGeneralModule[Float]]) {
        val m = op.asInstanceOf[IRGeneralModule[Float]].model
        if (m.isInstanceOf[Reshape[Float]] && node.nextNodes.length == 1 &&
          node.nextNodes(0).element.getOp().isInstanceOf[IRLinear[Float]]) {
          dnn = new Node(mkldnn.Identity[Float]().asInstanceOf[Module[Float]])
        } else if (m.isInstanceOf[View[Float]] && node.nextNodes.length == 1 &&
          node.nextNodes(0).element.getOp().isInstanceOf[IRLinear[Float]]) {
          dnn = new Node(mkldnn.Identity[Float]().asInstanceOf[Module[Float]])
        }
      }
      nodeMap.put(node, dnn)
    })
    cloneNode(allNodes, nodeMap)
    nodeMap
  }

  private def fromReLU(node: IRElement[Float]) : Module[Float] = {
    val layer = mkldnn.ReLU()
    ReflectionUtils.setScales(node, layer)
    layer
  }

  private def fromSpatialConvolution(node: IRElement[Float]) : Module[Float] = {
    val t = node.getOp().asInstanceOf[IRSpatialConvolution[Float]]
    require(t.format == DataFormat.NCHW, "Dnn SpatialConvolution only supports NCHW")
    ReflectionUtils.reflectFromIR(node, Class.forName(prefix + "SpatialConvolution"))
  }

  private def fromSpatialShareConvolution(node: IRElement[Float]) : Module[Float] = {
    val t = node.getOp().asInstanceOf[IRSpatialShareConvolution[Float]]
    require(t.format == DataFormat.NCHW, "Dnn SpatialConvolution only supports NCHW")
    ReflectionUtils.reflectFromIR(node, Class.forName(prefix + "SpatialConvolution"))
  }

  private def fromSpatialMaxPooling(node: IRElement[Float]) : Module[Float] = {
    val t = node.getOp().asInstanceOf[IRSpatialMaxPooling[Float]]
    require(t.format == DataFormat.NCHW, "Dnn SpatialMaxPooling only supports NCHW")
    val layer = ReflectionUtils.reflectFromIR(
      node, Class.forName(prefix + "MaxPooling")).asInstanceOf[MaxPooling]
    if (t.ceilMode) layer.ceil() else layer.floor()
    layer
  }

  private def fromSpatialAveragePooling(node: IRElement[Float]) : Module[Float] = {
    val t = node.getOp().asInstanceOf[IRSpatialAveragePooling[Float]]
    require(t.format == DataFormat.NCHW, "Dnn SpatialAveragePooling only supports NCHW")
    val layer = ReflectionUtils.reflectFromIR(
      node, Class.forName(prefix + "AvgPooling")).asInstanceOf[AvgPooling]
    if (t.ceilMode) layer.ceil() else layer.floor()
    layer
  }

  private def fromSpatialCrossMapLRN(node: IRElement[Float]) : Module[Float] = {
    val t = node.getOp().asInstanceOf[IRSpatialCrossMapLRN[Float]]
    require(t.format == DataFormat.NCHW, "Dnn LRN only supports NCHW")
    ReflectionUtils.reflectFromIR(node, Class.forName(prefix + "LRN"))
  }

  private def fromJoinTable(node: IRElement[Float]) : Module[Float] = {
    val t = node.getOp().asInstanceOf[IRJoinTable[Float]]
    require(t.nInputDims <= 0,
      s"Dnn JoinTable only supports nInputDims <= 0, but get ${t.nInputDims}")
    mkldnn.JoinTable(t.dimension)
  }

  private def fromSpatialBatchNormalization(node: IRElement[Float]) : Module[Float] = {
    val t = node.getOp().asInstanceOf[IRSpatialBatchNormalization[Float]]
    require(t.dataFormat == DataFormat.NCHW, "Dnn SpatialBatchNormalization only supports NCHW")
    val nOutput = t.nOutput
    val eps = t.eps
    val momentum = t.momentum
    val initWeight = t.initWeight
    val initBias = t.initBias
    val initGradWeight = t.initGradWeight
    val initGradBias = t.initGradBias

    val layer = mkldnn.SpatialBatchNormalization(nOutput, eps, momentum,
      true, initWeight, initBias, initGradWeight, initGradBias)

    val params = node.getParameters()
    if (params._1 != null) layer.weightAndBias.copy(params._1)
    if (params._2 != null) layer.gradWeightAndBias.copy(params._2)

    val extraParams = layer.getExtraParameter()
    if (t.runningMean != null) extraParams(0).copy(t.runningMean.toTensor[Float])
    if (t.runningVar != null) extraParams(1).copy(t.runningVar.toTensor[Float])

    ReflectionUtils.setScales(node, layer)

    // reminder: assume batch_norm is converted from blas
    layer.needScale = true
    layer
  }

  private def fromBlasModule(node: IRElement[Float]) : Module[Float] = {
    val model = node.getOp().asInstanceOf[IRGeneralModule[Float]].model
    if (model.isInstanceOf[BiRecurrent[Float]]) {
      fromBiRecurrent(node)
    } else if (model.isInstanceOf[Recurrent[Float]]) {
      fromRecurrent(node)
    } else BlasWrapper(node.getOp().asInstanceOf[IRGeneralModule[Float]].model)
  }

  private def fromRecurrent(node: IRElement[Float]): Module[Float] = {
    val model = node.getOp().asInstanceOf[IRGeneralModule[Float]]
      .model.asInstanceOf[Recurrent[Float]]
    val layer = model.getCell()
    if (layer.isInstanceOf[LSTM[Float]] && model.batchNormParams ==  null) {
      val lstm = layer.asInstanceOf[LSTM[Float]]
      if (lstm.activation.isInstanceOf[Tanh[Float]] &&
        lstm.innerActivation.isInstanceOf[Sigmoid[Float]] &&
        lstm.p == 0.0f &&
        lstm.wRegularizer == null &&
        lstm.bRegularizer == null &&
        lstm.uRegularizer == null) {
        val f = AlgKind.EltwiseTanh
        val direction = Direction.UnidirectionalLeft2Right
        val inputSize = lstm.inputSize
        val hiddenSize = lstm.hiddenSize
        val lstmDnn = nn.mkldnn.RNN(AlgKind.VanillaLstm, inputSize, hiddenSize,
          f, direction, layers = 1)

        // copy weight from blas lstm to dnn lstm
        val lstm_n_gates = 4

        val blasParams = model.parameters()._1
        val initWeight0 = blasParams(0)
        val initBias0 = blasParams(1)
        val initWeightIter0 = blasParams(2)

        var num = initWeight0.size(1) / lstm_n_gates
        var gate1 = initWeight0.narrow(1, 1, num)
        var gate3 = initWeight0.narrow(1, num + 1, num)
        var gate2 = initWeight0.narrow(1, num * 2 + 1, num)
        var gate4 = initWeight0.narrow(1, num * 3 + 1, num)

        var initWeight = Tensor[Float](lstm_n_gates, hiddenSize, inputSize)
        initWeight.select(1, 1).copy(gate1)
        initWeight.select(1, 2).copy(gate2)
        initWeight.select(1, 3).copy(gate3)
        initWeight.select(1, 4).copy(gate4)
        // original Array(inputSize, lstm_n_gates, hiddenSize)
        initWeight = initWeight.transpose(1, 3).transpose(2, 3)

        num = initBias0.size(1) / lstm_n_gates
        gate1 = initBias0.narrow(1, 1, num)
        gate3 = initBias0.narrow(1, num + 1, num)
        gate2 = initBias0.narrow(1, num * 2 + 1, num)
        gate4 = initBias0.narrow(1, num * 3 + 1, num)

        val initBias = Tensor[Float](lstm_n_gates, hiddenSize)
        initBias.select(1, 1).copy(gate1)
        initBias.select(1, 2).copy(gate2)
        initBias.select(1, 3).copy(gate3)
        initBias.select(1, 4).copy(gate4)

        num = initWeightIter0.size(1) / lstm_n_gates
        gate1 = initWeightIter0.narrow(1, 1, num)
        gate3 = initWeightIter0.narrow(1, num + 1, num)
        gate2 = initWeightIter0.narrow(1, num * 2 + 1, num)
        gate4 = initWeightIter0.narrow(1, num * 3 + 1, num)

        var initIterWeight = Tensor[Float](lstm_n_gates, hiddenSize, hiddenSize)
        initIterWeight.select(1, 1).copy(gate1)
        initIterWeight.select(1, 2).copy(gate2)
        initIterWeight.select(1, 3).copy(gate3)
        initIterWeight.select(1, 4).copy(gate4)
        // original Array(hiddenSize, lstm_n_gates, hiddenSize)
        initIterWeight = initIterWeight.transpose(1, 3).transpose(2, 3)

        val weights = lstmDnn.parameters()._1
        weights(0).copy(initWeight)
        weights(1).copy(initBias)
        weights(2).copy(initIterWeight)

        return lstmDnn
      }
    }
    BlasWrapper(node.getOp().asInstanceOf[IRGeneralModule[Float]].model)
  }

  private def fromBiRecurrent(node: IRElement[Float]): Module[Float] = {
    val model = node.getOp().asInstanceOf[IRGeneralModule[Float]]
      .model.asInstanceOf[BiRecurrent[Float]]
    val layer = model.layer.getCell()
    val revLayer = model.revLayer.getCell()
    val merge = model.getMerge()
    if ((layer equals revLayer) && layer.isInstanceOf[LSTM[Float]] &&
      model.batchNormParams ==  null && model.isSplitInput == false &&
      (merge.isInstanceOf[nn.CAddTable[Float, _]] || merge.isInstanceOf[nn.ConcatTable[Float]])) {
      val lstm = layer.asInstanceOf[LSTM[Float]]
      if (lstm.activation.isInstanceOf[Tanh[Float]] &&
        lstm.innerActivation.isInstanceOf[Sigmoid[Float]] &&
        lstm.p == 0.0f &&
        lstm.wRegularizer == null &&
        lstm.bRegularizer == null &&
        lstm.uRegularizer == null) {
        val f = AlgKind.EltwiseTanh
        val direction = if (merge.isInstanceOf[nn.CAddTable[Float, _]]) {
          Direction.BidirectionalSum
        } else Direction.BidirectionalConcat
        val inputSize = lstm.inputSize
        val hiddenSize = lstm.hiddenSize
        val lstmDnn = nn.mkldnn.RNN(AlgKind.VanillaLstm, inputSize, hiddenSize,
          f, direction, layers = 1)

        // copy weight from blas lstm to dnn lstm
        val lstm_n_gates = 4

        val blasParams = model.parameters()._1
        val initWeight0 = Tensor[Float](Array(2, hiddenSize * lstm_n_gates, inputSize))
        val initWeightIter0 = Tensor[Float](Array(2, hiddenSize * lstm_n_gates, hiddenSize))
        val initBias0 = Tensor[Float](Array(2, lstm_n_gates * hiddenSize))

        initWeight0(1).resizeAs(blasParams(0)).copy(blasParams(0))
        initBias0(1).resizeAs(blasParams(1)).copy(blasParams(1))
        initWeightIter0(1).resizeAs(blasParams(2)).copy(blasParams(2))
        initWeight0(2).resizeAs(blasParams(3)).copy(blasParams(3))
        initBias0(2).resizeAs(blasParams(4)).copy(blasParams(4))
        initWeightIter0(2).resizeAs(blasParams(5)).copy(blasParams(5))

        val initWeight = Tensor[Float](Array(2, lstm_n_gates, hiddenSize, inputSize))
        val initWeightIter = Tensor[Float](Array(2, lstm_n_gates, hiddenSize, hiddenSize))
        val initBias = Tensor[Float](Array(2, lstm_n_gates, hiddenSize))

        for (i <- 1 to 2) {
          var num = initWeight0(i).size(1) / lstm_n_gates
          var gate1 = initWeight0(i).narrow(1, 1, num)
          var gate3 = initWeight0(i).narrow(1, num + 1, num)
          var gate2 = initWeight0(i).narrow(1, num * 2 + 1, num)
          var gate4 = initWeight0(i).narrow(1, num * 3 + 1, num)
          initWeight(i).select(1, 1).copy(gate1)
          initWeight(i).select(1, 2).copy(gate2)
          initWeight(i).select(1, 3).copy(gate3)
          initWeight(i).select(1, 4).copy(gate4)

          num = initWeightIter0(i).size(1) / 4
          gate1 = initWeightIter0(i).narrow(1, 1, num)
          gate3 = initWeightIter0(i).narrow(1, num + 1, num)
          gate2 = initWeightIter0(i).narrow(1, num * 2 + 1, num)
          gate4 = initWeightIter0(i).narrow(1, num * 3 + 1, num)
          initWeightIter(i).select(1, 1).copy(gate1)
          initWeightIter(i).select(1, 2).copy(gate2)
          initWeightIter(i).select(1, 3).copy(gate3)
          initWeightIter(i).select(1, 4).copy(gate4)

          num = initBias0(i).size(1) / 4
          gate1 = initBias0(i).narrow(1, 1, num)
          gate3 = initBias0(i).narrow(1, num + 1, num)
          gate2 = initBias0(i).narrow(1, num * 2 + 1, num)
          gate4 = initBias0(i).narrow(1, num * 3 + 1, num)
          initBias(i).select(1, 1).copy(gate1)
          initBias(i).select(1, 2).copy(gate2)
          initBias(i).select(1, 3).copy(gate3)
          initBias(i).select(1, 4).copy(gate4)
        }
        val weights = lstmDnn.parameters()._1
        weights(0).copy(initWeight.transpose(2, 4).transpose(3, 4))
        weights(1).copy(initBias)
        weights(2).copy(initWeightIter.transpose(2, 4).transpose(3, 4))

        return lstmDnn
      }
    }
    BlasWrapper(node.getOp().asInstanceOf[IRGeneralModule[Float]].model)
  }

  private def fromInput(node: IRElement[Float]) : Module[Float] = {
    mkldnn.Identity[Float]()
  }

  private def checkRequirement(layer: IRElement[Float]) : Boolean = {
    try {
      layer.getOp() match {
        case conv: IRSpatialConvolution[Float] =>
          require(conv.format == DataFormat.NCHW)
        case maxPool: IRSpatialMaxPooling[Float] =>
          require(maxPool.format == DataFormat.NCHW)
        case avgPool: IRSpatialAveragePooling[Float] =>
          require(avgPool.format == DataFormat.NCHW)
        case sbn: IRSpatialBatchNormalization[Float] =>
          require(sbn.dataFormat == DataFormat.NCHW)
        case lrn: IRSpatialCrossMapLRN[Float] =>
          require(lrn.format == DataFormat.NCHW)
        case join: IRJoinTable[Float] =>
          require(join.nInputDims <= 0)
        case _ => null
      }
      true
    } catch {
      case e: Throwable => false
    }
  }
}

private[bigdl] object IRToDnn {
  def apply[T: ClassTag](implicit ev: TensorNumeric[T]): IRToDnn = new IRToDnn
}
