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

package com.intel.analytics.bigdl.utils

import java.io.{File, FileInputStream, InputStreamReader}

import scala.collection.JavaConverters._
import caffe.Caffe
import caffe.Caffe._
import com.google.protobuf.{CodedInputStream, TextFormat}
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.{ReLU, SpatialConvolution, SpatialCrossMapLRN}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.log4j.Logger

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * load caffe model parameters to defined bigdl model
 * @param prototxtPath caffe model define prototxt path
 * @param modelPath    caffe serialized model path
 * @param matchAll     if match all modules with parameters
 * @tparam T type
 */
class CaffeLoader[T: ClassTag](prototxtPath: String, modelPath: String,
  matchAll: Boolean = true
)(implicit ev: TensorNumeric[T]) {

  val logger = Logger.getLogger(getClass)

  private var netparam: Caffe.NetParameter = _
  private var name2LayerV1: Map[String, V1LayerParameter] = _
  private var name2LayerV2: Map[String, LayerParameter] = _

  private def loadCaffe(prototxtPath: String, modelPath: String): Unit = {
    if (name2LayerV2 == null) {
      netparam = loadBinary(prototxtPath, modelPath)
      name2LayerV2 = Map[String, LayerParameter]()
      name2LayerV1 = Map[String, V1LayerParameter]()
      import scala.collection.JavaConverters._
      // V1LayerParameter
      netparam.getLayersList.asScala.foreach(layer => name2LayerV1 += (layer.getName -> layer))
      // V2LayerParameter
      netparam.getLayerList.asScala.foreach(layer => name2LayerV2 += (layer.getName -> layer))
    }
  }

  private def loadBinary(prototxtPath: String, modelPath: String): Caffe.NetParameter = {
    val f = new java.io.File(prototxtPath)
    require(f.exists(), prototxtPath + " does not exists")
    val reader = new InputStreamReader(new FileInputStream(f), "ASCII")
    val builder = NetParameter.newBuilder
    TextFormat.merge(reader, builder)
    logger.info(s"start loading caffe model from $modelPath")
    val cis = CodedInputStream.newInstance(new FileInputStream(modelPath))
    cis.setSizeLimit(Integer.MAX_VALUE)
    builder.mergeFrom(cis)
    logger.info("load caffe model done")
    builder.build()
  }

  private def getBlob(name: String, ind: Int): Option[Caffe.BlobProto] = {
    if (name2LayerV2.contains(name) && name2LayerV2(name).getBlobsCount > ind) {
      Some(name2LayerV2(name).getBlobs(ind))
    } else if (name2LayerV1.contains(name) && name2LayerV1(name).getBlobsCount > ind) {
      Some(name2LayerV1(name).getBlobs(ind))
    } else {
      None
    }
  }

  private def loadParameters(name: String, params: Table): Unit = {
    logger.info(s"load parameters for $name ...")
    val caffeWeight = getBlob(name, 0)
    if (caffeWeight.isDefined) {
      require(params.contains("weight"), s"$name should contain weight")
      val caffeWeightData = caffeWeight.get.getDataList
      val weight = params[Tensor[T]]("weight")
      require(params != null && weight.nElement() == caffeWeightData.size(),
        s"weight element number is not equal between caffe layer and bigdl module $name, " +
          s"data shape in caffe is ${ caffeWeight.get.getShape() }," +
          s" while data shape in bigdl is ${ weight.size().mkString(",") }")
      var i = 0
      val weightData = weight.storage().array()
      var offset = weight.storageOffset() - 1
      while (i < caffeWeightData.size()) {
        weightData(offset) = ev.fromType[Float](caffeWeightData.get(i))
        offset += 1
        i += 1
      }
    }

    val caffeBias = getBlob(name, 1)
    if (caffeBias.isDefined) {
      require(params.contains("bias"), s"$name should contain bias")
      val caffeBiasList = caffeBias.get.getDataList
      val bias = params[Tensor[T]]("bias")
      require(bias.nElement() == caffeBiasList.size(),
        s"bias element number is not equal between caffe layer and bigdl module $name, " +
          s"data shape in caffe is ${ caffeBias.get.getShape() }," +
          s" while data shape in bigdl is ${ bias.size().mkString(",") }")
      var i = 0
      val biasData = bias.storage().array()
      var offset = bias.storageOffset() - 1
      while (i < caffeBiasList.size()) {
        biasData(offset) = ev.fromType[Float](caffeBiasList.get(i))
        offset += 1
        i += 1
      }
    }
  }

  /**
   * copy caffe parameters to module
   * if matchAll, throw an exception if some layers are not mapped
   * @param model the model defined in big-dl
   * @return
   */
  private def copyParameters(model: Module[T]): Module[T] = {
    loadCaffe(prototxtPath, modelPath)
    val parameterTable = model.getParametersTable()

    parameterTable.foreach {
      case (name: String, params: Table) =>
        copyParameter(name, params)
    }
    model
  }

  private def copyParameter(name: String, params: Table): Unit = {
    if (params == null || (!params.contains("weight") && !params.contains("bias"))) return
    if (!name2LayerV2.contains(name) && !name2LayerV1.contains(name)) {
      if (matchAll) throw new Exception(s"module $name cannot map a layer in caffe model")
      logger.info(s"$name uses initialized parameters")
      return
    }
    loadParameters(name, params)
  }

  def createCaffeModel(): Module[T] = {
    val model = null
    loadCaffe(prototxtPath, modelPath)
    val caffeTypedLayers = getCaffeTypedList
    convert(caffeTypedLayers)
    model
  }

  private def convert(caffeTypedLayers : ArrayBuffer[Node[String]]) : ArrayBuffer[Module[T]] = {
    val layers = new ArrayBuffer[Module[T]]()
    caffeTypedLayers.foreach(layer => {
      val converted : Module[T] = convertCaffeLayer(layer)
      if (converted != null) layers.append(converted)
    })
    return layers
  }

  private def convertCaffeLayer(node : Node[String]): Module[T] = {
    val layerName = node.element
    val layerType = getLayerType(layerName).get
    layerType match {
      case "Convolution" => fromCaffeConvolution(layerName)
      case "ReLU" => fromCaffeReLU(layerName)
      case "LRN" => fromCaffeLRN(layerName)
      case "Input" =>
      case _ =>
    }

    null
  }

  private def fromCaffeLRN(layerName : String) : Module[T] = {
    val param = getLRNParam(layerName).get
    val localSize = param.getLocalSize
    val alpha = param.getAlpha
    val belta = param.getBeta
    val k = param.getK
    SpatialCrossMapLRN(localSize, alpha, belta, k)
  }

  private def fromCaffeReLU(layerName : String) : Module[T] = {
    ReLU(true).setName(layerName)
  }

  private def fromCaffeConvolution(layerName : String) : Module[T] = {
    print(layerName + ": ")
    val param = getConvolutionParam(layerName).get
    val weightBlob = getBlob(layerName, 0).get
    val group = if (param.getGroup == 0)  1 else param.getGroup
    val nInputPlane = weightBlob.getChannels * group
    val nOutPlane = weightBlob.getNum
    var kw = param.getKernelW
    var kh = param.getKernelH
    var dw = param.getStrideW
    var dh = param.getStrideH
    if (kw ==0 || kh == 0) {
      kw = param.getKernelSize(0)
      kh = kw
    }
    if (dw == 0 || dh == 0) {
      if (param.getStrideList.size() != 0) {
        dw = param.getStride(0)
        dh = dw
      } else {
        // use default values if not found
        dw = 1
        dh = 1
      }

    }
    var pw = param.getPadW
    var ph = param.getPadH
    if (pw == 0 || ph == 0) {
      if (param.getPadList.size() != 0) {
        pw = param.getPad(0)
        ph = pw
      }
    }
    print(" " + nInputPlane)
    print(" " + nOutPlane)
    print(" " + kw)
    print(" " + kh)
    print(" " + dw)
    print(" " + dh)
    print(" " + pw)
    print(" " + ph)
    print(" " + group)
    println()
    SpatialConvolution[T](nInputPlane, nOutPlane, kw, kh, dw, dh, pw, ph, group).setName(layerName)
  }

  private def getLRNParam(name: String): Option[LRNParameter] = {
    if (name2LayerV2.contains(name)) {
      Some(name2LayerV2(name).getLrnParam)
    } else if (name2LayerV1.contains(name)) {
      Some(name2LayerV1(name).getLrnParam)
    } else {
      None
    }
  }

  private def getConvolutionParam(name: String): Option[ConvolutionParameter] = {
    if (name2LayerV2.contains(name)) {
      Some(name2LayerV2(name).getConvolutionParam)
    } else if (name2LayerV1.contains(name)) {
      Some(name2LayerV1(name).getConvolutionParam)
    } else {
      None
    }
  }

  private def getLayerType(name: String): Option[String] = {
    if (name2LayerV2.contains(name)) {
      Some(name2LayerV2(name).getType)
    } else if (name2LayerV1.contains(name)) {
      Some(name2LayerV1(name).getType.toString)
    } else {
      None
    }
  }

  private def getCaffeTypedList() : ArrayBuffer[Node[String]] = {
    val list = ArrayBuffer[Node[String]]()
    val layersMap = new mutable.HashMap[String, Node[String]]()
    val top2LayerMap = new mutable.HashMap[String, String]()
    netparam.getLayersList.asScala.foreach(layer => {
      val name = layer.getName
      val node = new Node(name)
      list.append(node)
      layersMap(name) = node
      val dependencies = layer.getBottomList.asScala
      dependencies.foreach(dependency => {
        val dependentNode = layersMap(top2LayerMap(dependency))
        dependentNode -> node
      })
      val outputs = layer.getTopList.asScala
      outputs.foreach(output => {
        top2LayerMap(output) = name
      })
    })
    return list
  }
/*
  private def buildCaffeTypedGraph() : DirectedGraph[String] = {
    val dummySource = new Node[String](null)
    val layersMap = new mutable.HashMap[String, Node[String]]()
    val top2LayerMap = new mutable.HashMap[String, String]()
    netparam.getLayersList.asScala.foreach(layer => {
      val name = layer.getName
      val node = new Node(name)
      layersMap(name) = node
      val dependencies = layer.getBottomList.asScala
      dependencies.foreach(dependency => {
        val dependentNode = layersMap(top2LayerMap(dependency))
        dependentNode -> node
      })
      val outputs = layer.getTopList.asScala
      outputs.foreach(output => {
        top2LayerMap(output) = name
      })
    })
    val zeroInDegreeNodes = layersMap.values.filter(node => {node.prevNodes.size == 0})
    zeroInDegreeNodes.foreach(node => {
      dummySource -> node
    })
    return new DirectedGraph(dummySource)
  }
  */

}

object CaffeLoader {

  def load[T: ClassTag](model: Module[T],
    defPath: String, modelPath: String, matchAll: Boolean = true)
    (implicit ev: TensorNumeric[T]): Module[T] = {
    val caffeLoader = new CaffeLoader[T](defPath, modelPath, matchAll)
    caffeLoader.copyParameters(model)
  }

  def loadDynamic[T: ClassTag](defPath: String, modelPath: String, matchAll: Boolean = true)
                       (implicit ev: TensorNumeric[T]): Module[T] = {
    val caffeLoader = new CaffeLoader[T](defPath, modelPath, matchAll)
    caffeLoader.createCaffeModel()
  }
}
