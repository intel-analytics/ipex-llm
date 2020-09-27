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
package com.intel.analytics.bigdl.utils.caffe

import java.io._

import caffe.Caffe
import caffe.Caffe._
import com.google.protobuf.TextFormat.ParseException
import com.google.protobuf.{CodedInputStream, GeneratedMessage, TextFormat}
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{FileReader, Table}
import org.apache.log4j.Logger

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

abstract class Customizable[T: ClassTag](implicit ev: TensorNumeric[T]) {
  var contexts: Map[String, Any] = _
  def convertor(layer: GeneratedMessage): Seq[ModuleNode[T]]

  def registerContext(name: String, context: Any): Unit = {
    if (contexts == null) {
      contexts = Map[String, Any]()
    }
    contexts += (name -> context)
  }
}
/**
 * An utility to load pre-trained caffe model from prototxt and binary
 * and convert it to BigDL equivalent modules
 * @param prototxtPath caffe model define prototxt path
 * @param modelPath    caffe serialized binary model path
 * @param matchAll     if match all modules with parameters
 * @param customizedConverters customized converter
 * @tparam T type
 */
class CaffeLoader[T: ClassTag](prototxtPath: String, modelPath: String,
  matchAll: Boolean = true,
  customizedConverters: mutable.HashMap[String, Customizable[T]] = null
)(implicit ev: TensorNumeric[T]) {

  private val logger = Logger.getLogger(getClass)

  private var netparam: Caffe.NetParameter = _
  private var name2LayerV1: Map[String, V1LayerParameter] = Map[String, V1LayerParameter]()
  private var name2LayerV2: Map[String, LayerParameter] = Map[String, LayerParameter]()

  private val layerConverter = new LayerConverter[T]()
  private val v1layerConverter = new V1LayerConverter[T]()

  private val criterions = ParallelCriterion[T]()

  private def registerCustomizedConverter(): Unit = {
    if (customizedConverters != null) {
      customizedConverters.foreach(entry => {
        layerConverter.registerCutomizedConverter(entry._1, entry._2.convertor)
        v1layerConverter.registerCutomizedConverter(entry._1, entry._2.convertor)
        entry._2.registerContext("name2LayerV1", name2LayerV1)
        entry._2.registerContext("name2LayerV2", name2LayerV2)
        entry._2.registerContext("netparam", netparam)
      })
    }
  }

  private def loadCaffe(prototxtPath: String, modelPath: String): Unit = {
    if (null == netparam) {
      netparam = loadBinary(prototxtPath, modelPath)
      import scala.collection.JavaConverters._
      // V1LayerParameter
      netparam.getLayersList.asScala.foreach(layer => name2LayerV1 += (layer.getName -> layer))
      // V2LayerParameter
      netparam.getLayerList.asScala.foreach(layer => name2LayerV2 += (layer.getName -> layer))
    }
  }

  private def loadBinary(prototxtPath: String, modelPath: String): Caffe.NetParameter = {
    var modelFr: FileReader = null
    var prototxtFr: FileReader = null
    var modelStream: InputStream = null
    var prototxtStream: InputStream = null
    var prototxtReader: InputStreamReader = null
    try {
      modelFr = FileReader(modelPath)
      prototxtFr = FileReader(prototxtPath)
      modelStream = modelFr.open()
      prototxtStream = prototxtFr.open()
      prototxtReader = new InputStreamReader(prototxtStream, "ASCII")

      val netBuilder = NetParameter.newBuilder
      TextFormat.merge(prototxtReader, netBuilder)
      logger.info(s"start loading caffe model from $modelPath")
      val cis = CodedInputStream.newInstance(modelStream)
      cis.setSizeLimit(Integer.MAX_VALUE)
      val weightBuilder = NetParameter.newBuilder
      weightBuilder.mergeFrom(cis)
      logger.info("load caffe model done")
      mergeNetWithParams(netBuilder.build, weightBuilder.build)
    } finally {
      if (null != prototxtReader) prototxtReader.close()
      if (null != modelStream) modelStream.close()
      if (null != prototxtStream) prototxtStream.close()
      if (modelFr != null) modelFr.close()
      if (prototxtFr != null) prototxtFr.close()
    }
  }

  private def mergeNetWithParams(net : NetParameter, weights : NetParameter): NetParameter = {

    val builder = NetParameter.newBuilder(net)
    val layers = new mutable.HashMap[String, GeneratedMessage]
    val v1Layers = new ArrayBuffer[V1LayerParameter]
    val v2Layers = new ArrayBuffer[LayerParameter]

    net.getLayersList.asScala.foreach(v1Layer => v1Layers.append(v1Layer))
    net.getLayerList.asScala.foreach(v2Layer => v2Layers.append(v2Layer))

    weights.getLayersList.asScala.foreach(v1Layer => layers(v1Layer.getName) = v1Layer)
    weights.getLayerList.asScala.foreach(v2Layer => layers(v2Layer.getName) = v2Layer)

    builder.clearLayers
    builder.clearLayer

    v1Layers.foreach(v1Layer => {
      val name = v1Layer.getName
      if (layers.contains(name)) {
        val weightLayer = layers(name)
        builder.addLayers(copyBlobs(weightLayer, v1Layer).asInstanceOf[V1LayerParameter])
      } else {
        builder.addLayers(v1Layer)
        if (customizedConverters ==null ||
          !customizedConverters.contains(v1Layer.getType.toString.toUpperCase)) {
          logger.warn(s"layer $name if type ${v1Layer.getType.toString}" +
            s"does not exist in weight file")
        }
      }
    })

    v2Layers.foreach(v2Layer => {
      val name = v2Layer.getName
      if (layers.contains(name)) {
        val weightLayer = layers(name)
        builder.addLayer(copyBlobs(weightLayer, v2Layer).asInstanceOf[LayerParameter])
      } else {
        builder.addLayer(v2Layer)
        if (customizedConverters ==null ||
          !customizedConverters.contains(v2Layer.getType.toUpperCase)) {
          logger.warn(s"layer $name if type ${v2Layer.getType} does not exist in weight file")
        }
      }
    })
    builder.build
  }

  private def copyBlobs(from : GeneratedMessage, to : GeneratedMessage): GeneratedMessage = {
    import scala.language.existentials
    val blobList = from match {
      case v1 : V1LayerParameter => v1.asInstanceOf[V1LayerParameter].getBlobsList.asScala
      case v2 : LayerParameter => v2.asInstanceOf[LayerParameter].getBlobsList.asScala
    }
    val layer = to match {
      case v1 : V1LayerParameter =>
        val layerBuilder = V1LayerParameter.newBuilder(to.asInstanceOf[V1LayerParameter])
        layerBuilder.clearBlobs
        blobList.foreach(blob => layerBuilder.addBlobs(blob))
        layerBuilder.build
      case v2 : LayerParameter =>
        val layerBuilder = LayerParameter.newBuilder(to.asInstanceOf[LayerParameter])
        layerBuilder.clearBlobs
        blobList.foreach(blob => layerBuilder.addBlobs(blob))
        layerBuilder.build
    }
    layer.asInstanceOf[GeneratedMessage]
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
    // for Bias layer, there is no weight but has bias so the bias is the first blob
    var index = 0
    val caffeWeight = getBlob(name, index)
    if (caffeWeight.isDefined && params.contains("weight")) {
      index += 1
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

    val caffeBias = getBlob(name, index)
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
      case _ =>
        throw new UnsupportedOperationException("unsupported $name and $params")
    }
    model
  }

  private def copyParameter(name: String, params: Table): Unit = {
    if (params == null || (!params.contains("weight") && !params.contains("bias"))) return
    if (!name2LayerV2.contains(name) && !name2LayerV1.contains(name)) {
      if (matchAll) throw new CaffeConversionException(s"module $name " +
        s"cannot map a layer in caffe model")
      logger.info(s"$name uses initialized parameters")
      return
    }
    loadParameters(name, params)
  }

/**
 * Load caffe model from prototxt file and binary pre-trained model and converted
 * to BigDL graph module
 * @param outputNames additional output layer names besides the default(layers without next nodes)
 * @return BigDL model and criterion
 */
  def createCaffeModel(outputNames: Array[String] = Array[String]())
  : (Module[T], ParallelCriterion[T]) = {
    loadCaffe(prototxtPath, modelPath)
    registerCustomizedConverter()
    val layers = createLayers(outputNames)
    val inputs = layers.filter(layer => layer.prevNodes.isEmpty).toArray
    val outputs = layers.filter(layer => layer.nextNodes.isEmpty ||
      outputNames.contains(layer.element.getName())).toArray
    val module = Graph(inputs, outputs)
    module.setName(netparam.getName)
    copyParameters(module)
    (module, criterions)
  }

  private val dataLayerList = Array("INPUT", "DATA", "DUMMYDATA", "ANNOTATEDDATA", "MEMORYDATA")

  private def tryConvertInput(layer: GeneratedMessage, layerType: String,
    layers: ArrayBuffer[ModuleNode[T]],
    top2LayerMap: mutable.HashMap[String, String],
    layersMap: mutable.HashMap[String, ModuleNode[T]]): Boolean = {
    val inputs = if (dataLayerList.contains(layerType)) convertCaffeLayer(layer) else null
    addInputList(inputs, layers, top2LayerMap, layersMap)
  }

  // try to get input list (without data layer)
  private def tryConvertInput(netparam: Caffe.NetParameter,
    layers: ArrayBuffer[ModuleNode[T]],
    top2LayerMap: mutable.HashMap[String, String],
    layersMap: mutable.HashMap[String, ModuleNode[T]]): Boolean = {
    val inputNames = netparam.getInputList
    val inputs = if (!inputNames.isEmpty) {
      (0 until inputNames.size()).map(i => {
        val input = Input()
        input.element.setName(inputNames.get(i))
        input
      })
    } else {
      null
    }
    addInputList(inputs, layers, top2LayerMap, layersMap)
  }

  private def addInputList(inputs: Seq[ModuleNode[T]],
    layers: ArrayBuffer[ModuleNode[T]],
    top2LayerMap: mutable.HashMap[String, String],
    layersMap: mutable.HashMap[String, ModuleNode[T]]): Boolean = {
    if (null != inputs) {
      inputs.foreach(input => {
        top2LayerMap(input.element.getName()) = input.element.getName()
        layersMap(input.element.getName()) = input
        layers.append(input)
      })
      true
    } else {
      false
    }
  }

  // create directed graph based on the module relationships
  private def createLayers(outputNames: Array[String]) : ArrayBuffer[ModuleNode[T]] = {
    val layers = ArrayBuffer[ModuleNode[T]]()
    val layersMap = new mutable.HashMap[String, ModuleNode[T]]()
    val top2LayerMap = new mutable.HashMap[String, String]()
    val splitLayerMap = new mutable.HashMap[String, ModuleNode[T]]()
    val allLayers = ArrayBuffer[GeneratedMessage]()
    if (netparam.getLayersList.size > 0 ) {
      // filter out those layers from prototxt but also occurs in binary
      val localMap = new mutable.HashMap[String, Int]()
      var i = 0
      netparam.getLayersList.asScala.
        foreach(layer => {
          if (!localMap.contains(layer.getName)) {
            allLayers.append(layer)
            localMap(layer.getName) = i
            i += 1
          } else {
            allLayers.update(localMap(layer.getName), layer)
          }
        })
    } else {
      // filter out those layers from prototxt but also occurs in binary
      val localMap = new mutable.HashMap[String, Int]()
      var i = 0
      netparam.getLayerList.asScala.
        foreach(layer => {
          if (!localMap.contains(layer.getName)) {
            allLayers.append(layer)
            localMap(layer.getName) = i
            i += 1
          } else {
            allLayers.update(localMap(layer.getName), layer)
          }
        })
    }

    tryConvertInput(netparam, layers, top2LayerMap, layersMap)
    allLayers.foreach(layer => {
      var name : String = null
      val topList = new ArrayBuffer[String]()
      val bottomList = new ArrayBuffer[String]()
      layer match {
        case v2 : LayerParameter =>
          name = v2.getName
          topList ++= v2.getTopList.asScala
          bottomList ++= v2.getBottomList.asScala
        case v1 : V1LayerParameter =>
          name = v1.getName
          topList ++= v1.getTopList.asScala
          bottomList ++= v1.getBottomList.asScala
      }
      val layerType = getLayerType(name).get.toUpperCase
      if ("SPLIT" == layerType) {
        // eliminate split layer in graph module, cache dependency only
        require(bottomList.size == 1, s"split dependency should only be one!")
        topList.foreach(top => {
          if (top2LayerMap.contains(bottomList(0))) {
            splitLayerMap(top) = layersMap(top2LayerMap(bottomList(0)))
          }
        })
      } else {
        // some criterion layers are not only for loss calculation,
        // we need to separate it with loss function and module
        val isCriterionLayerOnly: Boolean = tryAddCriterion(layerType, name)
        val isInput = if (!isCriterionLayerOnly) {
          tryConvertInput(layer, layerType, layers, top2LayerMap, layersMap)
        } else false

        if (!isCriterionLayerOnly && !isInput) {
          val nodes = convertCaffeLayer(layer)
          if (nodes != null) {
            var curr = nodes.head
            bottomList.foreach(dependency => {
              if (top2LayerMap.contains(dependency)) {
                layersMap(top2LayerMap(dependency)) -> curr
              }
            })
            while (curr.nextNodes.nonEmpty) {
              layers.append(curr)
              curr = curr.nextNodes.head
            }
            layers.append(curr)
            layersMap(name) = curr
            topList.foreach(output => {
              top2LayerMap(output) = name
            })
          }
        }
      }
    })
    // process with split separately in case of out of order
    allLayers.foreach(layer => {
      var name : String = null
      val bottomList = new ArrayBuffer[String]()
      layer match {
        case v2 : LayerParameter =>
          name = v2.getName
          bottomList ++= v2.getBottomList.asScala
        case v1 : V1LayerParameter =>
          name = v1.getName
          bottomList ++= v1.getBottomList.asScala
      }
      bottomList.foreach(bottom => {
        if (splitLayerMap.contains(bottom)) {
          splitLayerMap(bottom) -> layersMap(name)
        }
      })
    })
    layers.filter(layer => !(layer.prevNodes.isEmpty && layer.nextNodes.isEmpty)
    || outputNames.contains(layer.element.getName))
  }

  private def convertCaffeLayer(layer : GeneratedMessage): Seq[ModuleNode[T]] = {
    val node = if (layer.isInstanceOf[LayerParameter]) {
      layerConverter.convertLayerFromCaffe(layer)
    }
    else {
      v1layerConverter.convertLayerFromCaffe(layer)
    }
    node
  }

  /**
   * Add criterion according to layer type from train protocol
   * if only test/model define prototxt file provided, there won't be criterion detected
   *  @param layerType caffe layer type
   *  @param layerName caffe layer name
   *  @return if this layer is only criterion layer
   */
  private def tryAddCriterion(layerType : String, layerName: String = null) : Boolean = {
    layerType.toUpperCase match {
      case "SOFTMAX_LOSS" => criterions.add(ClassNLLCriterion[T]())
        false
      case "SOFTMAXWITHLOSS" => criterions.add(ClassNLLCriterion[T]())
        false
      case "EUCLIDEANLOSS" => criterions.add(MSECriterion[T]())
        true
      case "HINGELOSS" => criterions.add(HingeEmbeddingCriterion[T]())
        true
      case "SIGMOIDCROSSENTROPYLOSS" => criterions.add(CrossEntropyCriterion[T]())
        false
      case "INFOGAINLOSS" => criterions.add(createInfoGainCriterion(layerName))
        true
      case "CONTRASTIVELOSS" => criterions.add(CosineEmbeddingCriterion[T]())
        true
      case _ => false
    }
  }

  private def createInfoGainCriterion(layerName : String) : ClassNLLCriterion[T] = {
    val param = getInforgainParam(layerName).get
    val weightBlob = getBlob(layerName, 2)
    if (weightBlob.isDefined) {
      val size = weightBlob.get.getShape.getDimList.asScala.map(_.toInt).toArray
      val weightData = weightBlob.get.getDataList
      val weightArr = new Array[T](weightData.size)
      var i = 0
      while (i < weightData.size) {
        weightArr(i) = ev.fromType[Float](weightData.get(i))
        i += 1
      }
      val weightTensor = Tensor(weightArr, size)
      ClassNLLCriterion[T](weightTensor)
    } else {
      ClassNLLCriterion[T]()
    }
  }

  private def getInforgainParam(name: String): Option[InfogainLossParameter] = {
    if (name2LayerV2.contains(name)) {
      Some(name2LayerV2(name).getInfogainLossParam)
    } else if (name2LayerV1.contains(name)) {
      Some(name2LayerV1(name).getInfogainLossParam)
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
}

object CaffeLoader {

  /**
   * Load weight for pre-defined model
   * @param model pre-defined model
   * @param defPath prototxt file which defines the network
   * @param modelPath weight file which contains the parameters
   * @param matchAll  if we need to match all layers from prototxt in weight file
   * @param customizedConverters customized converters
   * @param ev tensor numeric
   * @tparam T data type
   * @return pre-defined model populated with weights
   */
  def load[T: ClassTag](model: Module[T],
                        defPath: String, modelPath: String, matchAll: Boolean = true,
                        customizedConverters : mutable.HashMap[String, Customizable[T]] = null)
                       (implicit ev: TensorNumeric[T]): Module[T] = {
    val caffeLoader = new CaffeLoader[T](defPath, modelPath, matchAll, customizedConverters)
    caffeLoader.copyParameters(model)
  }

/**
 * load caffe model dynamically from prototxt and binary files
 * @param defPath prototxt file which illustrates the caffe model structure
 * @param modelPath binary file containing the weight and bias
 * @param customizedConverters customized layer converter
 * @param outputNames additional output layer names besides the default(layers without next nodes)
 * @tparam T data type
 * @return created module (graph) and criterion
 */
  def loadCaffe[T: ClassTag](defPath: String, modelPath: String,
    customizedConverters : mutable.HashMap[String, Customizable[T]] = null,
    outputNames: Array[String] = Array[String]())
                              (implicit ev: TensorNumeric[T]): (Module[T], ParallelCriterion[T]) = {
    try {
      val caffeLoader = new CaffeLoader[T](defPath, modelPath, true, customizedConverters)
      caffeLoader.createCaffeModel(outputNames)
    } catch {
      case parseException : ParseException =>
        throw new CaffeConversionException("Parsing caffe model error," +
          "only standard Caffe format is supported"
          , parseException)
      case conversionExcepion : CaffeConversionException =>
        throw  conversionExcepion
    }
  }
}
