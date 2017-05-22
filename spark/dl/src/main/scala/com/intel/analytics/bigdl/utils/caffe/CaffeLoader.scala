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

import scala.collection.JavaConverters._
import caffe.Caffe
import caffe.Caffe.EltwiseParameter.EltwiseOp
import caffe.Caffe.PoolingParameter.PoolMethod
import caffe.Caffe._
import com.google.protobuf.{CodedInputStream, GeneratedMessage, TextFormat}
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractCriterion, AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
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

  val hdfsPrefix: String = "hdfs:"

  val logger = Logger.getLogger(getClass)

  private var netparam: Caffe.NetParameter = _
  private var name2LayerV1: Map[String, V1LayerParameter] = _
  private var name2LayerV2: Map[String, LayerParameter] = _

  private val layerConverter = new LayerConverter[T]()
  private val v1layerConverter = new V1LayerConverter[T]()

  private var criterions = ParallelCriterion[T]()

  def setCustomizedLayer(map : Map[String, AbstractModule[Activity, Activity, T]]) : Unit = {
    layerConverter.setCustomizedLayer(map)
    v1layerConverter.setCustomizedLayer(map)
  }
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
    var reader : InputStreamReader = null
    if (prototxtPath.startsWith(hdfsPrefix)) {
      val byteArrayOut = com.intel.analytics.bigdl.utils.File.readHdfsByte(prototxtPath)
      reader = new InputStreamReader(new ByteArrayInputStream(byteArrayOut))
    } else {
      val f = new java.io.File(prototxtPath)
      require(f.exists(), prototxtPath + " does not exists")
      reader = new InputStreamReader(new FileInputStream(f), "ASCII")
    }
    val builder = NetParameter.newBuilder
    TextFormat.merge(reader, builder)
    logger.info(s"start loading caffe model from $modelPath")
    var cis : CodedInputStream = null
    if (modelPath.startsWith(hdfsPrefix)) {
      val byteArrayOut = com.intel.analytics.bigdl.utils.File.readHdfsByte(modelPath)
      cis = CodedInputStream.newInstance(new ByteArrayInputStream(byteArrayOut))
    } else {
      cis = CodedInputStream.newInstance(new FileInputStream(modelPath))
    }
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

  /**
   * Load caffe model from prototxt file and binary pre-trained model and converted
   * to BigDL graph module
   * Pre-defined module structure is not needed, it will be created dynamically
   */
  def createCaffeModel(): (Module[T], ParallelCriterion[T]) = {
    loadCaffe(prototxtPath, modelPath)
    val layers = createLayers()
    val inputs = layers.filter(layer => layer.prevNodes.size == 0).toArray
    val outputs = layers.filter(layer => layer.nextNodes.size == 0).toArray
    val module = Graph(inputs, outputs)
    module.setName(netparam.getName)
    copyParameters(module)
    (module, criterions)
  }

  private def createLayers() : ArrayBuffer[ModuleNode[T]] = {
    val layers = ArrayBuffer[ModuleNode[T]]()
    val layersMap = new mutable.HashMap[String, ModuleNode[T]]()
    val top2LayerMap = new mutable.HashMap[String, String]()
    val splitLayerMap = new mutable.HashMap[String, ModuleNode[T]]()
    val allLayers = ArrayBuffer[GeneratedMessage]()
    if (netparam.getLayersList.size > 0 ) {
      netparam.getLayersList.asScala.foreach(layer => allLayers.append(layer))
    } else {
      // filter is to filter out layers from prototxt, this happens in V2 layer
      netparam.getLayerList.asScala.filter(layer => layer.getBlobsCount > 0)
      .foreach(layer => allLayers.append(layer))
    }
    allLayers.foreach(layer => {
      var name : String = s""
      var topList = new ArrayBuffer[String]()
      val bottomList = new ArrayBuffer[String]()
      if (layer.isInstanceOf[LayerParameter]) {
        val layerParameter = layer.asInstanceOf[LayerParameter]
        name = layerParameter.getName
        layerParameter.getTopList.asScala.foreach(top => topList.append(top))
        layerParameter.getBottomList.asScala.foreach(bottom => bottomList.append(bottom))
      } else {
        val layerParameter = layer.asInstanceOf[V1LayerParameter]
        name = layerParameter.getName
        layerParameter.getTopList.asScala.foreach(top => topList.append(top))
        layerParameter.getBottomList.asScala.foreach(bottom => bottomList.append(bottom))
      }
      val layerType = getLayerType(name).get.toUpperCase
      if ("SPLIT".equals(layerType)) {
        // eliminate split layer in graph module, cache dependency only
        require(bottomList.size == 1, s"split dependency should only be one!")
        topList.foreach(top => {
          if (top2LayerMap.contains(bottomList(0))) {
            splitLayerMap(top) = layersMap(top2LayerMap(bottomList(0)))
          }
        })
      } else {
        val isCriterionLayerOnly : Boolean = tryAddCriterion(name, layerType)
        if (!isCriterionLayerOnly) {
          var node = convertCaffeLayer(layer)
          if (node != null) {
            bottomList.foreach(dependency => {
              if (splitLayerMap.contains(dependency)) splitLayerMap(dependency) -> node
              else if (top2LayerMap.contains(dependency)) {
                layersMap(top2LayerMap(dependency)) -> node
              }
            })
            while (node.nextNodes.size != 0) {
              layers.append(node)
              node = node.nextNodes(0)
            }
            layers.append(node)
            layersMap(name) = node
            topList.foreach(output => {
              top2LayerMap(output) = name
            })
          }
        }
      }
    })
    return layers
  }

  private def convertCaffeLayer(layer : GeneratedMessage): ModuleNode[T] = {
    val node = if (layer.isInstanceOf[LayerParameter]) {
      layerConverter.convertLayerFromCaffe(layer)
    }
    else {
      v1layerConverter.convertLayerFromCaffe(layer)
    }
    node
  }

  /*
 * Add criterion according to layer type from train protocol
 * if only test/model define prototxt file provided, there won't be criterion detected
 */
  private def tryAddCriterion(layerType : String, layerName: String = null) : Boolean = {
    layerType match {
      case "SOFTMAX_LOSS" => criterions.add(ClassNLLCriterion[T]())
        false
      case "SoftmaxWithLoss" => criterions.add(ClassNLLCriterion[T]())
        false
      case "EuclideanLoss" => criterions.add(MSECriterion[T]())
        true
      case "HingeLoss" => criterions.add(HingeEmbeddingCriterion[T]())
        true
      case "SigmoidCrossEntropyLoss" => criterions.add(CrossEntropyCriterion[T]())
        false
      case "InfogainLoss" => criterions.add(createInfoGainCriterion(layerName))
        true
      case "ContrastiveLoss" => criterions.add(CosineEmbeddingCriterion[T]())
        true
      case _ => false
    }
  }

  private def createInfoGainCriterion(layerName : String) : ClassNLLCriterion[T] = {
    val param = getInforgainParam(layerName).get
    val weightBlob = getBlob(layerName, 2)
    if (weightBlob.isDefined) {
      val size = weightBlob.get.getShape.getDimList.toArray.asInstanceOf[Array[Int]]
      val weightData = weightBlob.get.getDataList
      var weightArr = new Array[T](weightData.size)
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

  def load[T: ClassTag](model: Module[T],
    defPath: String, modelPath: String, matchAll: Boolean = true)
    (implicit ev: TensorNumeric[T]): Module[T] = {
    val caffeLoader = new CaffeLoader[T](defPath, modelPath, matchAll)
    caffeLoader.copyParameters(model)
  }

  def loadDynamic[T: ClassTag](defPath: String, modelPath: String, matchAll: Boolean = true)
                       (implicit ev: TensorNumeric[T]): (Module[T], ParallelCriterion[T]) = {
    val caffeLoader = new CaffeLoader[T](defPath, modelPath, matchAll)
    caffeLoader.createCaffeModel()
  }
}
