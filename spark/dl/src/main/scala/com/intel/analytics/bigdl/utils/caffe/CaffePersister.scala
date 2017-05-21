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

import java.io.{FileInputStream, FileOutputStream, InputStreamReader, OutputStreamWriter}

import scala.collection.JavaConverters._
import caffe.Caffe.{LayerParameter, NetParameter, V1LayerParameter}
import com.intel.analytics.bigdl.nn.{Graph, View}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Node

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import caffe.Caffe
import com.google.protobuf.{GeneratedMessage, TextFormat}
import org.apache.log4j.Logger
/**
 * An utility to convert BigDL model into caffe format
 *
 * @param prototxtPath  path to store model definition file
 * @param modelPath  path to store model weights and bias
 * @param module BigDL module to be converted
 * @param useV2  convert to V2 layer or not
 */
class CaffePersister[T: ClassTag](val prototxtPath: String,
      val modelPath: String, val module : Graph[T],
      useV2 : Boolean = true)(implicit ev: TensorNumeric[T]) {

  val logger = Logger.getLogger(getClass)

  val v1Converter = new V1LayerConverter[T]()
  val v2Converter = new LayerConverter[T]()
  var v1Layers : ArrayBuffer[Node[V1LayerParameter]] = new ArrayBuffer[Node[V1LayerParameter]]()
  var v2Layers : ArrayBuffer[Node[LayerParameter]] = new ArrayBuffer[Node[LayerParameter]]()

  private var netparam: NetParameter.Builder = NetParameter.newBuilder

  def saveAsCaffe() : Unit = {
    convertToCaffe()
    save()
  }
  private def createNet() : Unit = {
    netparam.toString
  }
  // create caffe layers graph based on BigDL execution plan
  private def convertToCaffe() : Unit = {
    val top2Layers = new mutable.HashMap[String, String]()
    val layers = new mutable.HashMap[String, GeneratedMessage]()
    val executions = module.getExecutions
    netparam.setName(module.getName)
    executions.foreach(execution => {
      val preModules = execution.prevNodes
      val module = execution.element
      logger.info(s"find dependencies for ${module.getName}")
      if (module.isInstanceOf[View[T]]) {
        require(preModules.size == 1, "view pre-node size should be 1")
        val preNode = preModules(0)
        val nextNodes = execution.nextNodes
        nextNodes.foreach(nextNode => {
          preNode -> nextNode
          execution.delete(nextNode)
        })
        preNode.delete(execution)
      } else {
        var bottomList = ArrayBuffer[String]()
        preModules.foreach(pre => {
          val name = pre.element.getName
          val preLayer = layers(name)
          val bottoms = if (useV2) preLayer.asInstanceOf[LayerParameter].getBottomList.asScala
            else preLayer.asInstanceOf[V1LayerParameter].getBottomList.asScala
          bottoms.foreach(bottom => bottomList.append(bottom))
        })
        bottomList = bottomList.filter(bottom => {
          val nextModule = top2Layers(bottom)
          nextModule.equals(execution.element.getName)
        })
        val nextModules = execution.nextNodes.filter(_.element != null)
        if (useV2) {
          val caffeLayer = v2Converter.toCaffe(execution, bottomList).asInstanceOf[LayerParameter]
          val caffeNode = new Node(caffeLayer)
          val topList = caffeLayer.getTopList.asScala
          var i = 0
          while (i < nextModules.size) {
            top2Layers(topList(i)) = nextModules(i).element.getName()
            i += 1
          }
          v2Layers.append(caffeNode)
          layers(caffeLayer.getName) = caffeLayer
          netparam.addLayer(caffeLayer)
        } else {
          val caffeLayer = v1Converter.toCaffe(execution, bottomList).asInstanceOf[V1LayerParameter]
          val caffeNode = new Node(caffeLayer)
          val topList = caffeLayer.getTopList.asScala
          var i = 0
          while (i < nextModules.size) {
            top2Layers(topList(i)) = nextModules(i).element.getName()
            i += 1
          }
          v1Layers.append(caffeNode)
          layers(caffeLayer.getName) = caffeLayer
          netparam.addLayers(caffeLayer)
        }
      }
    })
  }

  private def save() : Unit = {

    // save binary
    val binaryFile = new java.io.File(modelPath)
    val binaryWriter = new FileOutputStream(binaryFile)
    binaryWriter.write(netparam.build.toByteArray)
    binaryWriter.close
    // save prototxt
    val netParameterWithoutData = NetParameter.newBuilder
    netParameterWithoutData.setName(netparam.getName)
    val prototxtFile = new java.io.File(prototxtPath)
    val prototxtWriter = new OutputStreamWriter(new FileOutputStream(prototxtFile))


    if (useV2) {
      netparam.getLayerList.asScala.foreach(layer => {
        val v2Layer = LayerParameter.newBuilder
        val blobSize = layer.getBlobsCount
        v2Layer.mergeFrom(layer)
        var i = 0
        while (i < blobSize) {
          v2Layer.removeBlobs(0)
          i += 1
        }
        netParameterWithoutData.addLayer(v2Layer)
      })
    } else {
      netparam.getLayersList.asScala.foreach(layer => {
        val v1Layer = V1LayerParameter.newBuilder
        val blobSize = layer.getBlobsCount
        v1Layer.mergeFrom(layer)
        var i = 0
        while (i < blobSize) {
          v1Layer.removeBlobs(0)
          i += 1
        }
        netParameterWithoutData.addLayers(v1Layer)
      })
    }
    prototxtWriter.write(netParameterWithoutData.build.toString)
    prototxtWriter.close
  }

}

object CaffePersister{
  def persist[T: ClassTag](prototxtPath: String,
               modelPath: String, module : Graph[T],
  useV2 : Boolean = true)(implicit ev: TensorNumeric[T]) : Unit = {
    val caffePersist = new CaffePersister[T](prototxtPath, modelPath, module, useV2)
    caffePersist.saveAsCaffe()
  }
}
