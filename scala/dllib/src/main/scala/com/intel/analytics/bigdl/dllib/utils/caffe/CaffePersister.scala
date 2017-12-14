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

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, OutputStream}

import scala.collection.JavaConverters._
import caffe.Caffe.{LayerParameter, NetParameter, V1LayerParameter}
import com.intel.analytics.bigdl.nn.{Container, Graph, Sequential, View}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{File, FileWriter, Node}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import caffe.Caffe
import com.google.protobuf.{CodedOutputStream, GeneratedMessage, TextFormat}
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.IOUtils
import org.apache.log4j.Logger
/**
 * A utility to convert BigDL model into caffe format and persist into local/hdfs file system
 *
 * @param prototxtPath  path to store model definition file
 * @param modelPath  path to store model weights and bias
 * @param module BigDL module to be converted
 * @param useV2  convert to V2 layer or not
 * @param overwrite whether to overwirte existing caffe files
 */
class CaffePersister[T: ClassTag](val prototxtPath: String,
      val modelPath: String, val module : AbstractModule[Activity, Activity, T],
      useV2 : Boolean = true, overwrite : Boolean = false)(implicit ev: TensorNumeric[T]) {

  private val logger = Logger.getLogger(getClass)

  private val hdfsPrefix: String = "hdfs:"

  private val v1Converter = new V1LayerConverter[T]()
  private val v2Converter = new LayerConverter[T]()

  private var netparam: NetParameter.Builder = NetParameter.newBuilder

  private def saveAsCaffe() : Unit = {
    convertToCaffe
    saveBinary
    savePrototext
  }

  // Convert module container to unified Graph, to be supported in module itselft in future
  private def toGraph() : Graph[T] = {
    if (module.isInstanceOf[Graph[T]]) {
      logger.info("model is graph")
      return module.asInstanceOf[Graph[T]]
    }
    // other containers/layers to be supported later
    throw new CaffeConversionException(s"container $module is not supported," +
      s"only graph supported")
  }
  // create caffe layers graph based on BigDL execution plan
  private def convertToCaffe() : Unit = {
    val graph = toGraph()
    val top2Layers = new mutable.HashMap[String, String]()
    val layers = new mutable.HashMap[String, GeneratedMessage]()
    val executions = graph.getSortedForwardExecutions
    netparam.setName(module.getName)
    executions.foreach(execution => {
      val preModules = execution.prevNodes
      val module = execution.element
      logger.info(s"find dependencies for ${module.getName}")
      if (module.isInstanceOf[View[T]]) {
        require(preModules.size == 1, "view pre-node size should be 1")
        val preNode = preModules(0)
        val nextNodes = execution.nextNodes
        val nextNodesName = nextNodes.map(_.element.getName())
        nextNodes.foreach(nextNode => {
          preNode -> nextNode
          execution.delete(nextNode)
        })
        preNode.delete(execution)

        // set top connection
        if (useV2) {
          var preNodeTopList = layers(preNode.element.getName).
            asInstanceOf[LayerParameter].getTopList.asScala
          preNodeTopList = preNodeTopList.filter(top => {top2Layers(top) == module.getName()})
          var topName = preNodeTopList(0)
          var i = 0
          while (i < nextNodesName.size) {
            top2Layers(s"$topName") = nextNodesName(i)
            i += 1
          }
        } else {
          var preNodeTopList = layers(preNode.element.getName).
            asInstanceOf[V1LayerParameter].getTopList.asScala
          preNodeTopList = preNodeTopList.filter(top => {top2Layers(top) == module.getName()})
          var topName = preNodeTopList(0)
          var i = 0
          while (i < nextNodesName.size) {
            top2Layers(s"topName_$i") = nextNodesName(i)
            i += 1
          }
        }
      } else {
        var bottomList = ArrayBuffer[String]()
        preModules.foreach(pre => {
          val name = pre.element.getName
          val preLayer = layers(name)
          val preTops = if (useV2) preLayer.asInstanceOf[LayerParameter].getTopList.asScala
            else preLayer.asInstanceOf[V1LayerParameter].getTopList.asScala
          preTops.foreach(top => bottomList.append(top))
        })
        bottomList = bottomList.filter(bottom => {
          val nextModule = top2Layers(bottom)
          nextModule.equals(execution.element.getName)
        })
        val nextModules = execution.nextNodes.filter(_.element != null)
        if (useV2) {
          val caffeLayers = v2Converter.toCaffe(execution.element, bottomList, nextModules.size)
          var curr : LayerParameter = null
          caffeLayers.foreach(layer => {
            val caffeLayer = layer.asInstanceOf[LayerParameter]
            curr = caffeLayer
            layers(caffeLayer.getName) = caffeLayer
            netparam.addLayer(caffeLayer)
          })
          // set last node's top list connecting to next nodes of current node
          val topList = curr.getTopList.asScala
          var i = 0
          while (i < nextModules.size) {
            top2Layers(topList(i)) = nextModules(i).element.getName()
            i += 1
          }

        } else {
          val caffeLayers = v1Converter.toCaffe(execution.element, bottomList, nextModules.size)
          var curr : V1LayerParameter = null
          caffeLayers.foreach(layer => {
            val caffeLayer = layer.asInstanceOf[V1LayerParameter]
            curr = caffeLayer
            layers(caffeLayer.getName) = caffeLayer
            netparam.addLayers(caffeLayer)
          })
          val topList = curr.getTopList.asScala
          var i = 0
          while (i < nextModules.size) {
            top2Layers(topList(i)) = nextModules(i).element.getName()
            i += 1
          }
        }
      }
    })
  }

  private def saveBinary() : Unit = {
    // save binary
    var binaryFileWriter: FileWriter = null
    try {
      binaryFileWriter = FileWriter(modelPath)
      val out = binaryFileWriter.create(overwrite)
      val byteArrayOut = new ByteArrayOutputStream()
      byteArrayOut.write(netparam.build.toByteArray)
      IOUtils.copyBytes(new ByteArrayInputStream(byteArrayOut.toByteArray), out, 1024, true)
    } finally {
      binaryFileWriter.close()
    }
  }

  private def savePrototext() : Unit = {
    // save prototxt
    val netParameterWithoutData = NetParameter.newBuilder
    netParameterWithoutData.setName(netparam.getName)
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

    var prototxtFileWriter: FileWriter = null
    try {
      prototxtFileWriter = FileWriter(prototxtPath)
      val out = prototxtFileWriter.create(overwrite)
      val byteArrayOut = new ByteArrayOutputStream()
      byteArrayOut.write(netParameterWithoutData.build().toString.getBytes)
      IOUtils.copyBytes(new ByteArrayInputStream(byteArrayOut.toByteArray), out, 1024, true)
    } finally {
      prototxtFileWriter.close()
    }
  }
}


object CaffePersister{
  def persist[T: ClassTag](prototxtPath: String,
               modelPath: String, module : AbstractModule[Activity, Activity, T],
  useV2 : Boolean = true, overwrite : Boolean = false)(implicit ev: TensorNumeric[T]) : Unit = {
    val caffePersist = new CaffePersister[T](prototxtPath, modelPath, module, useV2, overwrite)
    caffePersist.saveAsCaffe()
  }
}
