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
import com.intel.analytics.bigdl.nn.{Container, Graph, Sequential, View}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Node

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import caffe.Caffe
import com.google.protobuf.{GeneratedMessage, TextFormat}
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
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
      val modelPath: String, val module : Container[Activity, Activity, T],
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

  // Convert module container to unified Graph
  private def toGraph() : Graph[T] = {
    if (module.isInstanceOf[Graph[T]]) {
      logger.info("model is graph")
      return module.asInstanceOf[Graph[T]]
    } else if (module.isInstanceOf[Sequential[T]]) {
      logger.info("model is sequential")
      val modules = module.asInstanceOf[Sequential[T]].modules
      val allNodes = new ArrayBuffer[ModuleNode[T]]()
      val dummyNode = new ModuleNode[T](null)
      allNodes.append(dummyNode)
      var curr = dummyNode
      modules.foreach(node => {
        val nnode = new ModuleNode[T](node.asInstanceOf[AbstractModule[Activity, Tensor[T], T]])
        allNodes.append(nnode)
        curr -> nnode
        curr = nnode
      })
      val input = allNodes.filter(_.element != null).filter(_.prevNodes.size == 0).toArray
      val output = allNodes.filter(_.element != null).filter(_.prevNodes.size == 0).toArray
      return Graph(input, output)
    }
    throw  new UnsupportedOperationException(s"container $module is not supported!")
  }
  // create caffe layers graph based on BigDL execution plan
  private def convertToCaffe() : Unit = {
    val graph = toGraph()
    val top2Layers = new mutable.HashMap[String, String]()
    val layers = new mutable.HashMap[String, GeneratedMessage]()
    val executions = graph.getExecutions
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
               modelPath: String, module : Container[Activity, Activity, T],
  useV2 : Boolean = true)(implicit ev: TensorNumeric[T]) : Unit = {
    val caffePersist = new CaffePersister[T](prototxtPath, modelPath, module, useV2)
    caffePersist.saveAsCaffe()
  }
}
