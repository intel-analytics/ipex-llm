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

import scala.collection.JavaConverters._
import caffe.Caffe.{LayerParameter, V1LayerParameter}
import com.intel.analytics.bigdl.nn.Graph
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Node

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import caffe.Caffe
import com.google.protobuf.GeneratedMessage
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

  private var netparam: Caffe.NetParameter = _

  def saveAsCaffe() : Unit = {
    convertToCaffe()
  }
  private def createNet() : Unit = {
    netparam.toString
  }
  // create caffe layers graph based on BigDL execution plan
  private def convertToCaffe() : Unit = {
    val top2Layers = new mutable.HashMap[String, String]()
    val layers = new mutable.HashMap[String, GeneratedMessage]()
    val executions = module.getExecutions
    executions.foreach(execution => {
      val preModules = execution.prevNodes
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
      val nextModules = execution.nextNodes
      if (useV2) {
        val caffeLayer = v2Converter.toCaffe(execution, bottomList).asInstanceOf[LayerParameter]
        val caffeNode = new Node(caffeLayer)
        val topList = caffeLayer.getTopList.asScala
        val i = 0
        while (i < nextModules.size) {
          top2Layers(topList(i)) = nextModules(i).element.getName()
        }
        v2Layers.append(caffeNode)
        layers(caffeLayer.getName) = caffeLayer
      } else {
        val caffeLayer = v1Converter.toCaffe(execution, bottomList).asInstanceOf[V1LayerParameter]
        val caffeNode = new Node(caffeLayer)
        val topList = caffeLayer.getTopList.asScala
        val i = 0
        while (i < nextModules.size) {
          top2Layers(topList(i)) = nextModules(i).element.getName()
        }
        v1Layers.append(caffeNode)
        layers(caffeLayer.getName) = caffeLayer
      }
    })
  }

  private def save() : Unit = {

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
