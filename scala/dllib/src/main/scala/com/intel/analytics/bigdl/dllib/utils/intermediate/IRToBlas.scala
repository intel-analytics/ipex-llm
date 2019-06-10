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
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.optim.DistriOptimizer._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Node, ReflectionUtils, T}

import scala.collection.mutable
import scala.reflect.ClassTag


abstract class ConvertBase[T, D] {
  /**
   * clone node relations
   * @param nodeMap node element maps from T to D
   */
  def cloneNode(allNodes: Array[Node[T]], nodeMap: mutable.HashMap[Node[T], Node[D]]): Unit = {
    allNodes.foreach(node => {
      node.nextNodesAndEdges.foreach(nextNodeAndEdge => {
        if (nodeMap.contains(nextNodeAndEdge._1)) {
          nodeMap.get(node).get.add(nodeMap.get(nextNodeAndEdge._1).get, nextNodeAndEdge._2)
        }
      })
    })
    // sort previous node
    nodeMap.toArray.foreach(node => {
      // if node has more than one previous nodes, we have to consider nodes order
      if (node._1.prevNodesAndEdges.length > 1) {
        node._2.removePrevEdges()
        node._1.prevNodesAndEdges.foreach(prevNodeAndEdge => {
          if (nodeMap.contains(prevNodeAndEdge._1)) {
            node._2.from(nodeMap.get(prevNodeAndEdge._1).get, prevNodeAndEdge._2)
          }
        })
      }
    })
  }

  def convertLayerCheck(layer: T) : Boolean

  def convertLayer(layer : T) : D

  def convertingCheck(allNodes: Array[Node[T]]) : Boolean = {
    var convert = true
    allNodes.foreach(node => {
      if (!convertLayerCheck(node.element)) {
        logger.info(s"${node.element} convertion failed")
        convert = false
      }
    })
    convert
  }

  def convert(allNodes: Array[Node[T]]): mutable.HashMap[Node[T], Node[D]] = {
    val nodeMap = new mutable.HashMap[Node[T], Node[D]]()
    allNodes.foreach(node => {
      nodeMap.put(node, new Node(convertLayer(node.element)))
    })
    cloneNode(allNodes, nodeMap)
    nodeMap
  }
}

private[bigdl] class IRToBlas[T: ClassTag] extends ConvertBase[IRElement[T], Module[T]]{

  private def className(layer: IRElement[T]): String = {
    val name = layer.getOp().name
    s"com.intel.analytics.bigdl.nn.${name.substring(2)}"
  }

  override def convertLayerCheck(layer: IRElement[T]): Boolean = {
    ReflectionUtils.findClass(className(layer)) != null ||
      layer.getOp().isInstanceOf[IRGeneralModule[T]]
  }

  override def convertLayer(layer : IRElement[T]) : Module[T] = {
    if (layer.getOp().isInstanceOf[IRGeneralModule[T]]) {
      return layer.getOp().asInstanceOf[IRGeneralModule[T]].model
    }
    ReflectionUtils.reflectFromIR(layer, Class.forName(className(layer)))
  }
}

private[bigdl] object IRToBlas {
  def apply[T: ClassTag](implicit ev: TensorNumeric[T]): IRToBlas[T] = new IRToBlas
}


