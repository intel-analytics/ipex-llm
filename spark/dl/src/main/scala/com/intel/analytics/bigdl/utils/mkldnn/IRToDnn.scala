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

package com.intel.analytics.bigdl.utils.mkldnn

import java.nio.ByteOrder
import java.util
import java.util.List

import com.google.protobuf.GeneratedMessage
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat, TensorModule}
import com.intel.analytics.bigdl.nn.{Module => _, _}
import com.intel.analytics.bigdl.nn.mkldnn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.tf.TensorflowToBigDL
import com.intel.analytics.bigdl.utils.{DirectedGraph, Node, T}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import spire.macros.Auto.scala

class IRToDnn extends ConvertBase[IRElement[Float], Module[Float]] {

  // converter function mappings
  private val IR2DnnMap = new mutable.HashMap[String, (IRElement[Float]) => Module[Float]]
  mapInit()

  private def mapInit(): Unit = {
    IR2DnnMap("IRSpatialConvolution") = fromConv
    IR2DnnMap("IRSpatialMaxPooling") = fromMaxPooling
    IR2DnnMap("IRSpatialAveragePooling") = fromAvgPooling
    IR2DnnMap("IRSpatialBatchNormalization") = fromSbn
    IR2DnnMap("IRLinear") = fromLinear
    IR2DnnMap("IRReLU") = fromReLU
  }

  override def enableConvertLayer(layer: IRElement[Float]): Boolean = {
    val name = layer.getOp().name
    if (IR2DnnMap.contains(name)) return true
    val className = "com.intel.analytics.bigdl.nn.mkldnn." + name.substring(2)
    val cls = ReflectUtils.classFound(className)
    if ( cls != null) true
    else false
  }

  override def convertLayer(layer: IRElement[Float]) : Module[Float] = {
    val name = layer.getOp().name
    if (IR2DnnMap.contains(name)) {
      val dnn = IR2DnnMap(name)(layer)
      if (layer.getName != "") dnn.setName(layer.name)
      dnn
    } else {
      val cls = Class.forName("com.intel.analytics.bigdl.nn.mkldnn." + name.substring(2))
      ReflectUtils.reflectFromIR(layer, cls)
    }
  }

  override def enableConvert(allNodes: Array[Node[IRElement[Float]]]) : Boolean = {
    var convert = true
    allNodes.foreach(node => {
      val op = node.element.getOp()
      if (op.isInstanceOf[IRReshape[Float]] && node.nextNodes.length == 1 &&
        node.nextNodes(0).element.getOp().isInstanceOf[IRLinear[Float]]) {
        // support pattern "reshape -> linear"
      } else if (op.isInstanceOf[IRView[Float]] && node.nextNodes.length == 1 &&
        node.nextNodes(0).element.getOp().isInstanceOf[IRLinear[Float]]) {
        // support pattern "view -> linear"
      } else if (!enableConvertLayer(node.element)) {
        convert = false
      }
    })
    // TODO : log false element name
    convert
  }

  override def convert(allNodes: Array[Node[IRElement[Float]]])
    : mutable.HashMap[Node[IRElement[Float]], Node[Module[Float]]] = {
    val oldToNew = new mutable.HashMap[Node[IRElement[Float]], Node[Module[Float]]]()
    allNodes.foreach(node => {
      val op = node.element.getOp()
      val dnn = if (op.isInstanceOf[IRReshape[Float]] && node.nextNodes.length == 1 &&
        node.nextNodes(0).element.getOp().isInstanceOf[IRLinear[Float]]) {
        new Node(mkldnn.Identity[Float]().asInstanceOf[Module[Float]])
      } else if (op.isInstanceOf[IRView[Float]] && node.nextNodes.length == 1 &&
        node.nextNodes(0).element.getOp().isInstanceOf[IRLinear[Float]]) {
        new Node(mkldnn.Identity[Float]().asInstanceOf[Module[Float]])
      } else {
        if (enableConvertLayer(node.element)) {
          new Node(convertLayer(node.element))
        } else {
          // todo: may be can support non dnn layers
          throw new UnsupportedOperationException(s"can not find ${node.element.getOp()} ")
        }
      }
      oldToNew.put(node, dnn)
    })
    cloneNode(oldToNew)
    oldToNew
  }

  private def fromReLU(node: IRElement[Float]) : Module[Float] = mkldnn.ReLU()

  private def fromConv(node: IRElement[Float]) : Module[Float] = {
    val t = node.getOp().asInstanceOf[IRSpatialConvolution[Float]]
    require(t.wRegularizer == null && t.bRegularizer == null,
      "Dnn SpatialConvolution can not support Regularizer")
    val cls = Class.forName("com.intel.analytics.bigdl.nn.mkldnn.SpatialConvolution")
    ReflectUtils.reflectFromIR(node, cls)
  }

  private def fromMaxPooling(node: IRElement[Float]) : Module[Float] = {
    val cls = Class.forName("com.intel.analytics.bigdl.nn.mkldnn.MaxPooling")
    ReflectUtils.reflectFromIR(node, cls)
  }

  private def fromAvgPooling(node: IRElement[Float]) : Module[Float] = {
    val cls = Class.forName("com.intel.analytics.bigdl.nn.mkldnn.AvgPooling")
    ReflectUtils.reflectFromIR(node, cls)
  }

  private def fromSbn(node: IRElement[Float]) : Module[Float] = {
    val t = node.getOp().asInstanceOf[IRSpatialBatchNormalization[Float]]
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

    layer
  }

  private def fromLinear(node: IRElement[Float]) : Module[Float] = {
    val t = node.getOp().asInstanceOf[IRLinear[Float]]
    require(t.wRegularizer == null && t.bRegularizer == null,
      "Dnn Linear can not support Regularizer")
    val cls = Class.forName("com.intel.analytics.bigdl.nn.mkldnn.Linear")
    ReflectUtils.reflectFromIR(node, cls)
  }
}

object IRToDnn {
  def apply[T: ClassTag](implicit ev: TensorNumeric[T]): IRToDnn = new IRToDnn
}
