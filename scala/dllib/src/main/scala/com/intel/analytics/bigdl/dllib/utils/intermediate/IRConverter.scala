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

import com.intel.analytics.bigdl.nn.Graph
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.mkldnn._
import com.intel.analytics.bigdl.tensor.{FloatType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.{Module, utils}
import com.intel.analytics.bigdl.utils.{Engine, MklBlas, MklDnn, Node}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag


private[bigdl] class IRConverter[T: ClassTag](IRgraph: IRGraph[T])(implicit ev: TensorNumeric[T]) {
  private val allNodes = new ArrayBuffer[Node[IRElement[T]]]
  private val irInputs = IRgraph.inputs.toArray
  private val irOutputs = IRgraph.outputs.toArray

  init()
  private def init() : Unit = {
    getNodes(irInputs, allNodes)
    // reminder: some output nodes may not be searched from inputs
    irOutputs.foreach(node => {
      if (!allNodes.contains(node)) allNodes.append(node)
    })
  }


  private def getNodes(inputs: Seq[Node[IRElement[T]]],
                       nodesBuffer: ArrayBuffer[Node[IRElement[T]]]): Unit = {
    if (inputs.length == 0) return
    inputs.foreach(node => {
      if (!nodesBuffer.contains(node)) {
        nodesBuffer.append(node)
        getNodes(node.nextNodes, nodesBuffer)
      }
    })
  }

  /**
   * convert IRgraph to blas or dnn graph according to engine type
   * @return dnn graph or blas graph converted from ir graph
   */
  def toGraph() : Graph[T] = {
    if (utils.Engine.getEngineType() == MklBlas) {
      require(IRToBlas[T].convertingCheck(allNodes.toArray),
        "IR graph can not be converted to Blas layer")
      toBlasGraph()
    } else if (utils.Engine.getEngineType() == MklDnn) {
      require(ev.getType() == FloatType, "Mkldnn engine only supports float data")
      require(IRToDnn[Float].convertingCheck(
        allNodes.toArray.asInstanceOf[Array[Node[IRElement[Float]]]]),
        "IR graph can not be converted to Dnn layer")
      toDnnGraph()
    } else throw new UnsupportedOperationException(
      s"Only support engineType mkldnn/mklblas, but get ${Engine.getEngineType()}")
  }

  private def toDnnGraph(): Graph[T] = {
    val nodeMap = IRToDnn[Float].convert(
      allNodes.toArray.asInstanceOf[Array[Node[IRElement[Float]]]])
    val inputs = irInputs.map(
      n => nodeMap.get(n.asInstanceOf[Node[IRElement[Float]]]).get)
    val outputs = irOutputs.map(
      n => nodeMap.get(n.asInstanceOf[Node[IRElement[Float]]]).get)

    // add input node for dnn graph
    val realInputs = inputs.map(n => {
      val node = new Node[Module[Float]](new InputWrapper())
      n.from(node)
      node
    })

    // add output node for graph
    val realOutputs = outputs.zipWithIndex.map {
      case (model: Node[Module[Float]], index: Int) =>
        val node = if (model.element.isInstanceOf[BlasWrapper]) {
          model
        } else {
          model.add(new Node[Module[Float]](Output(IRgraph.outputFormats(index))))
        }
        node
    }

    DnnGraph(realInputs, realOutputs,
      IRgraph.variables.asInstanceOf[Option[(Array[Tensor[Float]], Array[Tensor[Float]])]],
      IRgraph.generateBackward).asInstanceOf[Graph[T]]
  }

  private def toBlasGraph(): Graph[T] = {
    val nodeMap = IRToBlas[T].convert(allNodes.toArray)
    val inputs = irInputs.map(n => nodeMap.get(n).get)
    val outputs = irOutputs.map(n => nodeMap.get(n).get)

    Graph.dynamic(inputs, outputs, IRgraph.variables, IRgraph.generateBackward)
  }
}
