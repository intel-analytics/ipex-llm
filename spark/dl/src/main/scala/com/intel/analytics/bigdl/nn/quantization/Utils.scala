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

package com.intel.analytics.bigdl.nn.quantization

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.{Container, Dummy, Graph}
import com.intel.analytics.bigdl.nn.{Linear => NNLinear}
import com.intel.analytics.bigdl.nn.{SpatialConvolution => NNConv}
import com.intel.analytics.bigdl.nn.{SpatialDilatedConvolution => NNDilatedConv}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Node
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object Utils {
  type ModuleNode[R] = AbstractModule[Activity, Tensor[R], R]
  type SeqNodes[R] = Seq[Node[ModuleNode[R]]]
  type ArrayNodes[R] = Array[Node[ModuleNode[R]]]
  type ANode[R] = Node[ModuleNode[R]]

  /**
   * replace the node in place in SeqNodes
   *
   * @param node old node
   * @param newNode new node
   * @param refs the refs, previous or next
   * @tparam T data type: Float or Double
   */
  def replaceRef[T: ClassTag](node: ANode[T], newNode: ANode[T], refs: SeqNodes[T]): Unit = {
    val buffer = refs.asInstanceOf[ArrayBuffer[Node[ModuleNode[T]]]]
    refs.zipWithIndex.filter(_._1 == node).foreach { x =>
      buffer.update(x._2, newNode)
    }
  }

  /**
   * replace the node in list to a new node and certainly the refs
   *
   * In a graph, a node has previous and next array buffers, which contain the refs of
   * previous and next nodes. It will replace the node at the index of list to a new node,
   * which contains a quantized layer.
   *
   * @param node old node
   * @param module quantized node
   * @param list node list
   * @param index index at list
   * @tparam T data type: Float or Double
   */
  def replaceNode[T: ClassTag](node: ANode[T], module: ModuleNode[T], list: ArrayNodes[T],
    index: Int): Unit = {
    // create a new node
    val newNode = Node(module)
    newNode.nextNodes.asInstanceOf[ArrayBuffer[Node[ModuleNode[T]]]] ++= node.nextNodes
    newNode.prevNodes.asInstanceOf[ArrayBuffer[Node[ModuleNode[T]]]] ++= node.prevNodes

    // prev.next
    newNode.prevNodes.foreach(n => replaceRef(node, newNode, n.nextNodes))

    // next.prev
    newNode.nextNodes.foreach(n => replaceRef(node, newNode, n.prevNodes))

    // update the list
    list.update(index, newNode)
  }

  /**
   * recursive call replace for graph
   *
   * It will create a new graph with mixing origin modules and quantized modules
   *
   * @param model graph model
   * @tparam T data type: Float or Double
   * @return new Graph
   */
  def convertGraph[T: ClassTag](model: Graph[T])(implicit ev: TensorNumeric[T]): Graph[T] = {
    val sortedNodes = model.backGraph.topologySort

    for (i <- sortedNodes.indices) {
      val node = sortedNodes(i)
      val module = node.element
      val waitedModule = replace[T](module)

      if (waitedModule != module) {
        replaceNode[T](node, waitedModule.asInstanceOf[ModuleNode[T]], sortedNodes, i)
      }
    }

    val inputs = sortedNodes.filter(n => n.prevNodes.isEmpty)
    // because all outputs point to dummy nodes, we should filter these nodes as outputs of Graph
    val outputs = sortedNodes.filter {n =>
      n.nextNodes.length == 1 && (n.nextNodes.head.element match {
        case d if d.isInstanceOf[Dummy[T]] => true
        case _ => false
      })
    }

    // delete all dummy nodes
    outputs.foreach { node =>
      node.nextNodes.asInstanceOf[ArrayBuffer[ANode[T]]].remove(0)
    }

    // create a new Graph, much simpler than replacing others in the old graph
    Graph[T](inputs, outputs)
  }

  /**
   * recursive call replace for container
   *
   * @param container container model
   * @tparam T data type: Float or Double
   * @return old container with quantized modules
   */
  def replaceContainer[T: ClassTag](container: Container[Activity, Activity, T])(
    implicit ev: TensorNumeric[T]): Module[T] = {
    // do with container
    for (i <- container.modules.indices) {
      container.modules(i) = replace(container.modules(i))
    }
    container
  }

  /**
   * replace the modules with quantized modules.
   *
   * Currently, it only supports three kinds of modules, SpatialConvolution,
   * SpatialDilatedConvolution and Linear.
   *
   * @param model un-quantized model
   * @tparam T data type: Float or Double
   * @return quantized model
   */
  def replace[T: ClassTag](model: Module[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    model match {
      case container: Container[Activity, Activity, T] =>
        container match {
          case graph: Graph[T] => convertGraph(graph)
          case _ => replaceContainer(container)
        }
      case normalConv if normalConv.isInstanceOf[NNConv[T]] =>
        val conv = normalConv.asInstanceOf[NNConv[T]]

        val quantizedConv = new SpatialConvolution[T](
          conv.nInputPlane, conv.nOutputPlane, conv.kernelW, conv.kernelH, conv.strideW,
          conv.strideH, conv.padW, conv.padH, conv.nGroup)

        quantizedConv.initWeightAndBias(conv.weight, conv.bias)
      case dilatedConv if dilatedConv.isInstanceOf[NNDilatedConv[T]] =>
        val conv = dilatedConv.asInstanceOf[NNDilatedConv[T]]

        val quantizedConv = new SpatialDilatedConvolution[T](
          conv.nInputPlane, conv.nOutputPlane, conv.kW, conv.kH, conv.dW, conv.dH, conv.padW,
          conv.padH, conv.dilationW, conv.dilationH)

        quantizedConv.initWeightAndBias(conv.weight, conv.bias)
      case normalLinear if normalLinear.isInstanceOf[NNLinear[T]] =>
        val linear = normalLinear.asInstanceOf[NNLinear[T]]

        val quantizedLinear = new Linear[T](linear.weight.size(2), linear.weight.size(1))

        quantizedLinear.initWeightAndBias(linear.weight, linear.bias)
      case _ => model
    }
  }
  /**
   * delete parameters of SpatialConvolution, SpatialDilatedConvolution) and linear.
   *
   * because it will make all parameters into a long array in a BigDL model by default,
   * so the origin parameters will exist in the quantized model. We have to delete them
   * for reducing the size.
   *
   * After deleting all these matched parameters, it will make a **new** long array of
   * other layers parameters.
   *
   * @param parameters parameters of all layers
   * @tparam T data type Float or Double
   * @return parameters reorganized
   */
  def reorganizeParameters[T: ClassTag](parameters: Array[Tensor[T]])(
    implicit ev: TensorNumeric[T]): Tensor[T] = {
    var length = 0
    for (i <- parameters.indices) {
      // we recognize quantized layers by the parameters, which will return 2 nulls
      // in quantized layers.
      if (parameters(i) != null) {
        length += parameters(i).nElement()
      }
    }

    val result = Tensor[T](length)

    var offset = 0
    for (i <- parameters.indices) {
      val parameter = parameters(i)

      if (parameter != null) {
        val length = parameter.nElement()

        val (src, srcOffset) = (parameter.storage().array(), parameter.storageOffset() - 1)
        val (dst, dstOffset) = (result.storage().array(), offset)

        val (size, stride) = (parameter.size(), parameter.stride())

        System.arraycopy(src, srcOffset, dst, dstOffset, length)
        parameter.set(result.storage(), offset + 1, size, stride)

        offset += length
      }
    }

    result
  }
}
