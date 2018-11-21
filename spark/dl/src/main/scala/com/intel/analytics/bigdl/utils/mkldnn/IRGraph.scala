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

import java.util.List

import breeze.linalg.reverse
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn.{Graph, keras}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat}
import com.intel.analytics.bigdl.nn.mkldnn.{DnnGraph, HeapData, MklDnnRuntime, Phase}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, MklBlas, Node, T}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * Generate IR graph
 * @param inputs input nodes for graph
 * @param outputs output nodes for graph
 * @param variables
 * @param generateBackward
 * @param inputFormats input memory layout for graph
 * @param outputFormats output memory layout for graph
 * @param ev$1
 * @param ev
 * @tparam T The numeric type in this module parameters.
 */
class IRGraph[T: ClassTag](
    val inputs : Seq[Node[IRElement[T]]],
    val outputs : Seq[Node[IRElement[T]]],
    val variables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None,
    val generateBackward: Boolean = true,
    val inputFormats: Int = Memory.Format.nchw,
    val outputFormats: Int = Memory.Format.nc)
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Activity, Activity, T] with Serializable {

  @transient private var initFwd: Boolean = false
  @transient private var initBwd: Boolean = false
  @transient private var initAcc: Boolean = false

  var allNodes = new ArrayBuffer[Node[IRElement[T]]]
  private var graph: Graph[T] = null

  // init should find all nodes
  init()

  private def init() : Unit = {
    getNodes(inputs, allNodes)
    // remind: some output nodes may not be searched from inputs
    outputs.foreach(node => {
      if (!allNodes.contains(node)) allNodes.append(node)
    })
  }

  private def getNodes(inputs: Seq[Node[IRElement[T]]],
                       nodesBuffer: ArrayBuffer[Node[IRElement[T]]]): Unit = {
    if (inputs.length == 0) return
    inputs.foreach(node => {
      if (!nodesBuffer.contains(node)) nodesBuffer.append(node)
      getNodes(node.nextNodes, nodesBuffer)
    })
  }

  override def updateOutput(input: Activity): Activity = {
    if (graph == null) {
      throw new UnsupportedOperationException("forward not supported, Please build graph first")
    }
    if (!initFwd && graph.isInstanceOf[DnnGraph]) {
      graph.asInstanceOf[DnnGraph].setRuntime(new MklDnnRuntime())
      graph.asInstanceOf[DnnGraph].initFwdPrimitives(
        Array(HeapData(input.toTensor[T].size(), inputFormats)), Phase.TrainingPhase)
      initFwd = true
    }
    output = graph.updateOutput(input)
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    if (graph == null) {
      throw new UnsupportedOperationException("backward not supported, Please build graph first")
    }
    if (!initBwd && graph.isInstanceOf[DnnGraph]) {
      graph.asInstanceOf[DnnGraph].initBwdPrimitives(
        graph.asInstanceOf[DnnGraph].outputFormats(), Phase.TrainingPhase)
      initBwd = true
    }
    gradInput = graph.updateGradInput(input, gradOutput)
    gradInput
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    if (graph == null) {
      throw new UnsupportedOperationException("backward not supported, Please build graph first")
    }
    if (!initAcc && graph.isInstanceOf[DnnGraph]) {
      graph.asInstanceOf[DnnGraph].initGradWPrimitives(
        graph.asInstanceOf[DnnGraph].outputFormats(), Phase.TrainingPhase)
      initAcc = true
    }
    graph.accGradParameters(input, gradOutput)
  }

  def build(): Unit = {
    graph = new IRConverter[T](this).toGraph()
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    graph.parameters()
  }
}

object IRGraph {
  def apply[T: ClassTag](
    inputs: Seq[Node[IRElement[T]]],
    outputs: Seq[Node[IRElement[T]]],
    variables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None,
    generateBackward: Boolean = true,
    inputFormats: Int = Memory.Format.nchw,
    outputFormats: Int = Memory.Format.nc
  )( implicit ev: TensorNumeric[T]): IRGraph[T] = {
    new IRGraph[T](inputs, outputs, variables, generateBackward, inputFormats, outputFormats)
  }
}