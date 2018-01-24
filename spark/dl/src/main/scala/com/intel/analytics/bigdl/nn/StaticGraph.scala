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
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.ops.Operation
import com.intel.analytics.bigdl.nn.tf.ControlDependency
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{DirectedGraph, Node, Util}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * A graph container. The modules in the container are connected as a DAG graph.
 *
 * @param _inputs inputs modules, user can feed data into these modules in the forward method
 * @param _outputs output modules
 * @param _variables
 * @param backGraphPruning whether enable backward graph pruning,
 *                         which means remove all pre-processing nodes
 *                         whose element is extended from [[Operation]] or
 *                         who is simply depended on [[Operation]] based nodes during backward.
 * @tparam T Numeric type. Only support float/double now
 */
class StaticGraph[T: ClassTag](
  private val _inputs : Seq[ModuleNode[T]],
  private val _outputs : Seq[ModuleNode[T]],
  private val _variables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None,
  private val excludeKeras: Boolean = true,
  private val backGraphPruning: Boolean = false
)(implicit ev: TensorNumeric[T]) extends Graph[T](_inputs, _outputs, _variables) {
  private val forwardExecution = forwardGraph.topologySort.reverse
  private var backwardExecution: Array[Node[AbstractModule[Activity, Activity, T]]] = _
  private val inputCache = new Array[Activity](forwardExecution.length)
  private var backId2ForwardId: Array[Int] = _
  private var gradOutputCache: Array[Activity] = _

  if (excludeKeras) {
    Util.excludeNotTorch(inputs.map(_.element))
    Util.excludeNotTorch(outputs.map(_.element))
  }

  buildBackwardGraph()

  override def updateOutput(input: Activity): Activity = {
    var i = 0
    while(i < forwardExecution.length) {
      val node = forwardExecution(i)
      val nodeInput = findInput(node, input)
      inputCache(i) = nodeInput
      node.element.forward(nodeInput)
      i += 1
    }

    output = dummyOutput.element.output
    output
  }

  override def backward(input: Activity, gradOutput: Activity): Activity = {
    val before = System.nanoTime()
    val gradients = backwardExecution(input, gradOutput, true)
    backwardTime = System.nanoTime() - before
    gradients
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    backwardExecution(input, gradOutput, false)
  }


  override def buildBackwardGraph(): this.type = {
    super.buildBackwardGraph()
    backwardExecution =
      if (!backGraphPruning) {
        backwardGraph.topologySort.reverse
      } else {
        val wholeResult = backwardGraph.topologySort.reverse
        val pruningModules = invTopology(forwardGraph).map(_.element).toSet
        wholeResult.filterNot(node => pruningModules.contains(node.element))
      }

    backId2ForwardId = new Array[Int](backwardExecution.length)
    gradOutputCache = new Array[Activity](backwardExecution.length)

    var i = 0
    while(i < backwardExecution.length - 1) {
      var j = 0
      var find = false
      while(j < forwardExecution.length) {
        if (forwardExecution(j).element.getName() == backwardExecution(i).element.getName()) {
          backId2ForwardId(i) = j
          find = true
        }
        j += 1
      }
      require(find, "Cannot find backward layer in forward executions")
      i += 1
    }

    this
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    var i = 0
    while (i < backwardExecution.length - 1) {
      val curNode = backwardExecution(i)
      val curInput = inputCache(backId2ForwardId(i))
      curNode.element.accGradParameters(curInput, gradOutputCache(i))
      i += 1
    }
  }

  override def populateModules(): Unit = {
    modules.appendAll(
      forwardGraph.topologySort
        // todo: convert control dep node to edge
        .filterNot(_.element.isInstanceOf[ControlDependency[T]])
        .filter(n => !n.eq(dummyOutput)).map(_.element)
        .reverse
    )
  }


  private def backwardExecution(input: Activity, gradOutput: Activity,
    executeBackward: Boolean): Activity = {
    dummyOutputGrad.element.gradInput = gradOutput

    var i = 0
    while (i < backwardExecution.length - 1) {  // do not execute the dummy backward end
      val curNode = backwardExecution(i)
      val curGradOutput = findGradOutput(curNode, gradOutput)
      gradOutputCache(i) = curGradOutput
      val curInput = inputCache(backId2ForwardId(i))
      if (!isStopGradient(curNode.element)) {
        if (executeBackward) {
          curNode.element.backward(curInput, curGradOutput)
        } else {
          curNode.element.updateGradInput(curInput, curGradOutput)
        }
      } else if (executeBackward) {
        curNode.element.accGradParameters(curInput, curGradOutput)
      }
      i += 1
    }

    gradInput = fetchModelGradInput()
    gradInput
  }

  /**
   * Inverse Topology specialize for pruning [[backwardGraph]],
   * which find out Node[Operation] or nodes only depended on Node[Operation] in backward.
   *
   * @return A sequence of graph nodes which can be pruned.
   */
  private def invTopology(graph: DirectedGraph[Module[T]]): Array[ModuleNode[T]] = {
    val (source, reverse) = graph.source -> graph.reverse
    // Build indegree list, LinkedHashMap can preserve the order of the keys, so it's good to
    // write unittest.
    val inDegrees = new mutable.LinkedHashMap[ModuleNode[T], Int]()
    inDegrees(source) = 0
    graph.DFS.foreach(n => {
      val nextNodes = if (!reverse) n.nextNodes else n.prevNodes
      nextNodes.foreach(m => {
        inDegrees(m) = inDegrees.getOrElse(m, 0) + 1
      })
    })

    val isOperation = (n: ModuleNode[T]) => n.element.isInstanceOf[Operation[_, _, T]]
    val canStop = new mutable.HashMap[ModuleNode[T], Boolean]()
    val noBackwardNodes = new ArrayBuffer[ModuleNode[T]]()
    while(inDegrees.nonEmpty) {
      // toArray is not lazy eval, which is not affected by inDegrees - 1 operations below
      val startNodes = inDegrees.filterKeys(inDegrees(_) == 0).keySet.toArray
      require(startNodes.length != 0, "There's a cycle in the graph")
      noBackwardNodes.appendAll(
        startNodes.filter(n => isOperation(n) || canStop.getOrElse(n, false))
      )
      startNodes.foreach(n => {
        val canCurrentStop = isOperation(n) || canStop.getOrElse(n, false)
        val nextNodes = if (!reverse) n.nextNodes else n.prevNodes
        nextNodes.foreach { nextNode =>
          // If only one parent of [[nextNode]] isn't `canStop`, it should NOT be `canStop`.
          if (!isOperation(nextNode)) {
            val status = canStop.getOrElseUpdate(nextNode, true)
            if (status && !canCurrentStop) {
              canStop(nextNode) = false
            }
          }
          inDegrees(nextNode) = inDegrees(nextNode) - 1
        }
        inDegrees.remove(n)
      })
    }
    noBackwardNodes.toArray
  }
}
