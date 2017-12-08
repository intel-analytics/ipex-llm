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

import java.util

import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.tf.WithoutInput
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{DirectedGraph, Node, T, Table}

import scala.reflect.ClassTag

/**
 * A graph container. The modules in the container are connected as a DAG graph.
 *
 * @param inputs inputs modules, user can feed data into these modules in the forward method
 * @param outputs output modules
 * @param variables
 * @param ev$1
 * @param ev
 * @tparam T Numeric type. Only support float/double now
 */
class StaticGraph[T: ClassTag](
  val inputs : Seq[ModuleNode[T]],
  private val outputs : Seq[ModuleNode[T]],
  private val variables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None)
  (implicit ev: TensorNumeric[T])
  extends Container[Activity, Activity, T]{

  type absModule = AbstractModule[_ <: Activity, _ <: Activity, T]

  override def updateOutput(input: Activity): Activity = {
    var i = 0
    while(i < forwardExecutions.length) {
      val node = forwardExecutions(i)
      val nodeInput = if (node.prevNodes.isEmpty && !node.element.isInstanceOf[WithoutInput]) {
        inputData(node, input)
      } else {
        val prevActivities = node.prevNodesAndEdges.map(n => {
          n._2.fromIndex match {
            case Some(i) => n._1.element.output.toTable.apply[Activity](i)
            case None => n._1.element.output
          }
        })
        if (prevActivities.length == 1) {
          prevActivities.head
        } else {
          seqToTable(prevActivities)
        }
      }
      node.element.forward(nodeInput)
      inputsBP.put(node.element.getName(), nodeInput)
      i += 1
    }

    output = dummyOutput.element.output
    output
  }

  override def backward(input: Activity, gradOutput: Activity): Activity = {
    val before = System.nanoTime()
    dummyOutputGrad.element.gradInput = gradOutput

    var i = 0
    while (i < backwardExecutions.length) {
      val curNode = backwardExecutions(i)
      var curGradOutput : Activity = null

      curNode.nextNodesAndEdges.foreach(n => {
        val otherActivity = if (n._1.element.gradInput.isTensor || n._1.prevEdges.length == 1) {
          n._1.element.gradInput
        } else {
          val index = n._1.prevEdges.indexOf(n._2) + 1
          n._1.element.gradInput.toTable.apply[Activity](index)
        }

        n._2.fromIndex match {
          case Some(i) =>
            if (curNode.element.output.isTable && curGradOutput == null) {
              curGradOutput = T()
            }
            val curActivity = curGradOutput.toTable.getOrElse[Activity](i, null)
            curGradOutput.toTable(i) = accActivity(curActivity, otherActivity)
          case None =>
            curGradOutput = accActivity(curGradOutput, otherActivity)
        }
      })

      gradOutputBP(i) = curGradOutput
      if (!isStopGradient(curNode.element)) {
        curNode.element.backward(inputsBP.get(curNode.element.getName()), curGradOutput)
      } else {
        curNode.element.accGradParameters(inputsBP.get(curNode.element.getName()), curGradOutput)
      }
      i += 1
    }

    gradInput = if (inputs.length == 1) {
      inputs.head.element.gradInput
    } else {
      seqToTable(inputs.map(n => n.element.gradInput))
    }
    backwardTime = System.nanoTime() - before
    gradInput
  }

  private def calcSumTimesOfAllNodes(timesOfAllNodes: Array[(absModule, Long, Long)])
  : (Long, Long) = {
    var sumForward = 0L
    var sumBackward = 0L
    timesOfAllNodes.foreach(x => {
      sumForward += x._2
      sumBackward += x._3
    })
    (sumForward, sumBackward)
  }

  override def getTimes():
  Array[(AbstractModule[_ <: Activity, _ <: Activity, T], Long, Long)] = {
    val timesOfAllNodes = this.modules.flatMap(_.getTimes()).toArray
    val (sumForward, sumBackward) = calcSumTimesOfAllNodes(timesOfAllNodes)
    timesOfAllNodes ++ Array((this, this.forwardTime - sumForward, this.backwardTime - sumBackward))
  }

  override def resetTimes(): Unit = {
    this.forwardTime = 0L
    this.backwardTime = 0L
    modules.foreach(_.resetTimes())
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    dummyOutputGrad.element.gradInput = gradOutput

    var i = 0
    while (i < backwardExecutions.length) {
      val curNode = backwardExecutions(i)
      var curGradOutput : Activity = null
      if (curNode.element.output.isTable) {
        curGradOutput = T()
      }

      curNode.nextNodesAndEdges.foreach(n => {
        val otherActivity = if (n._1.element.gradInput.isTensor || n._1.prevEdges.length == 1) {
          n._1.element.gradInput
        } else {
          val index = n._1.prevEdges.indexOf(n._2) + 1
          n._1.element.gradInput.toTable.apply[Activity](index)
        }

        n._2.fromIndex match {
          case Some(i) =>
            val curActivity = curGradOutput.toTable.getOrElse[Activity](i, null)
            curGradOutput.toTable(i) = accActivity(curActivity, otherActivity)
          case None =>
            curGradOutput = accActivity(curGradOutput, otherActivity)
        }
      })

      gradOutputBP(i) = curGradOutput

      if (!isStopGradient(curNode.element)) {
        curNode.element.updateGradInput(inputsBP.get(curNode.element.getName()), curGradOutput)
      }
      i += 1
    }

    gradInput = if (inputs.length == 1) {
      inputs.head.element.gradInput
    } else {
      seqToTable(inputs.map(n => n.element.gradInput))
    }
    gradInput
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    var i = 0
    while (i < backwardExecutions.length) {
      val curNode = backwardExecutions(i)
      curNode.element.accGradParameters(inputsBP.get(curNode.element.getName()), gradOutputBP(i))
      i += 1
    }
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    variables match {
      case None => super.parameters()
      case Some((weights, gradients)) => (weights, gradients)
    }
  }

  override def add(module: AbstractModule[_ <: Activity, _ <: Activity, T]): this.type = {
    throw new IllegalArgumentException("Graph: Please don't use add method in Graph container. " +
      "A graph container should not be changed after it is constructed")
  }

  // Add a dummy output node, to get an one end graph. So the nodes that are not dependent by
  // the outputs will be excluded
  private val dummyOutput = new ModuleNode[T](new Identity[T]())
  // Add a dummy output node for backward graph,
  // dummyOutputGrad has the same function as dummyOutput
  // used to construct a backward graph
  private var dummyOutputGrad: ModuleNode[T] = _
  outputs.foreach(_ -> dummyOutput)

  /**
   * Computing backgraph
   */
  private val backGraph = dummyOutput.graph(reverse = true)
  private var gradGraph: DirectedGraph[AbstractModule[Activity, Activity, T]] = null

  /**
   * Execution plan
   */
  private val forwardExecutions = backGraph.topologySort.reverse
  private var backwardExecutions: Array[Node[AbstractModule[Activity, Activity, T]]] = null

  modules.appendAll(forwardExecutions.filter(n => !n.eq(dummyOutput)).map(_.element))

  /**
   * build is needed when the stopGrad is changed
   */
  private[bigdl] def build(): this.type = {
    val gradGraph = backGraph.cloneGraph()
    dummyOutputGrad = gradGraph.source
    val nodes = gradGraph.DFS
    nodes.filter(x => isStopGradient(x.element)).foreach(_.removePrevEdges())
    backwardExecutions = gradGraph.topologySort.filter(n => !n.eq(dummyOutputGrad))
    clearState()
    this
  }


  private val inputsBP = new util.HashMap[String, Activity]()

  // Check all inputs of the graph should be passed in
  checkRoots
  build

  private val gradOutputBP = new Array[Activity](forwardExecutions.length - 1)

  private def checkRoots: Unit = {
    val roots = forwardExecutions.filter(_.prevNodes.size == 0)
      .filter(node => !node.element.isInstanceOf[WithoutInput])
    require(roots.size == inputs.length,
      s"There're ${inputs.length} inputs, but graph has ${roots.size} roots")
    inputs.foreach(n =>
      require(roots.contains(n), "inputs and graph roots are not match")
    )
  }

  private[nn] def shift[B](data : Array[B], from : Int, to : Int): Array[B] = {
    require(from < data.length && from >= 0, s"invalid from $from array length is ${data.length}")
    require(to < data.length && to >= 0, s"invalid to $to array length is ${data.length}")
    if (from == to) {
      data
    } else if (from < to) {
      var i = from
      while(i < to) {
        val tmp = data(i)
        data(i) = data(i + 1)
        data(i + 1) = tmp
        i += 1
      }
      data
    } else {
      var i = from
      while(i > to) {
        val tmp = data(i)
        data(i) = data(i - 1)
        data(i - 1) = tmp
        i -= 1
      }
      data
    }
  }

  private def seqToTable(inputs: Seq[Activity]) : Table = {
    val t = T()
    var j = 1
    inputs.foreach(tensor => {
      t(j) = tensor
      j += 1
    })
    t
  }

  private def inputData(
    node: Node[AbstractModule[Activity, Activity, T]],
    input: Activity
  ): Activity = {
    if (inputs.length == 1) {
      require(inputs(0).eq(node), "input node is not in the input list")
      input.toTensor
    } else {
      val i = inputs.indexOf(node)
      require(i != -1, "input node is not in the input list")
      input.toTable[Tensor[T]](i + 1)
    }
  }

  /**
   * set an array of layers that match the given ```names``` to be "freezed",
   * i.e. their parameters(weight/bias, if exists) are not changed in training process
   * @param names an array of layer names
   * @return current graph model
   */
  def freeze(names: Array[String]): this.type = {
    names.foreach(name => {
      val layer = this (name)
      require(layer.isDefined, s"cannot find layer match ${name}")
      layer.get.setScaleW(0)
      layer.get.setScaleB(0)
    })
    this
  }

  private var stopGradientLayers: util.HashSet[String] = _

  /**
   * whether stop propagating gradInput back
   * @return
   */
  private def isStopGradient(module: AbstractModule[_ <: Activity, _ <: Activity, T]): Boolean = {
    null != stopGradientLayers && stopGradientLayers.contains(module.getName())
  }

  /**
   * stop the input gradient of layers that match the given ```names```
   * their input gradient are not computed.
   * And they will not contributed to the input gradient computation of
   * layers that depend on them.
   * @param names an array of layer names
   * @return current graph model
   */
  def stopGradient(names: Array[String]): this.type = {
    names.foreach(name => {
      val layer = this (name)
      require(layer.isDefined, s"cannot find layer match ${name}")
      if (stopGradientLayers == null) stopGradientLayers =
        new util.HashSet[String]()
      stopGradientLayers.add(layer.get.getName())
    })
    build()
    this
  }


  override def reset(): Unit = {
    if (null != stopGradientLayers) stopGradientLayers.clear()
    unFreeze()
    build()
  }

  /**
   * get forward executions, the dummy node will be filtered
   * @return
   */
  def getForwardExecutions: Array[Node[AbstractModule[Activity, Activity, T]]] = {
    forwardExecutions.filter(n => !n.eq(dummyOutput))
  }

  @inline
  private def accActivity(activity: Activity, other: Activity): Activity = {
    if (activity == null) {
      other
    } else {
      if (other.isTensor) {
        require(activity.isTensor, "Cannot add a table to a tensor")
        activity.toTensor[T].add(other.toTensor[T])
      } else {
        val actTable = activity.toTable
        val actLen = actTable.length()
        val otherTable = other.toTable
        val otherLen = otherTable.length()
        require(actLen == otherLen, "table length is not equal")
        var i = 1
        while(i <= actLen) {
          require(actTable[Activity](i) != null, "Invalid table element")
          accActivity(actTable[Activity](i), otherTable[Activity](i))
          i += 1
        }
        actTable
      }
    }
  }
}

private[bigdl] class Dummy[T: ClassTag]()(implicit ev: TensorNumeric[T])
  extends AbstractModule[Activity, Tensor[T], T] {
  override def updateOutput(input: Activity): Tensor[T] = null
  override def updateGradInput(input: Activity, gradOutput: Tensor[T]): Activity = null
}
