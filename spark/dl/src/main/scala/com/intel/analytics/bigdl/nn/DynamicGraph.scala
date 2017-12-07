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
import com.intel.analytics.bigdl.nn.ops.ControlOps
import com.intel.analytics.bigdl.nn.tf.{ControlDependency, WithoutInput}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{DirectedGraph, Node, T, Table}
import com.intel.analytics.bigdl.utils.tf.Tensorflow
import org.tensorflow.framework.GraphDef

import scala.collection.mutable
import scala.reflect.ClassTag

class DynamicGraph[T: ClassTag](val inputs : Seq[ModuleNode[T]],
  private val outputs : Seq[ModuleNode[T]],
  private val variables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None,
  private val generateBackward: Boolean = true
)(implicit ev: TensorNumeric[T])
  extends Container[Activity, Activity, T]{

  type absModule = AbstractModule[_ <: Activity, _ <: Activity, T]

  override def updateOutput(input: Activity): Activity = {
    forwardScheduler.reset()
    while (!forwardScheduler.isFinished()) {
      val node = forwardScheduler.fetch()
      val nodeInput = if (node.prevNodes.isEmpty && !node.element.isInstanceOf[WithoutInput]) {
        inputData(node, input)
      } else {
        val prevActivities = node.prevNodesAndEdges
          .filterNot(n => n._1.element.isInstanceOf[ControlDependency[T]])
          .map(n => {
            n._2.fromIndex match {
              case Some(i) =>
                if (n._1.element.output == null || (i == 1 && n._1.element.output.isTensor)) {
                  n._1.element.output
                } else {
                  n._1.element.output.toTable.apply[Activity](i)
                }
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
      inputCache(node.element.getName()) = nodeInput
      forwardScheduler.schedule(node)
    }

    output = dummyOutput.element.output
    output
  }

  override def backward(input: Activity, gradOutput: Activity): Activity = {
    if (!generateBackward) return null

    val before = System.nanoTime()
    backwardScheduler.reset()
    while (!backwardScheduler.isFinished()) {
      val curNode = backwardScheduler.fetch()
      var curGradOutput : Activity = if (curNode.eq(dummyOutputGrad)) gradOutput else null

      curNode.prevNodesAndEdges.filterNot(n => n._1.element.isInstanceOf[ControlDependency[T]])
        .foreach(n => {
          val otherActivity = if (n._1.element.gradInput.isTensor || n._1.nextEdges.length == 1) {
            n._1.element.gradInput
          } else {
            val index = n._1.nextEdges.indexOf(n._2) + 1
            n._1.element.gradInput.toTable.apply[Activity](index)
          }

          n._2.fromIndex match {
            case Some(i) =>
              if (i == 1 && curNode.element.output.isTensor) {
                curGradOutput = accActivity(curGradOutput, otherActivity)
              } else {
                if (curNode.element.output.isTable && curGradOutput == null) {
                  curGradOutput = T()
                }
                val curActivity = curGradOutput.toTable.getOrElse[Activity](i, null)
                curGradOutput.toTable(i) = accActivity(curActivity, otherActivity)
              }
            case None =>
              curGradOutput = accActivity(curGradOutput, otherActivity)
          }
        })

      if (curNode.element.output.isTable) {
        addZeroTensorToMissingGradOutput(curNode.element.output.toTable, curGradOutput.toTable)
      }

      gradOutputCache(curNode.element.getName()) = curGradOutput
      if (!isStopGradient(curNode.element)) {
        curNode.element.backward(inputCache(curNode.element.getName()), curGradOutput)
      } else {
        curNode.element.accGradParameters(inputCache(curNode.element.getName()), curGradOutput)
      }
      backwardScheduler.schedule(curNode)
    }

    gradInput = if (inputs.length == 1) {
      inputs.head.element.gradInput
    } else {
      seqToTable(inputs.map(n => n.element.gradInput))
    }
    backwardTime = System.nanoTime() - before
    gradInput
  }

  private def addZeroTensorToMissingGradOutput(output: Table, gradOutput: Table): Unit = {
    var i = 0
    while (i < output.length()) {
      if (!gradOutput.contains(i + 1)) {
        val tensor = output[Tensor[T]](i + 1)
        val zero = Tensor(tensor.size())
        gradOutput(i + 1) = zero
      }
      i = i + 1
    }
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
    if (!generateBackward) return null

    backwardScheduler.reset()
    while (!backwardScheduler.isFinished()) {
      val curNode = backwardScheduler.fetch()
      var curGradOutput : Activity = if (curNode.eq(dummyOutputGrad)) gradOutput else null

      curNode.prevNodesAndEdges.filterNot(n => n._1.element.isInstanceOf[ControlDependency[T]])
        .foreach(n => {
          val otherActivity = if (n._1.element.gradInput.isTensor || n._1.nextEdges.length == 1) {
            n._1.element.gradInput
          } else {
            val index = n._1.nextEdges.indexOf(n._2) + 1
            n._1.element.gradInput.toTable.apply[Activity](index)
          }

          n._2.fromIndex match {
            case Some(i) =>
              if (i == 1 && curNode.element.output.isTensor) {
                curGradOutput = accActivity(curGradOutput, otherActivity)
              } else {
                if (curNode.element.output.isTable && curGradOutput == null) {
                  curGradOutput = T()
                }
                val curActivity = curGradOutput.toTable.getOrElse[Activity](i, null)
                curGradOutput.toTable(i) = accActivity(curActivity, otherActivity)
              }
            case None =>
              curGradOutput = accActivity(curGradOutput, otherActivity)
          }
        })

      if (curNode.element.output.isTable) {
        addZeroTensorToMissingGradOutput(curNode.element.output.toTable, curGradOutput.toTable)
      }

      gradOutputCache(curNode.element.getName()) = curGradOutput
      if (!isStopGradient(curNode.element)) {
        curNode.element.updateGradInput(inputCache(curNode.element.getName()), curGradOutput)
      }
      backwardScheduler.schedule(curNode)
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
    while (i < backwardNodes.length) {
      val curNode = backwardNodes(i)
      curNode.element.accGradParameters(inputCache(curNode.element.getName()),
        gradOutputCache(curNode.element.getName()))
      i += 1
    }
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    variables match {
      case None => super.parameters()
      case Some((weights, gradients)) => (weights, gradients)
    }
  }

  override def add(module: AbstractModule[_ <: Activity, _ <: Activity, T]): Graph.this.type = {
    throw new IllegalArgumentException("Graph: Please don't use add method in Graph container. " +
      "A graph container should not be changed after it is constructed")
  }

  // todo: expand the graph
  override def toGraph(startNodes: ModuleNode[T]*): Graph[T] = this

  /**
   * Return the corresponding node has the given name. If the given name doesn't match any node,
   * NoSuchElementException will be thrown
   * @param name
   * @return
   */
  def node(name: String): ModuleNode[T] = {
    val matchNodes = backGraph.BFS.filter(_.element.getName() == name).toArray
    if (matchNodes.length == 0) {
      throw new NoSuchElementException(s"Can not find node with name $name")
    } else {
      return matchNodes.head
    }
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
  private var gradGraph: DirectedGraph[AbstractModule[Activity, Activity, T]] = _

  /**
   * Execution plan
   */
  private val forwardNodes = backGraph.DFS.toArray
  private val forwardScheduler = new Scheduler(
    forwardNodes.filter(_.prevNodes.length == 0),
    Seq(dummyOutput),
    forwardNodes.map(_.element.getName()).toSet
  )

  private var backwardScheduler : Scheduler[T] = _
  private var backwardNodes: Array[Node[AbstractModule[Activity, Activity, T]]] = _


  modules.appendAll(backGraph.topologySort
    .filterNot(_.element.isInstanceOf[ControlDependency[T]]).reverse
    .filter(n => !n.eq(dummyOutput)).map(_.element))

  /**
   * Generate backward graph and apply the stopGrad
   */
  private[bigdl] def build(): this.type = {
    val gradGraph = backGraph.cloneGraph(true)
    dummyOutputGrad = gradGraph.source
    val originalNodes = gradGraph.DFS
    originalNodes.filter(x => isStopGradient(x.element)).foreach(removeStopNodes(_))
    backwardNodes = gradGraph.DFS.filter(n => !n.eq(dummyOutputGrad))
      .filterNot(_.element.isInstanceOf[ControlDependency[_]]).toArray

    val inputNames = inputs.map(_.element.getName()).toSet
    val dummyBackwardEnd = Input()
    val backwardTargets = backwardNodes
      .filter(n => (n.element.parameters() != null && n.element.parameters()._1.length != 0)
        || inputNames.contains(n.element.getName()))
    backwardTargets.foreach(_ -> dummyBackwardEnd)
    val graph = dummyBackwardEnd.graph(true)
    val forwardNodeNames = forwardNodes.map(_.element.getName()).toSet
    val executableNodes = graph.DFS.map(_.element.getName())
      .filter(forwardNodeNames.contains(_)).toSet
    dummyBackwardEnd.removePrevEdges()

    backwardScheduler = new Scheduler[T](
      Seq(dummyOutputGrad),
      backwardTargets,
      executableNodes
    )
    clearState()
    this
  }

  private[bigdl] def removeStopNodes(n: Node[_]): Unit = {
    val nodes = n.nextNodes
    n.removeNextEdges()
    nodes.filter(_.prevNodes.length == 0).foreach(removeStopNodes(_))
  }


  private val inputCache = new mutable.HashMap[String, Activity]()

  // Check all inputs of the graph should be passed in
  checkRoots
  if (generateBackward) {
    forwardNodes.foreach(n => require(!n.element.isInstanceOf[ControlOps[_]],
      "Not suppot generate back graph with control ops node"))
    build()
  }

  private val gradOutputCache = new mutable.HashMap[String, Activity]()

  private def duplicatedNames(names: Seq[String]): mutable.Set[String] = {
    names.sortWith(_ < _)
    val buffer = new mutable.HashSet[String]()
    var i = 1
    while(i < names.length) {
      if (names(i) == names(i - 1)) buffer.add(names(i))
      i += 1
    }
    buffer
  }

  private def checkRoots: Unit = {
    require(forwardNodes.map(_.element.getName()).distinct.length == forwardNodes.length,
      s"the name of node in the graph should be unique, but find duplicated name " +
        s"${duplicatedNames(forwardNodes.map(_.element.getName())).mkString(", ")}")
    val roots = forwardNodes.filter(_.prevNodes.size == 0)
      .filter(node => !node.element.isInstanceOf[WithoutInput]
        && !node.element.isInstanceOf[ControlDependency[_]])
    require(roots.size == inputs.filter(node => !node.element.isInstanceOf[WithoutInput]).length,
      s"There're ${inputs.length} inputs, but graph has ${roots.size} roots")
    inputs.filter(node => !node.element.isInstanceOf[WithoutInput]).foreach(n =>
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
    forwardNodes.filter(n => !n.eq(dummyOutput))
  }

  /**
   * Get forward executions, the dummy nodes and control dependency nodes will be filtered.
   *
   * This method will output a sorted executions. If the graph contains loop, it will throw an
   * exception
   * @return
   */
  def getSortedForwardExecutions: Array[Node[AbstractModule[Activity, Activity, T]]] = {
    backGraph.topologySort
      .filterNot(_.element.isInstanceOf[ControlDependency[T]]).reverse
      .filter(n => !n.eq(dummyOutput))
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
        // if 'activity' and 'other' are both table, we need to merge 'other' to 'activity'
        // if 'other' and 'activity' both contains the index, update 'activity' by sum
        // if 'other' contains the index while 'activity' does not,
        // just insert the corresponding tensor of 'other' to 'activity'
        val actTable = activity.toTable
        val otherTable = other.toTable
        otherTable.keySet.foreach(index => {
          if (actTable.contains(index)) {
            accActivity(actTable[Activity](index), otherTable[Activity](index))
          } else {
            actTable.insert(index.asInstanceOf[Int], otherTable(index))
          }
        })
        actTable
      }
    }
  }

  /**
   * Save current model graph to a folder, which can be display in tensorboard by running
   *   tensorboard --logdir logPath
   * @param logPath
   * @param backward Draw backward graph instead of forward
   * @return
   */
  def saveGraphTopology(logPath: String, backward: Boolean = false): this.type = {
    val writer = new TFFileWriter(logPath)
    val graphBuilder = GraphDef.newBuilder()
    val nodes = if (backward) {
      backwardNodes.filter(n => !n.eq(dummyOutputGrad))
    } else {
      forwardNodes.filter(n => !n.eq(dummyOutput))
    }
    nodes.map(m => {
      val nodeDef = Tensorflow.bigdlModule(m.element, m.prevNodes.map(_.element.getName()).asJava)
      graphBuilder.addNode(nodeDef)
    })

    writer.addGraphDef(graphBuilder.build())
    writer.close()
    this
  }

  def resetModules(): Unit = {
    modules.clear()
    modules.appendAll(backGraph.topologySort
      .filterNot(_.element.isInstanceOf[ControlDependency[T]]).reverse
      .filter(n => !n.eq(dummyOutput)).map(_.element))
  }
}