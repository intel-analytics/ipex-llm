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

import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.ops.ControlOps
import com.intel.analytics.bigdl.nn.tf.{ControlDependency, WithoutInput}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.T

import scala.collection.mutable
import scala.reflect.ClassTag

class DynamicGraph[T: ClassTag](
  private val _inputs : Seq[ModuleNode[T]],
  private val _outputs : Seq[ModuleNode[T]],
  private val _variables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None,
  private val generateBackward: Boolean = true
)(implicit ev: TensorNumeric[T])
  extends Graph[T](_inputs, _outputs, _variables) {

  override def updateOutput(input: Activity): Activity = {
    forwardScheduler.reset()
    while (!forwardScheduler.isFinished()) {
      val node = forwardScheduler.fetch()
      val nodeInput = if (node.prevNodes.isEmpty && !node.element.isInstanceOf[WithoutInput]) {
        getInput(node, input)
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
          T(prevActivities)
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
      T(inputs.map(n => n.element.gradInput))
    }
    backwardTime = System.nanoTime() - before
    gradInput
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
      T(inputs.map(n => n.element.gradInput))
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

  private val forwardScheduler = new Scheduler(
    forwardNodes.filter(_.prevNodes.length == 0),
    Seq(dummyOutput),
    forwardNodes.map(_.element.getName()).toSet
  )
  private var backwardScheduler : Scheduler[T] = _

  /**
   * Generate backward graph and apply the stopGrad
   */
  override private[bigdl] def buildBackwardGraph(): this.type = {
    super.buildBackwardGraph()
    val forwardNodeNames = forwardNodes.map(_.element.getName()).toSet
    val executableNodes = backwardGraph.DFS.map(_.element.getName())
      .filter(forwardNodeNames.contains(_)).toSet

    val inputNames = inputs.map(_.element.getName()).toSet
    val backwardTargets = backwardNodes
      .filter(n => (n.element.parameters() != null && n.element.parameters()._1.length != 0)
        || inputNames.contains(n.element.getName()))

    backwardScheduler = new Scheduler[T](
      Seq(dummyOutputGrad),
      backwardTargets,
      executableNodes
    )
    clearState()
    this
  }

  private val inputCache = new mutable.HashMap[String, Activity]()
  private val gradOutputCache = new mutable.HashMap[String, Activity]()

  if (generateBackward) {
    forwardNodes.foreach(n => require(!n.element.isInstanceOf[ControlOps[_]],
      "Not suppot generate back graph with control ops node"))
    buildBackwardGraph()
  }
}
