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
import com.intel.analytics.bigdl.nn.FrameManager.Frame
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.ops._
import com.intel.analytics.bigdl.nn.tf._

import scala.collection.mutable

/**
 * Scheduler of a graph execution. It supports a graph with cycle. Please note that the cycle must
 * be created from ControlNodes.while.
 *
 * Scheduler also records execution status. So some const graph won't be executed multiple times.
 *
 * @param inputNodes start nodes
 * @param outputNodes target nodes
 * @tparam T
 */
private[bigdl] class Scheduler[T] (
    inputNodes: Seq[ModuleNode[T]], outputNodes: Seq[ModuleNode[T]],
    executableNodes: Set[String] = null
  ) extends Serializable {

  import Scheduler._

  private val readyQueue = new mutable.Queue[ModuleNode[T]]()
  private val nodeStatus = new NodeStatusManager[T]()
  private val frameManayger = new FrameManager[T]()

  /**
   * User must reset the scheduler after first use it or finish a graph execution
   */
  def reset(): Unit = {
    readyQueue.clear()
    inputNodes.foreach(n => {
      readyQueue.enqueue(n)
    })
    nodeStatus.removeUnConstStatus()
  }


  /**
   * If every output nodes is executed. Please note if some of the output nodes has not been
   * executed but the execution can't move forward, an exception will be thrown
   * @return
   */
  def isFinished(): Boolean = {
    val isEmpty = readyQueue.isEmpty
    if (isEmpty) {
      outputNodes.foreach(n => {
        require(!nodeStatus.notExecuted(n), "Some output nodes have not been executed")
      })
    }
    isEmpty
  }


  /**
   * Fetch a node to execute. Don't call it when isFinished is true, or it will throw an exception
   * @return
   */
  def fetch(): ModuleNode[T] = {
    var node = readyQueue.dequeue()
    while (skipExecution(node)) {
      node = readyQueue.dequeue()
    }

    node
  }

  private def skipExecution(node: ModuleNode[T]): Boolean = {
    if (node.element.isInstanceOf[ControlDependency[_]] || nodeStatus.isConst(node)) {
      schedule(node)
      return true
    }

    return false
  }

  /**
   * Schedule nodes depend on the given node
   * @param node
   */
  def schedule(node: ModuleNode[T]): Unit = {
    val curFrame = frameManayger(node)

    val nextNodeFrame = if (node.element.isInstanceOf[Enter[_]]) {
      val e = node.element.asInstanceOf[Enter[_]]
      Some(frameManayger.createFrame(e.frame, curFrame))
    } else if (node.element.isInstanceOf[LoopCondition[_]]) {
      require(curFrame.isDefined, "LoopCondition should be in a frame")
      val f = curFrame.get
      require(f.barrier.get() == 0, "frame barrier should be 0 when execute loop condition")
      f.barrier.set(node.nextNodes.size)
      curFrame
    } else if (node.element.isInstanceOf[NextIteration[_, _]]) {
      require(curFrame.isDefined, "NextIteration should be in a frame")
      curFrame
    } else if (node.element.isInstanceOf[Exit[_]]) {
      require(curFrame.isDefined, "Exit should be in a frame")
      val f = curFrame.get
      f.barrier.set(0)
      f.parent
    } else {
      curFrame
    }

    if (!nodeStatus.isConst(node)) {
      // Update status of current node
      nodeStatus(node) = if (node.prevNodes.length == 0) {
        if (node.element.isInstanceOf[com.intel.analytics.bigdl.nn.tf.Const[_, _]]) {
          Const()
        } else {
          Ready()
        }
      } else {
        val constNodes = node.prevNodes.filter(nodeStatus.isConst(_))
        if (constNodes.length == node.prevNodes.length && !node.element.isInstanceOf[RandomNode]) {
          Const()
        } else {
          Ready()
        }
      }
    }

    // Schedule next nodes
    node.element match {
      case s: SwitchOps[_] =>
        val switchNode = node.asInstanceOf[SwitchControlNode[Module[T]]]
        selectNexts(switchNode.availableNodes(), node, nextNodeFrame)
      case _ =>
        selectNexts(node.nextNodes, node, nextNodeFrame)
    }
  }

  private def startNextIteration(frame: Frame[T]): Unit = {
    // Wake up the waiting nodes
    frame.waitingNodes.foreach(readyQueue.enqueue(_))
    frame.waitingNodes.clear()

    // As frame is refreshed, mark all nodes in the frame are not ready
    frame.nodes.filterNot(_.element.isInstanceOf[NextIteration[_, _]]).foreach(n => {
      nodeStatus.unset(n)
    })
  }

  private def selectNexts(candidateNodes: Seq[ModuleNode[T]], curNode: ModuleNode[T],
    frame: Option[Frame[T]]): Unit = {
    val nodeSet = new mutable.LinkedHashSet[ModuleNode[T]]()
    candidateNodes.foreach(nodeSet.add(_))  // remove duplicate nodes and keep the order

    nodeSet.filter(n => executableNodes.contains(n.element.getName())).foreach(nextNode => {
      if (nextNode.element.isInstanceOf[MergeOps[_]]) {
        val merge = nextNode.element.asInstanceOf[MergeOps[_]]
        require(nodeStatus.notExecuted(nextNode), s"Merge node(${nextNode.element.getName()}) " +
          s"should not be executed twice out of loop or in a same iteration of a loop")
        merge.setSwitch(nextNode.prevNodes.indexOf(curNode) + 1)
        enQueue(nextNode, frame)
      } else {
        if (isNodeReady(nextNode)) {
          enQueue(nextNode, frame)
        }
      }
    })
  }

  private def isNodeReady(node: ModuleNode[T]): Boolean = {
    if (node.prevNodes.filter(nodeStatus.notExecuted(_)).length != 0) {
      return false
    }
    node.prevNodes.filter(_.isInstanceOf[SwitchControlNode[_]]).foreach(n => {
      if (!n.asInstanceOf[SwitchControlNode[T]].availableNodes().contains(node)) {
        return false
      }
    })

    return true
  }

  private def enQueue(node: ModuleNode[T], frame: Option[Frame[T]]): Unit = {
    if (node.element.isInstanceOf[NextIteration[_, _]]) {
      require(frame.isDefined, "current node should be in a frame")
      frameManayger.pend(node, frame.get)
      nodeStatus.unset(node) // mark current node is in not ready status
      if (frame.get.barrier.get() == 0) {
        startNextIteration(frame.get)
      }
    } else {
      frame.foreach(frameManayger.enter(node, _))
      readyQueue.enqueue(node)
    }
  }
}

object Scheduler {
  class NodeStatusManager[T] extends Serializable {
    private val nodeStatus = new mutable.HashMap[String, NodeStatus]()

    /**
     * Update node status
     * @param node
     * @param status
     */
    def update(node: ModuleNode[T], status: NodeStatus): Unit = {
      require(node != null && status != null, "Not accept null")
      nodeStatus(node.element.getName()) = status
    }

    /**
     * Get status of node. Throw exception if it doesn't exist.
     * @param node
     * @return
     */
    def apply(node: ModuleNode[T]): NodeStatus = {
      nodeStatus(node.element.getName())
    }

    /**
     * Check if a given node status is const
     * @param node
     * @return
     */
    def isConst(node: ModuleNode[T]): Boolean = {
      nodeStatus.contains(node.element.getName()) &&
        nodeStatus(node.element.getName()).isInstanceOf[Const]
    }

    /**
     * If the given node has been executed or out of date
     * @param node
     * @return
     */
    def notExecuted(node: ModuleNode[T]): Boolean = {
      if (!nodeStatus.contains(node.element.getName())) return true
      return false
    }

    /**
     * Remove unconst node status
     * @return
     */
    def removeUnConstStatus(): this.type = {
      val iter = nodeStatus.iterator
      while (iter.hasNext) {
        val entry = iter.next()
        if (!entry._2.isInstanceOf[Const]) {
          nodeStatus.remove(entry._1)
        }
      }
      this
    }

    /**
     * Remove status of node.
     * @param node
     */
    def unset(node: ModuleNode[T]): Unit = {
      nodeStatus.remove(node.element.getName())
    }
  }


  /**
   * Node status
   */
  private[nn] sealed trait NodeStatus

  /**
   * Current node is const or all of its dependencies are from const node
   */
  private[nn] case class Const() extends NodeStatus

  /**
   * Current nodes has been executed, while it's not const
   */
  private[nn] case class Ready() extends NodeStatus
}
