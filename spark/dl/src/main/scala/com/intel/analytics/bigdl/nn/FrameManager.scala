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

import java.util.concurrent.atomic.AtomicInteger

import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.tf.{Exit, MergeOps, NextIteration}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
 * Manage frame in scheduler. When scheduler execute nodes, it may enter a `frame`. Before
 * scheduler leave a frame, it must make sure all nodes in that frames has been run.
 * @tparam T
 */
class FrameManager[T] extends Serializable {
  import FrameManager._

  /**
   * Create a frame. If it exists, nothing happen
   *
   * @param frame
   * @param parentFrame
   */
  def createFrame(frame: String, parentFrame: Option[Frame[T]]): Frame[T] = {
    if (!frames.contains(frame)) {
      frames(frame) = new Frame(frame, parentFrame)
    }
    frames(frame)
  }
  /**
   * Mark a node enter into frame. The node cannot be in two frames at the same time
   * @param node node
   * @param frame the name of the frame
   */
  def enter(node: ModuleNode[T], frame : Frame[T]): Unit = {
    val name = node.element.getName()
    if (nodeFrame.contains(name)) {
      require(nodeFrame(name).eq(frame), "node cannot be in two different fames at the same time")
    } else {
      nodeFrame(name) = frame
    }

    if (!frame.nodes.contains(node)) {
      if (isExecuteManyTimes(node, frame)) frame.nodes.append(node)
    }
  }

  def pend(node: ModuleNode[T], frame : Frame[T]): Unit = {
    val name = node.element.getName()
    require(node.element.isInstanceOf[NextIteration[_, _]], "you can only pend next iteration node")
    if (nodeFrame.contains(name)) {
      require(nodeFrame(name).eq(frame), "node cannot be in two different fames at the same time")
    } else {
      nodeFrame(name) = frame
    }

    frame.barrier.decrementAndGet()
    frame.waitingNodes.append(node)
  }

  /**
   * Check if the node should be executed many times in the loop
   * @param node
   * @param frame
   * @return
   */
  private def isExecuteManyTimes(node: ModuleNode[T], frame : Frame[T]): Boolean = {
    // Here's a little tricky. We find the begin of these execute many times nodes by looking for
    // pattern of "NextIteration -> Merge"
    if (node.element.isInstanceOf[MergeOps[_]] && node.prevNodes.size == 2 &&
      (node.prevNodes(0).element.isInstanceOf[NextIteration[_, _]] ||
        node.prevNodes(1).element.isInstanceOf[NextIteration[_, _]])) {
      return true
    }

    // If its parents will be re-executed, it will be re-executed
    node.prevNodes.foreach(n => if (frame.nodes.contains(n)) return true)

    return false
  }

  /**
   * Get the frame of the given node. If the node isn't in any frame, return None.
   * @param node
   * @return
   */
  def apply(node: ModuleNode[T]): Option[Frame[T]] = {
    nodeFrame.get(node.element.getName())
  }

  private val frames = new mutable.HashMap[String, Frame[T]]()
  private val nodeFrame = new mutable.HashMap[String, Frame[T]]()
}

object FrameManager {
  /**
   * A frame
   * @param name the name of a frame, it must be unique in a graph
   * @param parent parent frame, if a frame is created in another frame, it has parent frame
   * @tparam T
   */
  class Frame[T] private[FrameManager] (
    val name: String,
    val parent: Option[Frame[T]]
  ) {
    // Sync all next iteration nodes execution
    private[bigdl] var barrier: AtomicInteger = new AtomicInteger(0)
    // User can use NextIteration to sync execution. This is a list of those type of nodes
    private[bigdl] val waitingNodes: ArrayBuffer[ModuleNode[T]] = new ArrayBuffer[ModuleNode[T]]()

    // Nodes should be refreshed in a iteration of the frame
    private[bigdl] val nodes: ArrayBuffer[ModuleNode[T]] = new ArrayBuffer[ModuleNode[T]]()
  }
}
