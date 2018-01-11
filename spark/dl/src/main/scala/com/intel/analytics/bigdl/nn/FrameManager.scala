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
      frames(frame) = new Frame(frame, 0, new ArrayBuffer[ModuleNode[T]], parentFrame)
    }
    frames(frame)
  }
  /**
   * Mark a node enter into frame. The node should not be in any frame before and frame should
   * exist.
   * @param node node
   * @param frame the name of the frame
   */
  def enter(node: ModuleNode[T], frame : Frame[T]): Unit = {
    require(!nodeFrame.contains(node.element.getName()), "Node already in a frame")
    nodeFrame(node.element.getName()) = frame
  }

  /**
   * The node leaves its frame. If the node doesn't exist in any frame, nothing happen.
   *
   * This method return frame the node leave. If the node belong to no frame, None will be return.
   *
   * @param node
   * @return
   */
  def leave(node: ModuleNode[T]): Option[Frame[T]] = {
    nodeFrame.remove(node.element.getName())
  }

  /**
   * Get a frame. The frame must exist.
   * @param node
   * @return
   */
  def apply(node: ModuleNode[T]): Option[Frame[T]] = nodeFrame.get(node.element.getName())

  /**
   * Remove a frame from a frame manager.
   * @param frame
   */
  def release(frame: Frame[T]): Unit = {
    require(frames.contains(frame.name), "Cannot remove the given frame")
    frames.remove(frame.name)
  }

  private val frames = new mutable.HashMap[String, Frame[T]]()
  private val nodeFrame = new mutable.HashMap[String, Frame[T]]()
}

object FrameManager {
  /**
   * A frame
   * @param name the name of a frame, it must be unique in a grap
   * @param mutex sync all next iteration / exit nodes execution
   * @param pendingNodes user can use NextIteration or Exit to leave current frame. This is a list
   *                     of those type of nodes, which are ready to leave
   * @param parent parent frame, if a frame is created in another frame, it has parent frame
   * @tparam T
   */
  class Frame[T] private[FrameManager] (
    val name: String,
    var mutex: Int,
    val pendingNodes: ArrayBuffer[ModuleNode[T]],
    val parent: Option[Frame[T]]
  )
}