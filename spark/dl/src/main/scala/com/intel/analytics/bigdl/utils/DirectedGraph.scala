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
package com.intel.analytics.bigdl.utils

import com.intel.analytics.bigdl.Module

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
 * Provides useful graph operations for a directed graph.
 *
 * The node connection in the graph has direction. This class stores a source node. Note that it
 * doesn't maintain the topology. The topology of the graph is stored in the connection of the
 * nodes.
 * @param source source node of the directed graph
 * @param reverse use the original direction or the reversed direction
 * @tparam T Node element type
 */
@SerialVersionUID(- 6252604964316218479L)
class DirectedGraph[T](val source : Node[T], val reverse : Boolean = false) extends Serializable {

  /**
   * How many nodes in the graph
   * @return
   */
  def size : Int = BFS.size

  /**
   * How many edges in the graph
   * @return
   */
  def edges : Int = BFS.map(_.nextNodes.length).reduce(_ + _)

  /**
   * Topology sort.
   * @return A sequence of sorted graph nodes
   */
  def topologySort : Array[Node[T]] = {
    // Build indegree list, LinkedHashMap can preserve the order of the keys, so it's good to
    // write unittest.
    val inDegrees = new mutable.LinkedHashMap[Node[T], Int]()
    inDegrees(source) = 0
    DFS.foreach(n => {
      val nextNodes = if (!reverse) n.nextNodes else n.prevNodes
      nextNodes.foreach(m => {
        inDegrees(m) = inDegrees.getOrElse(m, 0) + 1
      })
    })

    val result = new ArrayBuffer[Node[T]]()
    while(!inDegrees.isEmpty) {
      // toArray is not lazy eval, which is not affected by inDegrees - 1 operations below
      val startNodes = inDegrees.filterKeys(inDegrees(_) == 0).keySet.toArray
      require(startNodes.size != 0, "There's a cycle in the graph")
      result.appendAll(startNodes)
      startNodes.foreach(n => {
        val nextNodes = if (!reverse) n.nextNodes else n.prevNodes
        nextNodes.foreach(nextNode => inDegrees(nextNode) = inDegrees(nextNode) - 1)
        inDegrees.remove(n)
      })
    }
    result.toArray
  }

  // scalastyle:off methodName
  /**
   * Depth first search on the graph. Note that this is a directed DFS. Although eachs node
   * contains both previous and next nodes, only one direction is used.
   * @return An iterator to go through nodes in the graph in a DFS order
   */
  def DFS : Iterator[Node[T]] = {
    new Iterator[Node[T]] {
      private val stack = new mutable.Stack[Node[T]]().push(source)
      private val visited = new mutable.HashSet[Node[T]]()

      override def hasNext: Boolean = !stack.isEmpty

      override def next(): Node[T] = {
        require(hasNext, "No more elements in the graph")
        val node = stack.pop()
        visited.add(node)
        val nextNodes = if (!reverse) node.nextNodes else node.prevNodes
        nextNodes.filter(!visited.contains(_)).filter(!stack.contains(_)).foreach(stack.push(_))
        node
      }
    }
  }

  /**
   * Breadth first search on the graph. Note that this is a directed BFS. Although eachs node
   * contains both previous and next nodes, only one direction is used.
   * @return An iterator to go through nodes in the graph in a BFS order
   */
  def BFS : Iterator[Node[T]] = {
    new Iterator[Node[T]] {
      private val queue = new mutable.Queue[Node[T]]()
      queue.enqueue(source)
      private val visited = new mutable.HashSet[Node[T]]()

      override def hasNext: Boolean = !queue.isEmpty

      override def next(): Node[T] = {
        require(hasNext, "No more elements in the graph")
        val node = queue.dequeue()
        visited.add(node)
        val nextNodes = if (!reverse) node.nextNodes else node.prevNodes
        nextNodes.filter(!visited.contains(_)).filter(!queue.contains(_)).foreach(queue.enqueue(_))
        node
      }
    }
  }
  // scalastyle:on methodName
}

/**
 * Represent a node in a graph. The connections between nodes are directed.
 * @param element element
 * @tparam T element type
 */
@SerialVersionUID(- 6021651923538325999L)
class Node[T](val element: T) extends Serializable {
  /**
   * The nodes pointed by current node
   * @return
   */
  def nextNodes: Seq[Node[T]] = nexts

  /**
   * The nodes point to currect node
   * @return
   */
  def prevNodes: Seq[Node[T]] = prevs

  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  /**
   * Point to another node
   * @param node another node
   * @return another node
   */
  def -> (node: Node[T]): Node[T] = {
    this.add(node)
  }
  // scalastyle:on noSpaceBeforeLeftBracket
  // scalastyle:on methodName

  /**
   * Point to another node
   * @param node another node
   * @return another node
   */
  def add(node: Node[T]): Node[T] = {
    if (!node.prevs.contains(this)) node.prevs.append(this)
    if (!this.nexts.contains(node)) this.nexts.append(node)
    node
  }

  /**
   * Use current node as source to build a direct graph
   * @param reverse
   * @return
   */
  def graph(reverse : Boolean = false) : DirectedGraph[T] = {
    new DirectedGraph[T](this, reverse)
  }

  override def toString: String = s"(${element.toString})"

  private val nexts = new ArrayBuffer[Node[T]]()
  private val prevs = new ArrayBuffer[Node[T]]()
}

object Node {
  def apply[T](element : T) : Node[T] = new Node(element)
}
