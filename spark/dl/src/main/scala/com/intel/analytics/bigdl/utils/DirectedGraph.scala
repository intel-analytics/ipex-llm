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

import java.util

import com.intel.analytics.bigdl.tensor.Tensor

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
    while(inDegrees.nonEmpty) {
      // toArray is not lazy eval, which is not affected by inDegrees - 1 operations below
      val startNodes = inDegrees.filterKeys(inDegrees(_) == 0).keySet.toArray
      require(startNodes.length != 0, "There's a cycle in the graph")
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
        // to preserve order
        val nodesSet = mutable.LinkedHashSet[Node[T]]()
        nextNodes.foreach(nodesSet.add)
        nodesSet.filter(!visited.contains(_))
          .filter(!stack.contains(_)).foreach(stack.push(_))
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
        // to preserve order
        val nodesSet = mutable.LinkedHashSet[Node[T]]()
        nextNodes.foreach(nodesSet.add)
        nodesSet.filter(!visited.contains(_))
          .filter(!queue.contains(_)).foreach(queue.enqueue(_))
        node
      }
    }
  }
  // scalastyle:on methodName

  /**
   * Clone the graph structure, will not clone the node element
   * @param reverseEdge if reverse the edge in the nodes
   * @return
   */
  def cloneGraph(reverseEdge: Boolean = false): DirectedGraph[T] = {
    val oldToNew = new util.HashMap[Node[T], Node[T]]()
    val bfs = BFS.toArray
    bfs.foreach(node => {
      oldToNew.put(node, new Node[T](node.element))
    })
    // Keep the order in the nextNodes array and prevNodes array of the current node.
    // As we go through all node in bfs from source, the prevNodes order can be preserved.
    // For each node, we iterate and add their nextNodes, the nextNodes order can also be preserved.
    bfs.foreach(node => {
      if (reverseEdge) {
        node.nextNodesAndEdges.foreach(nextNodeAndEdge => {
          // Some next nodes may be not included in the graph
          if (oldToNew.containsKey(nextNodeAndEdge._1)) {
            oldToNew.get(node).addPrevious(
              oldToNew.get(nextNodeAndEdge._1), nextNodeAndEdge._2)
          }
        })
        node.prevNodesAndEdges.foreach(prevNodeAndEdge => {
          if (oldToNew.containsKey(prevNodeAndEdge._1)) {
            oldToNew.get(node).addNexts(
              oldToNew.get(prevNodeAndEdge._1), prevNodeAndEdge._2)
          }
        })
      } else {
        node.nextNodesAndEdges.foreach(nextNodeAndEdge => {
          if (oldToNew.containsKey(nextNodeAndEdge._1)) {
            oldToNew.get(node).add(oldToNew.get(nextNodeAndEdge._1), nextNodeAndEdge._2)
          }
        })
      }
    })

    if (reverseEdge) {
      new DirectedGraph[T](oldToNew.get(source), !reverse)
    } else {
      new DirectedGraph[T](oldToNew.get(source), reverse)
    }
  }
}

/**
 * Represent a node in a graph. The connections between nodes are directed.
 * @param element element
 * @tparam T element type
 */
@SerialVersionUID(- 6021651923538325999L)
class Node[T](var element: T) extends Serializable {
  /**
   * The nodes pointed by current node
   * @return
   */
  def nextNodes: Seq[Node[T]] = nexts.map(_._1)

  /**
   * The edges start from this node
   * @return
   */
  def nextEdges: Seq[Edge] = nexts.map(_._2)

  /**
   * The nodes pointed by current node with the connect edges
   * @return
   */
  def nextNodesAndEdges: Seq[(Node[T], Edge)] = nexts

  /**
   * The nodes point to current node
   * @return
   */
  def prevNodes: Seq[Node[T]] = prevs.map(_._1)

  /**
   * The edges connect to this node
   * @return
   */
  def prevEdges: Seq[Edge] = prevs.map(_._2)

  /**
   * The nodes pointed to current node with the connect edges
   * @return
   */
  def prevNodesAndEdges: Seq[(Node[T], Edge)] = prevs

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
  def add(node: Node[T], e: Edge = Edge()): Node[T] = {
    if (!node.prevs.contains((this, e))) node.prevs.append((this, e))
    if (!this.nexts.contains((node, e))) this.nexts.append((node, e))
    node
  }

  def addPrevious(node: Node[T], e: Edge = Edge()): Unit = {
    if (!this.prevs.contains((node, e))) this.prevs.append((node, e))
  }

  def addNexts(node: Node[T], e: Edge = Edge()): Unit = {
    if (!this.nexts.contains((node, e))) this.nexts.append((node, e))
  }

  def from(node: Node[T], e: Edge = Edge()): Node[T] = {
    if (!node.nexts.contains((this, e))) node.nexts.append((this, e))
    if (!this.prevs.contains((node, e))) this.prevs.append((node, e))
    node
  }

  /**
   * Remove linkage with another node
   *  @param node another node
   *  @return current node
   */
  def delete(node: Node[T], e: Edge = null): Node[T] = {
    if (e != null) {
      if (node.prevs.contains((this, e))) node.prevs.-=((this, e))
      if (this.nexts.contains((node, e))) this.nexts.-=((node, e))
    } else {
      val curNode = this  // Because of the closure
      node.prevs.filter(_._1 == curNode).foreach(k => node.prevs.-=(k))
      this.nexts.filter(_._1 == node).foreach(k => this.nexts.-=(k))
    }
    this
  }

  /**
   * A sugar allows user to generate the pair (n, something) via n(something)
   * @param meta
   * @tparam M
   * @return
   */
  def apply[M](meta: M): (this.type, M) = {
    (this, meta)
  }

  /**
   * remove edges that connect previous nodes
   * @return current node
   */
  def removePrevEdges(): Node[T] = {
    val curNode = this  // Because of the closure
    prevs.map(_._1).foreach(pn =>
      pn.nexts.filter(_._1 == curNode).foreach(e =>
        pn.nexts -= e
      )
    )
    prevs.clear()
    this
  }

  /**
   * remove edges that connect next nodes
   * @return current node
   */
  def removeNextEdges(): Node[T] = {
    val curNode = this  // Because of the closure
    nexts.map(_._1).foreach(pn =>
      pn.prevs.filter(_._1 == curNode).foreach(e =>
        pn.prevs -= e
      )
    )
    nexts.clear()
    this
  }

  def setElement(e: T): this.type = {
    element = e
    this
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

  private val nexts = new ArrayBuffer[(Node[T], Edge)]()
  private val prevs = new ArrayBuffer[(Node[T], Edge)]()
}

object Node {
  def apply[T](element: T): Node[T] = new Node(element)
}

/**
 * An edge in the graph
 * @param fromIndex A preserved position to store meta info.
 */
private[bigdl] class Edge private (val fromIndex: Option[Int]) extends Serializable {
  override def toString: String = {
    s"Edge(fromIndex: $fromIndex)"
  }

  /**
   * Create a new Instance of this Edge
   * @return a new Instance of this Edge
   */
  def newInstance(): Edge = {
    fromIndex match {
      case Some(index) => Edge(index)
      case None => Edge()
    }
  }
}

object Edge {
  def apply(value : Int): Edge = new Edge(Some(value))
  def apply(): Edge = new Edge(None)
}
