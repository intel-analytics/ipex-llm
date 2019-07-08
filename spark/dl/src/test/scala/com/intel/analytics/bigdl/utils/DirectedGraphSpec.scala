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
import com.intel.analytics.bigdl.nn.{CAddTable, Graph, Input, Reshape}
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}

class DirectedGraphSpec extends FlatSpec with Matchers {
  "Node add" should "be correct" in {
    val nodeA = new Node("A")
    val nodeB = new Node("B")
    val test = nodeA -> nodeB
    test should be(nodeB)
    nodeA.prevNodes.length should be(0)
    nodeA.nextNodes.length should be(1)
    nodeA.nextEdges.length should be(1)
    nodeA.nextNodes(0) should be(nodeB)
    nodeA.nextEdges(0).fromIndex should be(None)
    nodeB.prevNodes.length should be(1)
    nodeB.prevEdges.length should be(1)
    nodeB.prevNodes(0) should be(nodeA)
    nodeB.prevEdges(0).fromIndex should be(None)
    nodeB.nextNodes.length should be(0)
    nodeB.nextEdges.length should be(0)
  }

  "Node add with edge" should "be correct" in {
    val nodeA = new Node("A")
    val nodeB = new Node("B")
    val test = nodeA.add(nodeB, Edge(1))
    test should be(nodeB)
    nodeA.prevNodes.length should be(0)
    nodeA.nextNodes.length should be(1)
    nodeA.nextEdges.length should be(1)
    nodeA.nextNodes(0) should be(nodeB)
    nodeA.nextEdges(0).fromIndex.get should be(1)
    nodeB.prevNodes.length should be(1)
    nodeB.prevEdges.length should be(1)
    nodeB.prevNodes(0) should be(nodeA)
    nodeB.prevEdges(0).fromIndex.get should be(1)
    nodeB.nextNodes.length should be(0)
    nodeB.nextEdges.length should be(0)
  }

  "Topology sort" should "be correct in a graph" in {
    val nodeA = new Node("A")
    val nodeB = new Node("B")
    val nodeC = new Node("C")
    val nodeD = new Node("D")
    val nodeE = new Node("E")
    val nodeF = new Node("F")
    val nodeG = new Node("G")
    val nodeH = new Node("H")
    nodeA -> nodeB -> nodeE -> nodeF -> nodeG
    nodeA -> nodeC -> nodeF
    nodeA -> nodeD -> nodeF
    nodeF -> nodeH

    val graph = nodeA.graph()
    val sorted = graph.topologySort.map(_.element)
    sorted.length should be(8)
    sorted.indexOf("B") > sorted.indexOf("A") should be(true)
    sorted.indexOf("C") > sorted.indexOf("A") should be(true)
    sorted.indexOf("D") > sorted.indexOf("A") should be(true)
    sorted.indexOf("E") > sorted.indexOf("B") should be(true)
    sorted.indexOf("F") > sorted.indexOf("E") should be(true)
    sorted.indexOf("F") > sorted.indexOf("C") should be(true)
    sorted.indexOf("F") > sorted.indexOf("D") should be(true)
    sorted.indexOf("G") > sorted.indexOf("F") should be(true)
    sorted.indexOf("H") > sorted.indexOf("F") should be(true)
  }

  "Topology sort" should "be correct in a reversed graph" in {
    val nodeA = new Node("A")
    val nodeB = new Node("B")
    val nodeC = new Node("C")
    val nodeD = new Node("D")
    val nodeE = new Node("E")
    val nodeF = new Node("F")
    val nodeG = new Node("G")
    val nodeH = new Node("H")
    val nodeI = new Node("I")
    nodeA -> nodeB -> nodeE -> nodeF -> nodeG -> nodeI
    nodeA -> nodeC -> nodeF
    nodeA -> nodeD -> nodeF
    nodeF -> nodeH -> nodeI

    val graph = nodeI.graph(true)
    val sorted = graph.topologySort.map(_.element)
    sorted.length should be(9)
    sorted.indexOf("G") > sorted.indexOf("I") should be(true)
    sorted.indexOf("H") > sorted.indexOf("I") should be(true)
    sorted.indexOf("F") > sorted.indexOf("G") should be(true)
    sorted.indexOf("F") > sorted.indexOf("H") should be(true)
    sorted.indexOf("E") > sorted.indexOf("F") should be(true)
    sorted.indexOf("B") > sorted.indexOf("E") should be(true)
    sorted.indexOf("C") > sorted.indexOf("F") should be(true)
    sorted.indexOf("D") > sorted.indexOf("F") should be(true)
    sorted.indexOf("A") > sorted.indexOf("B") should be(true)
    sorted.indexOf("A") > sorted.indexOf("C") should be(true)
    sorted.indexOf("A") > sorted.indexOf("D") should be(true)
  }

  "Topology sort" should "be correct in a sub-graph" in {
    val nodeA = new Node("A")
    val nodeB = new Node("B")
    val nodeC = new Node("C")
    val nodeD = new Node("D")
    val nodeE = new Node("E")
    val nodeF = new Node("F")
    val nodeG = new Node("G")
    nodeA -> nodeD -> nodeE -> nodeF
    nodeB -> nodeD
    nodeC -> nodeE
    nodeE -> nodeG

    val graph = nodeA.graph()
    val sorted = graph.topologySort.map(_.element)
    sorted.length should be(5)
    sorted.indexOf("D") > sorted.indexOf("A") should be(true)
    sorted.indexOf("E") > sorted.indexOf("D") should be(true)
    sorted.indexOf("F") > sorted.indexOf("E") should be(true)
    sorted.indexOf("G") > sorted.indexOf("E") should be(true)
  }

  "Topology sort" should "be correct in a reversed sub-graph" in {
    val nodeA = new Node("A")
    val nodeB = new Node("B")
    val nodeC = new Node("C")
    val nodeD = new Node("D")
    val nodeE = new Node("E")
    val nodeF = new Node("F")
    val nodeG = new Node("G")
    nodeA -> nodeD -> nodeE -> nodeF
    nodeB -> nodeD
    nodeC -> nodeE
    nodeE -> nodeG

    val graph = nodeF.graph(true)
    val sorted = graph.topologySort.map(_.element)
    sorted.length should be(6)
    sorted.indexOf("E") > sorted.indexOf("F") should be(true)
    sorted.indexOf("D") > sorted.indexOf("E") should be(true)
    sorted.indexOf("C") > sorted.indexOf("E") should be(true)
    sorted.indexOf("A") > sorted.indexOf("D") should be(true)
    sorted.indexOf("B") > sorted.indexOf("D") should be(true)
  }

  "Topology sort" should "throw exception in a cycled graph" in {
    val nodeA = new Node("A")
    val nodeB = new Node("B")
    val nodeC = new Node("C")
    val nodeD = new Node("D")
    nodeA -> nodeB -> nodeC -> nodeA
    nodeB -> nodeD

    val graph = nodeA.graph()

    intercept[IllegalArgumentException] {
      val sorted = graph.topologySort.map(_.element)
    }
  }

  "Topology sort" should "throw exception in a reversed cycled graph" in {
    val nodeA = new Node("A")
    val nodeB = new Node("B")
    val nodeC = new Node("C")
    val nodeD = new Node("D")
    nodeA -> nodeB -> nodeC -> nodeA
    nodeB -> nodeD

    val graph = nodeD.graph(true)

    intercept[IllegalArgumentException] {
      val sorted = graph.topologySort.map(_.element)
    }
  }

  "DFS" should "be correct" in {
    val nodeA = new Node("A")
    val nodeB = new Node("B")
    val nodeC = new Node("C")
    val nodeD = new Node("D")
    nodeA -> nodeB -> nodeC -> nodeA
    nodeB -> nodeD

    val graph = nodeA.graph()
    val set = graph.DFS.toSet
    set.size should be(4)
    set should contain(nodeA)
    set should contain(nodeB)
    set should contain(nodeC)
    set should contain(nodeD)
  }

  "DFS" should "be correct for node has multiple outputs" in {
    val nodeA = new Node("A")
    val nodeB = new Node("B")
    val nodeC = new Node("C")
    val nodeD = new Node("D")
    nodeB -> nodeD
    nodeA -> nodeC -> nodeD
    nodeB -> nodeC

    val graph = nodeD.graph(true)
    val set = graph.DFS.toArray
    set.size should be(4)
    set should contain(nodeA)
    set should contain(nodeB)
    set should contain(nodeC)
    set should contain(nodeD)
  }

  "DFS" should "be correct in reverse graph" in {
    val nodeA = new Node("A")
    val nodeB = new Node("B")
    val nodeC = new Node("C")
    val nodeD = new Node("D")
    nodeA -> nodeB -> nodeC -> nodeA
    nodeB -> nodeD

    val graph = nodeD.graph(true)
    val set = graph.DFS.toSet
    set.size should be(4)
    set should contain(nodeA)
    set should contain(nodeB)
    set should contain(nodeC)
    set should contain(nodeD)
  }

  "BFS" should "be correct" in {
    val nodeA = new Node("A")
    val nodeB = new Node("B")
    val nodeC = new Node("C")
    val nodeD = new Node("D")
    nodeA -> nodeB -> nodeC -> nodeA
    nodeB -> nodeD

    val graph = nodeA.graph()
    val set = graph.BFS.toSet
    set.size should be(4)
    set should contain(nodeA)
    set should contain(nodeB)
    set should contain(nodeC)
    set should contain(nodeD)
  }

  "BFS" should "be correct for node has multiple inputs" in {
    val nodeA = new Node("A")
    val nodeB = new Node("B")
    val nodeC = new Node("C")
    val nodeD = new Node("D")
    nodeA -> nodeB
    nodeA -> nodeC
    nodeA -> nodeD
    nodeB -> nodeD

    val graph = nodeA.graph()
    val set = graph.BFS.toArray
    set.size should be(4)
    set should contain(nodeA)
    set should contain(nodeB)
    set should contain(nodeC)
    set should contain(nodeD)
  }

  "BFS" should "be correct in a reversed graph" in {
    val nodeA = new Node("A")
    val nodeB = new Node("B")
    val nodeC = new Node("C")
    val nodeD = new Node("D")
    nodeA -> nodeB -> nodeC -> nodeA
    nodeB -> nodeD

    val graph = nodeD.graph(true)
    val set = graph.BFS.toSet
    set.size should be(4)
    set should contain(nodeA)
    set should contain(nodeB)
    set should contain(nodeC)
    set should contain(nodeD)
  }

  "Edge" should "be not equal for different instances" in {
    val edge1 = Edge(1)
    val edge2 = Edge(1)

    edge1.equals(edge2) should be(false)

    val edge3 = Edge()
    val edge4 = Edge()

    edge3.equals(edge4) should be(false)
  }

  "Clone graph" should "be correct" in {
    val nodeA = new Node("A")
    val nodeB = new Node("B")
    val nodeC = new Node("C")
    val nodeD = new Node("D")
    nodeA -> nodeB -> nodeC
    nodeB -> nodeD

    val graph = nodeA.graph()
    val cloneGraph = graph.cloneGraph()
    val sort1 = graph.topologySort
    val sort2 = cloneGraph.topologySort
    sort1.map(_.element) should be(sort2.map(_.element))
    sort1.zip(sort2).foreach(x => {
      x._1.prevNodes.map(_.element) should be (x._2.prevNodes.map(_.element))
      x._1.nextNodes.map(_.element) should be (x._2.nextNodes.map(_.element))
    })
  }

  "Clone graph" should "should reuse the edge" in {
    val nodeA = new Node("A")
    val nodeB = new Node("B")
    val nodeC = new Node("C")
    val nodeD = new Node("D")
    nodeA -> nodeB -> nodeC
    nodeB -> nodeD

    val graph = nodeA.graph()
    val cloneGraph = graph.cloneGraph()
    graph.topologySort.zip(cloneGraph.topologySort).foreach(n => {
      n._1.prevEdges should be(n._2.prevEdges)
      n._1.nextEdges should be(n._2.nextEdges)
    })
  }

  "Reverse graph" should "be correct" in {
    val nodeA = new Node("A")
    val nodeB = new Node("B")
    val nodeC = new Node("C")
    val nodeD = new Node("D")
    nodeA -> nodeB -> nodeC
    nodeB -> nodeD

    val graph = nodeA.graph()
    val reverseGraph = graph.cloneGraph(true)
    val originSort = graph.topologySort
    val sorted = reverseGraph.topologySort
    originSort.map(_.element) should be(sorted.map(_.element))
    originSort(1).nextNodes.length should be(2)
    originSort(1).prevNodes.length should be(1)
    sorted(1).nextNodes.length should be(1)
    sorted(1).prevNodes.length should be(2)
  }

  "delete edge" should "be correct when specify edge" in {
    val e1 = Edge(1)
    val e2 = Edge(2)
    val nodeA = new Node("A")
    val nodeB = new Node("B")
    nodeA.add(nodeB, e1)
    nodeA.add(nodeB, e2)
    nodeA.delete(nodeB, e1)
    nodeA.nextEdges.length should be(1)
    nodeB.prevEdges.length should be(1)
  }

  "delete edge" should "be correct when not specify edge" in {
    val e1 = Edge(1)
    val e2 = Edge(2)
    val nodeA = new Node("A")
    val nodeB = new Node("B")
    nodeA.add(nodeB, e1)
    nodeA.add(nodeB, e2)
    nodeA.delete(nodeB)
    nodeA.nextEdges.length should be(0)
    nodeB.prevEdges.length should be(0)
  }

  "delete edge" should "be correct when use default edge" in {
    val nodeA = new Node("A")
    val nodeB = new Node("B")
    nodeA -> nodeB
    nodeA.delete(nodeB)
    nodeA.nextEdges.length should be(0)
    nodeB.prevEdges.length should be(0)
  }

  "remove previous edge" should "be correct" in {
    val nodeA = new Node("A")
    val nodeB = new Node("B")
    val nodeC = new Node("C")
    nodeA -> nodeC
    nodeA -> nodeB
    nodeB -> nodeC
    nodeC.removePrevEdges()
    nodeC.prevEdges.length should be(0)
    nodeA.nextEdges.length should be(1)
    nodeB.nextEdges.length should be(0)
  }

  "keep backward topology" should "be correct" in {
    val input1 = Tensor[Float](2, 2, 3, 3).rand()
    val input2 = Tensor[Float](1, 1, 3, 3).rand()

    def modelDef(): Module[Float] = {
      val input1 = Input[Float]()
      val input2 = Input[Float]()

      val add1 = CAddTable[Float]().inputs(input1, input2)
      val add2 = CAddTable[Float]().inputs(add1, input2)
      val add3 = CAddTable[Float]().inputs(add2, input2)
      val add4 = CAddTable[Float]().inputs(add3, input2)
      Graph[Float](Array(input1, input2), Array(add4))
    }

    val model = modelDef()
    val output = model.forward(T(input1, input2))
    val gradInput = model.backward(T(input1, input2), output)
  }
}
