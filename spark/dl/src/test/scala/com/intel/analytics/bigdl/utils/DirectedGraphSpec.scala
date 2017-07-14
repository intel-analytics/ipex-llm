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

import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}

class DirectedGraphSpec extends FlatSpec with Matchers {


  "Graph" should " performs cse corretly" in {
    val fc1 = Linear[Double](4, 2).inputs()
    val fc2 = Linear[Double](4, 2).inputs()
    val fc3 = CAddTable[Double]().inputs(fc1, fc2)
    val fc4 = Identity[Double]().inputs(fc3)
    val fc5 = Identity[Double]().inputs(fc3)

    val sink = new ModuleNode[Double](new Dummy[Double]())
    fc4 -> sink
    fc5 -> sink

    val graph = sink.graph(reverse = true)

    var numOfIdentity = 0
    graph.topologySort.foreach(n => {
      if (n.element.isInstanceOf[Identity[Double]]) {
          numOfIdentity += 1
      }
    })

    graph.cse

    var numOfIdentityCSE = 0
    graph.topologySort.foreach(n => {
      if (n.element.isInstanceOf[Identity[Double]]) {
        numOfIdentityCSE += 1
      }
    })

    numOfIdentity should be (2)
    numOfIdentityCSE should be (1)
  }

  "Node add" should "be correct" in {
    val nodeA = new Node("A")
    val nodeB = new Node("B")
    val test = nodeA -> nodeB
    test should be(nodeB)
    nodeA.prevNodes.length should be(0)
    nodeA.nextNodes.length should be(1)
    nodeA.nextNodes(0) should be(nodeB)
    nodeB.prevNodes.length should be(1)
    nodeB.prevNodes(0) should be(nodeA)
    nodeB.nextNodes.length should be(0)
  }

  "Node add" should "ignore duplicated add" in {
    val nodeA = new Node("A")
    val nodeB = new Node("B")
    val test = nodeA -> nodeB
    nodeA -> nodeB
    test should be(nodeB)
    nodeA.prevNodes.length should be(0)
    nodeA.nextNodes.length should be(1)
    nodeA.nextNodes(0) should be(nodeB)
    nodeB.prevNodes.length should be(1)
    nodeB.prevNodes(0) should be(nodeA)
    nodeB.nextNodes.length should be(0)
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
}
