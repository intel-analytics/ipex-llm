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

import java.io.{DataInputStream, FileInputStream}
import java.util

import org.tensorflow.framework.{GraphDef, NodeDef}
import com.google.protobuf.CodedInputStream
import java.util.{Collections, List}

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Graph
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

case class TFInputOutputNodes(input: Seq[Node[NodeDef]], output: Seq[Node[NodeDef]])

object TensorflowLoader{
  /**
   * Load tensor flow module from protobuf file
   * @param graphPrototxt where is the tf protobuf file
   * @return
   */
  def load(graphPrototxt: String, inputs: Seq[String], outputs: Seq[String]): Module[Float] = {
    val nodeList = parse(graphPrototxt)
    val tfGraph = buildTFGraph(nodeList)
    buildBigDLModel(tfGraph, inputs, outputs)
  }

  private val inputPlaceholder : String = "*"

  private val nInputPlaceholder : String = "..."

  /**
   * Extract one module and the corresponding node list from the given graph
   * @param graph
   * @return
   */
  private[bigdl] def extract(graph: DirectedGraph[NodeDef],
                             context: Context)
  : (Option[AbstractModule[Activity, Tensor[Float], Float]],
    List[Node[NodeDef]],
    TFInputOutputNodes) = {
    var i = 0
    while(i < TFToBigDL.patterns.length) {
      val result = matchGraph(graph, TFToBigDL.patterns(i).topology)
      if (result.size != 0) {
        // get model
        val input = TFToBigDL.patterns(i).getInputNodes(graph)
        val output = TFToBigDL.patterns(i).getOutputNodes(graph)
        val inputOutput = TFInputOutputNodes(input, output)
        return (Some(TFToBigDL.patterns(i).layer(graph, context)), result, inputOutput)
      }
      i += 1
    }
    (None, Collections.emptyList(), null)
  }

  private def matchGraph(graph: DirectedGraph[NodeDef], pattern: DirectedGraph[String])
      : List[Node[NodeDef]] = {
    require(graph.reverse && pattern.reverse, "Must pass in reversed graph")
    val patternToGraph = new mutable.HashMap[Node[String], Node[NodeDef]]()
    patternToGraph(pattern.source) = graph.source

    pattern.BFS.foreach(patternNode => {
      if (patternNode.element != nInputPlaceholder && patternNode.element != inputPlaceholder) {
        // Normal operation node
        if (patternToGraph.get(patternNode).isEmpty) return util.Collections.emptyList()

        val graphNode = patternToGraph.get(patternNode).get
        // Operation type should match
        if (patternNode.element != graphNode.element.getOp) return util.Collections.emptyList()

        // Prev nodes number should be same except for the Ninput case
        if (patternNode.prevNodes.length != graphNode.prevNodes.length &&
          patternNode.prevNodes.filter(_.element == nInputPlaceholder).length == 0) {
          return util.Collections.emptyList()
        }

        var i = 0
        var direction = 0
        var j = 0
        while (i < patternNode.prevNodes.length) {
          if (patternNode.prevNodes(i).element == nInputPlaceholder) {
            require(patternNode.prevNodes.count(_.element == nInputPlaceholder) == 1,
              s"only support one $nInputPlaceholder ")
            direction = 1
            // skip the left input nodes of graphNode,
            // once we find the placeholder, we start from another side
          } else if (patternNode.prevNodes(i).element == inputPlaceholder) {
            // skip input placeholder
          } else {
            val posPattern = { if (direction == 0) i else patternNode.prevNodes.length - 1 - j}
            val posGraph = { if (direction == 0) i else graphNode.prevNodes.length - 1 - j}
            val pn = patternNode.prevNodes(posPattern)
            val gn = graphNode.prevNodes(posGraph)
            if (patternToGraph.keySet.contains(pn)) {
              if (!patternToGraph(pn).eq(gn)) return util.Collections.emptyList()
            } else {
              patternToGraph(pn) = gn
            }
            if (direction == 1) j += 1
          }
          i += 1
        }
      }
    })
    import scala.collection.JavaConverters._
    return patternToGraph.valuesIterator.toList.asJava
  }

  private[bigdl] def buildBigDLModel(
      tfGraph: DirectedGraph[NodeDef],
      inputs: Seq[String],
      outputs: Seq[String]
  ): Module[Float] = {
    import scala.collection.JavaConverters._

    val convertedNode = new mutable.HashMap[Node[NodeDef],
      Node[AbstractModule[Activity, Tensor[Float], Float]]]()
    val nameToNode =
      new mutable.HashMap[String, Node[AbstractModule[Activity, Tensor[Float], Float]]]()

    val context = new Context

    tfGraph.BFS.foreach(n => {
      if (n.element == null) {
        // Dummy node, skip
      } else if (convertedNode.get(n).isDefined) {
        // converted node, skip
      } else {
        val (module, nodes, inputOutput) = extract(n.graph(reverse = true), context)
        require(module.isDefined, s"Can not find matched graph ${n}")
        val node = new Node(module.get)
        nodes.asScala.foreach(m => {
          convertedNode(m) = node
          nameToNode(m.element.getName) = node
        })

        val outputNodes = if (inputOutput.output == null) nodes.asScala else inputOutput.output
        val nextNodes = outputNodes.flatMap(_.nextNodes)
          .filter(n => n.element != null && convertedNode.contains(n)
            && !context.contains(n.element))
          .map(convertedNode(_)).filter(_ != node).toSet
        nextNodes.foreach(node -> _)

        val inputNodes = if (inputOutput.input == null) nodes.asScala else inputOutput.input
        val preNodes = inputNodes.flatMap(_.prevNodes)
          .filter(n => n.element != null && convertedNode.contains(n)
            && !context.contains(n.element))
          .map(convertedNode(_)).filter(_ != node).toSet
        preNodes.foreach(_ -> node)
      }
    })

    val inputNodes = inputs
      .map(n => nameToNode.getOrElse(n, throw new IllegalArgumentException(s"Can't find node $n")))
    val outputNodes = outputs
      .map(n => nameToNode.getOrElse(n, throw new IllegalArgumentException(s"Can't find node $n")))


    val weights = ArrayBuffer[Tensor[Float]]()
    val gradients = ArrayBuffer[Tensor[Float]]()
    for ((weight, grad) <- context.values) {
      weights += weight
      gradients += grad
    }

    Graph(inputNodes.toArray, outputNodes.toArray, Some((weights.toArray, gradients.toArray)))
  }

  /**
   * Build tf ops graph from a given node list
   * @param nodes
   * @return
   */
  private[bigdl] def buildTFGraph(nodes : List[NodeDef]): DirectedGraph[NodeDef] = {
    import scala.collection.JavaConverters._
    var name2Node = nodes.asScala.map(n => n.getName -> new Node(n)).toMap
    nodes.asScala
      .flatMap(_.getInputList.asScala)
      .filter(_.split(":").length > 1)
      .foreach { nameWithChannel =>
        val name = nameWithChannel.split(":").head
        val tfNode = NodeDef.newBuilder(name2Node(name).element)
          .setName(nameWithChannel).build()
        name2Node += nameWithChannel -> new Node(tfNode)
      }
    // Connect nodes
    name2Node.valuesIterator.foreach(n => {
      n.element.getInputList.asScala.foreach(name2Node(_) -> n)
    })
    val outputNodes = name2Node.valuesIterator.filter(_.nextNodes.length == 0)
    val dummyOutput = new Node[NodeDef](null)
    outputNodes.foreach(_ -> dummyOutput)
    dummyOutput.graph(reverse = true)
  }

  /**
   * Parse a tensor flow model protobuf file, read a list of op nodes from it
   * @param graphPrototxt where is the tf protobuf file
   * @return
   */
  private[bigdl] def parse(graphPrototxt: String) : List[NodeDef] = {
    val f = new java.io.File(graphPrototxt)
    val reader = CodedInputStream.newInstance(new DataInputStream(new FileInputStream(f)))
    reader.setSizeLimit(0x7fffffff)
    require(f.exists(), graphPrototxt + " does not exists")

    val graph = GraphDef.parseFrom(reader)
    graph.getNodeList
  }
}