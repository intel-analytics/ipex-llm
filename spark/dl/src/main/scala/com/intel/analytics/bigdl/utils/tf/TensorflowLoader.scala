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
package com.intel.analytics.bigdl.utils.tf

import java.io.{DataInputStream, FileInputStream}
import java.nio.ByteOrder
import java.util

import org.tensorflow.framework.{GraphDef, NodeDef}
import com.google.protobuf.CodedInputStream
import java.util.List

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Graph
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{DirectedGraph, Node}
import com.intel.analytics.bigdl.utils.tf.TensorflowToBigDL._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object TensorflowLoader{

  type Context[T] = mutable.HashMap[NodeDef, (Tensor[T], Tensor[T])]

  /**
   * Load tensorflow model from a prototxt file
   * @param graphPrototxt where is the tensorflow protobuf file
   * @param inputs input node names
   * @param outputs output node names
   * @param byteOrder file byteOrder
   * @return
   */
  def load[T: ClassTag](graphPrototxt: String, inputs: Seq[String], outputs: Seq[String],
        byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): Module[T] = {
    // Get node list
    val nodeList = parse(graphPrototxt)

    // Construct tf node graph
    val tfGraph = buildTFGraph(nodeList, outputs)

    // Build BigDL model from the tf node graph
    buildBigDLModel(tfGraph, inputs, outputs, byteOrder)
  }

  /**
   * Parse a tensorflow model protobuf file, read a list of op nodes from it
   * @param graphProtoTxt where is the tf protobuf file
   * @return
   */
  private[bigdl] def parse(graphProtoTxt: String) : List[NodeDef] = {
    val f = new java.io.File(graphProtoTxt)
    require(f.exists(), graphProtoTxt + " does not exists")

    val reader = CodedInputStream.newInstance(new DataInputStream(new FileInputStream(f)))
    reader.setSizeLimit(0x7fffffff)

    val graph = GraphDef.parseFrom(reader)
    graph.getNodeList
  }

  /**
   * Build tf ops graph from a given node list
   * @param nodes
   * @param outputNodeNames
   * @return
   */
  private[bigdl] def buildTFGraph(nodes : List[NodeDef], outputNodeNames: Seq[String])
  : DirectedGraph[NodeDef] = {
    import scala.collection.JavaConverters._
    var name2Node = nodes.asScala.map(n => n.getName -> new Node(n)).toMap

    // Process node with multiple tensor output, each tensor is regarded as a node
    nodes.asScala
      .flatMap(_.getInputList.asScala)
      .filter(_.split(TENSOR_SEPARATOR).length > 1)
      .foreach { nameWithChannel =>
        val name = nameWithChannel.split(TENSOR_SEPARATOR).head
        val tfNode = NodeDef.newBuilder(name2Node(name).element)
          .setName(nameWithChannel).build()
        name2Node += nameWithChannel -> new Node(tfNode)
      }

    // Connect nodes
    name2Node.valuesIterator.foreach(n => {
      n.element.getInputList.asScala.foreach{
        input =>
          // It is tricky here, remove the first char in the name of control dep node
          val name = if (input.charAt(0) == '^') input.substring(1) else input
          name2Node(name) -> n
      }
    })

    // Build graph
    val outputNodes = if (outputNodeNames == null) {
      name2Node.valuesIterator.filter(_.nextNodes.length == 0).toArray
    } else {
      val results = name2Node.valuesIterator.toArray.filter(n =>
        outputNodeNames.contains(n.element.getName))
      require(results.length == outputNodeNames.length, "Invalid outputNode names")
      results
    }

    val dummyOutput = new Node[NodeDef](null)
    outputNodes.foreach(_ -> dummyOutput)
    dummyOutput.graph(reverse = true)
  }

  private[bigdl] def buildBigDLModel[T: ClassTag](
      tfGraph: DirectedGraph[NodeDef],
      inputs: Seq[String],
      outputs: Seq[String],
      byteOrder: ByteOrder,
      ctx: Option[Context[T]] = None
    )(implicit ev: TensorNumeric[T]): Module[T] = {
    import scala.collection.JavaConverters._

    // Map from tensorflow node to the converted BigDL node
    val convertedNode = new mutable.HashMap[Node[NodeDef],
      Node[AbstractModule[Activity, Tensor[T], T]]]()
    val nameToNode =
      new mutable.HashMap[String, Node[AbstractModule[Activity, Tensor[T], T]]]()
    val context = ctx.getOrElse(new mutable.HashMap[NodeDef, (Tensor[T], Tensor[T])])

    // BFS to keep the input order same
    tfGraph.BFS.foreach(n => {
      if (n.element == null) {
        // Dummy node, skip
      } else if (convertedNode.get(n).isDefined) {
        // converted node, skip
      } else {
        val (module, nodes, inputNodes) =
          extract[T](n.graph(reverse = true), context, byteOrder).getOrElse(
            throw new UnsupportedOperationException(s"Can not find matched graph \n${n}\n\n" +
              s"Its inputs are\n ${n.prevNodes.mkString("\n")}")
          )

        val node = new Node(module)
        nodes.asScala.foreach(m => {
          convertedNode(m) = node
          nameToNode(m.element.getName) = node
        })

        // These two pieces of code are all necessary
        val nextNodes = n.nextNodes.filter(
          n => n.element != null && convertedNode.contains(n) && !context.contains(n.element)
        ).map(convertedNode(_)).filter(_ != node).toSet
        nextNodes.foreach(node -> _)

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


    val weights = ArrayBuffer[Tensor[T]]()
    val gradients = ArrayBuffer[Tensor[T]]()
    for ((weight, grad) <- context.values) {
      weights += weight
      gradients += grad
    }

    Graph(inputNodes.toArray, outputNodes.toArray, Some((weights.toArray, gradients.toArray)))
  }

  /**
   * Extract one module and the corresponding node list from the given graph
   * @param graph
   * @return
   */
  private[bigdl] def extract[T: ClassTag](graph: DirectedGraph[NodeDef],
      context: Context[T], byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): Option[(
    AbstractModule[Activity, Tensor[T], T],
      List[Node[NodeDef]],
      Seq[Node[NodeDef]]
    )] = {

    var i = 0
    while(i < patterns.length) {
      val (result, inputs) = matchGraph(graph, patterns(i).topology)
      if (result.size != 0) {
        // get model
        return Some(patterns(i).layer(graph, context, byteOrder), result, inputs)
      }
      i += 1
    }
    None
  }

  private def matchGraph(graph: DirectedGraph[NodeDef], pattern: DirectedGraph[String])
      : (List[Node[NodeDef]], Seq[Node[NodeDef]]) = {
    require(graph.reverse && pattern.reverse, "Must pass in reversed graph")
    val patternToGraph = new mutable.HashMap[Node[String], Node[NodeDef]]()
    val inputs = new ArrayBuffer[Node[NodeDef]]()
    patternToGraph(pattern.source) = graph.source

    pattern.BFS.foreach(patternNode => {
      if (patternNode.element != N_INPUT_PLACEHOLDER && patternNode.element != INPUT_PLACEHOLDER) {
        // Normal operation node
        if (patternToGraph.get(patternNode).isEmpty) return (util.Collections.emptyList(), Seq())

        val graphNode = patternToGraph(patternNode)
        // Operation type should match
        if (patternNode.element != graphNode.element.getOp) return (
          util.Collections.emptyList(), Seq())

        // Prev nodes number should be same except for the Ninput case
        if (patternNode.prevNodes.length != graphNode.prevNodes.length &&
          patternNode.prevNodes.filter(_.element == N_INPUT_PLACEHOLDER).length == 0) {
          return (util.Collections.emptyList(), Seq())
        }

        var i = 0
        var direction = 0
        var j = 0
        while (i < patternNode.prevNodes.length) {
          if (patternNode.prevNodes(i).element == N_INPUT_PLACEHOLDER) {
            require(patternNode.prevNodes.count(_.element == N_INPUT_PLACEHOLDER) == 1,
              s"only support one $N_INPUT_PLACEHOLDER ")
            direction = 1
            // skip the left input nodes of graphNode,
            // once we find the placeholder, we start from another side
            if (!inputs.contains(graphNode)) {
              inputs.append(graphNode)
            }
          } else if (patternNode.prevNodes(i).element == INPUT_PLACEHOLDER) {
            // skip input placeholder
            if (!inputs.contains(graphNode)) {
              inputs.append(graphNode)
            }
          } else {
            val posPattern = { if (direction == 0) i else patternNode.prevNodes.length - 1 - j}
            val posGraph = { if (direction == 0) i else graphNode.prevNodes.length - 1 - j}
            val pn = patternNode.prevNodes(posPattern)
            val gn = graphNode.prevNodes(posGraph)
            if (patternToGraph.contains(pn)) {
              if (!patternToGraph(pn).eq(gn)) return (util.Collections.emptyList(), Seq())
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
    return (patternToGraph.valuesIterator.toList.asJava, inputs)
  }
}
