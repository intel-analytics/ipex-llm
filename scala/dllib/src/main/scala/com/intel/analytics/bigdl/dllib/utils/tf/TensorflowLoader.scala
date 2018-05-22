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

import java.io.{DataInputStream, InputStream, FileReader => JFileReader}
import java.nio.ByteOrder
import java.util
import java.util.List
import java.util.{HashMap => JHashMap}

import com.google.protobuf.{CodedInputStream, TextFormat}
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Graph
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.tf.AssignGrad
import com.intel.analytics.bigdl.python.api.{JTensor, PythonBigDL, PythonBigDLUtils}
import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.nn.tf.{SwitchControlNode, SwitchOps}
import com.intel.analytics.bigdl.utils.tf.TensorflowToBigDL._
import com.intel.analytics.bigdl.utils.tf.loaders.TensorflowOpsLoader
import org.tensorflow.framework.{GraphDef, NodeDef}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.collection.JavaConverters._

object TensorflowLoader{
  /**
   * Load tensorflow model from a prototxt file
   * @param graphPrototxt where is the tensorflow protobuf file
   * @param inputs input node names, where feed in the data. You can feed data into an internal
   *               node. If the internal node has multiple dependency, you can use name:n to specify
   *               which dependency you choose to feed
   * @param outputs output node names
   * @param byteOrder file byteOrder
   * @param generatedBackward if generate backward graph
   * @return
   */
  def load[T: ClassTag](graphPrototxt: String, inputs: Seq[String], outputs: Seq[String],
        byteOrder: ByteOrder, binFile: Option[String] = None,
        generatedBackward: Boolean = true)(
    implicit ev: TensorNumeric[T]): Module[T] = {
    // Get node list
    val nodeList = parse(graphPrototxt)

    // Input name remove the port
    val realInputNames = inputs.map(i => if (i.split(":").length == 2) i.split(":")(0) else i)
      .distinct

    // Construct tf node graph
    val (tfGraph, newInputMap, _) =
      buildTFGraph(nodeList, outputs, (node: NodeDef) => realInputNames.contains(node.getName),
        Some(getInputPorts(inputs)))

    // If you choose an internal node with multiple inputs, extra placeholder will be insert into
    // the model
    // Keep the order with the inputs list
    val newInputs = ArrayBuffer[String]()
    realInputNames.foreach(i => {
      if (newInputMap.isDefinedAt(i)) {
        newInputMap(i).foreach(n => newInputs.append(n))
      }
    })
    // Try to load variables
    val context = binFile.map(loadBinFiles(_))

    // Build BigDL model from the tf node graph
    buildBigDLModel(tfGraph, newInputs, outputs, byteOrder, graphPrototxt,
      context, generatedBackward)
  }

  def checkpoints[T: ClassTag](graphFile: String, binFile: String, byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): Session[T] = {
    // Get node list
    val nodeList = parse(graphFile)

    new BigDLSessionImpl[T](nodeList.asScala, loadBinFiles(binFile), byteOrder)
  }

  /**
   * Get the input ports if user specify. It will also check the input name list.
   * @param inputs
   * @return
   */
  private def getInputPorts(inputs: Seq[String]): mutable.Map[String, ArrayBuffer[Int]] = {
    require(inputs.distinct.length == inputs.length,
      "input should not contain duplicated names")
    val inputPorts = inputs.filter(_.split(":").length == 2)
    val result = mutable.HashMap[String, ArrayBuffer[Int]]()
    inputPorts.foreach(s => {
      val name = s.split(":")(0)
      val pos = s.split(":")(1)
      require(!inputs.contains(name), "You should not specify node name and node name " +
        "with port at same time")
      if (!result.isDefinedAt(name)) {
        result(name) = ArrayBuffer[Int]()
      }
      result(name).append(pos.toInt)
    })
    result
  }

  /**
   * Parse a tensorflow model protobuf binary file, read a list of op nodes from it
   * @param graphProtoTxt where is the tf protobuf file
   * @return
   */
  private[bigdl] def parse(graphProtoTxt: String) : List[NodeDef] = {
    var fr: FileReader = null
    var in: InputStream = null
    try {
      fr = FileReader(graphProtoTxt)
      in = fr.open()
      val reader = CodedInputStream.newInstance(new DataInputStream(in))
      reader.setSizeLimit(0x7fffffff)

      val graph = GraphDef.parseFrom(reader)
      graph.getNodeList
    } finally {
      if (fr != null) fr.close()
      if (in != null) in.close()
    }

  }

  private[bigdl] def saveBinFile[T: ClassTag](file: String, context: Context[T])
    (implicit ev: TensorNumeric[T]): Unit = {
    val save = new JHashMap[String, JTensor]()
    context.tensorNames().foreach(n => {
      val tensor = context(n)._1
      val saveTensor = ev.getType() match {
        case FloatType => new JTensor(tensor.asInstanceOf[Tensor[Float]].storage().array(),
          tensor.size(), "float")
        case DoubleType => new JTensor(tensor.asInstanceOf[Tensor[Double]].storage().array()
          .map(_.toFloat), tensor.size(), "double")
        case t => throw new NotImplementedError(s"$t is not supported")
      }
      save.put(n, saveTensor)
    })
    File.save(save, file, true)
  }

  private def loadBinFiles[T: ClassTag](file: String)(implicit ev: TensorNumeric[T]): Context[T] = {
    val m = File.load(file).asInstanceOf[JHashMap[String, JTensor]].asScala
    val map = new mutable.HashMap[String, (Tensor[T], Tensor[T], Option[Seq[(Int, Int)]])]()
    for (k <- m.keys) {
      val tensor = ev.getType() match {
        case FloatType => PythonBigDLUtils.toTensor(m(k), "float")
        case DoubleType => PythonBigDLUtils.toTensor(m(k), "double")
        case t => throw new NotImplementedError(s"$t is not supported")
      }

      map(k) = (tensor, tensor.clone(), None)
    }
    new Context[T](map)
  }

  /**
   * Parse a tensorflow model protobuf text file, read a list of op nodes from it
   * @param graphProtoTxt where is the tf protobuf file
   * @return
   */
  private[bigdl] def parseTxt(graphProtoTxt: String) : List[NodeDef] = {
    val f = new java.io.File(graphProtoTxt)
    require(f.exists(), graphProtoTxt + " does not exists")

    val reader = new JFileReader(f)

    val graph = GraphDef.newBuilder()

    TextFormat.merge(reader, graph)

    graph.build().getNodeList
  }

  /**
   * Build tf ops graph from a given node list. The graph output is the outputs.
   *
   * @param nodes Tensorflow NodeDefs
   * @param outputs output node names
   * @param isInput check if a node is input
   * @param inputPorts if user want to use a part of the node input as model input
   * @return
   */
  private[bigdl] def buildTFGraph(
    nodes : List[NodeDef], outputs: Seq[String],
    isInput: (NodeDef) => Boolean = (_: NodeDef) => false,
    inputPorts: Option[mutable.Map[String, ArrayBuffer[Int]]] = None
  ): (DirectedGraph[NodeDef], mutable.HashMap[String, ArrayBuffer[String]], Seq[String]) = {
    val name2Node = nodes.asScala.map(n => n.getName -> new Node(n)).toMap

    // Build graph
    val outputNodes = if (outputs == null) {
      name2Node.valuesIterator.filter(_.nextNodes.isEmpty).toArray
    } else {
      val results = name2Node.valuesIterator.toArray.filter(n =>
        outputs.contains(n.element.getName))
      require(results.length == outputs.length, "Invalid outputNode names")
      results
    }
    val (inputs, originInputs) = connect(outputNodes, name2Node, isInput, inputPorts)

    val dummyOutput = new Node[NodeDef](null)
    outputNodes.foreach(_ -> dummyOutput)
    (dummyOutput.graph(reverse = true), inputs, originInputs)
  }

  /**
   * Build a graph from the output. The build process will stop when met node without inputs
   * or specified input node
   */
  private[bigdl] def connect(
    nodes: Seq[Node[NodeDef]],
    name2Node: Map[String, Node[NodeDef]],
    isInput: (NodeDef) => Boolean,
    inputPorts: Option[mutable.Map[String, ArrayBuffer[Int]]]
  ): (mutable.HashMap[String, ArrayBuffer[String]], Seq[String]) = {

    var inputCounter = 0
    var depCounter = 0
    val queue = new mutable.Queue[Node[NodeDef]]()
    val visited = mutable.Set[Node[NodeDef]]()
    val newInputs = new mutable.HashMap[String, ArrayBuffer[String]]()
    val originInputs = new mutable.ArrayBuffer[String]()

    // Do a BFS to connect the nodes
    queue.enqueue(nodes: _*)
    while(queue.nonEmpty) {
      val node = queue.dequeue()
      if (!visited(node)) {
        visited += node
        if (!isInput(node.element) && !node.element.getInputList.isEmpty) {
          // continue to traverse
          node.element.getInputList.asScala.foreach { preNodeName =>
            depCounter = pushPreNode(preNodeName, name2Node, depCounter, node, queue)
          }
        } else {
          if (newInputs.get(node.element.getName).isEmpty) {
            newInputs(node.element.getName) = new ArrayBuffer[String]()
          }
          if (isInput(node.element) && node.element.getOp != "Placeholder") {
            // if the predefined input node is not a Placeholder, add one to match the Input node
            val inputNum = getInputNumber(node.element)
            if (inputNum == 0) {
              require(!inputPorts.isDefined ||
                !inputPorts.get.isDefinedAt(node.element.getName),
                  s"node ${node.element.getName} has no input")
              newInputs(node.element.getName).append(node.element.getName)
            } else {
              if (inputPorts.isDefined &&
                inputPorts.get.isDefinedAt(node.element.getName)) {
                val selectInputs = inputPorts.get(node.element.getName)
                selectInputs.foreach(i => require(i < inputNum && i >= 0,
                  s"invalid input port $i at ${node.element.getName}, it should between 0 and" +
                    s" ${inputNum - 1}"))
                var i = 0
                while (i < inputNum) {
                  if (selectInputs.contains(i)) {
                    val name = s"input$inputCounter"
                    val placeholder = NodeDef.newBuilder()
                      .setName(name)
                      .setOp("Placeholder").build()
                    inputCounter = inputCounter + 1
                    val n = Node(placeholder)
                    n -> node
                    newInputs(node.element.getName).append(name)
                  } else {
                    val preNodeName = node.element.getInputList.asScala.apply(i)
                    depCounter = pushPreNode(preNodeName, name2Node, depCounter, node, queue)
                  }
                  i = i + 1
                }
              } else {
                val name = s"input$inputCounter"
                val placeholder = NodeDef.newBuilder()
                  .setName(name)
                  .setOp("Placeholder").build()
                inputCounter = inputCounter + 1
                val n = Node(placeholder)
                n -> node
                newInputs(node.element.getName).append(name)
              }
            }
            originInputs += node.element.getName
          } else if (node.element.getOp == "Placeholder") {
            newInputs(node.element.getName).append(node.element.getName)
            originInputs += node.element.getName
          }
        }
      }
    }
    (newInputs, originInputs)
  }

  private def pushPreNode(name: String, name2Node: Map[String, Node[NodeDef]],
    depCounter: Int, node: Node[NodeDef], queue: mutable.Queue[Node[NodeDef]]): Int = {
    // It is tricky here, remove the first char in the name of control dep node
    var realName = name
    var controlDep = false
    var channel = 0
    var _depCounter = depCounter

    if (realName.charAt(0) == '^') {
      realName = realName.substring(1)
      controlDep = true
    }
    if (realName.split(":").length > 1) {
      val pair = realName.split(":")
      realName = pair(0)
      channel = pair(1).toInt
    }

    val preNode = name2Node(realName)

    val curNode = if (controlDep) {
      val dependencyNode = Node(NodeDef.newBuilder()
        .setOp("DependencyNode")
        .addInput(preNode.element.getName)
        .setName(s"depends_on_${preNode.element.getName}$depCounter")
        .build())
      _depCounter += 1
      dependencyNode -> node
      dependencyNode
    } else {
      node
    }

    preNode.add(curNode, Edge(channel + 1))
    queue.enqueue(preNode)
    _depCounter
  }

  private def getInputNumber(nodeDef: NodeDef): Int = {
    import scala.collection.JavaConverters._
    nodeDef.getOp match {
      case "QueueDequeueV2" => nodeDef.getAttrOrThrow("component_types").getList.getTypeCount
      case "QueueDequeueManyV2" => nodeDef.getAttrOrThrow("component_types").getList.getTypeCount
      case _ => nodeDef.getInputList.asScala.filterNot(_.charAt(0) == '^').length
    }
  }

  private[bigdl] def buildBigDLModel[T: ClassTag](
      tfGraph: DirectedGraph[NodeDef],
      inputs: Seq[String],
      outputs: Seq[String],
      byteOrder: ByteOrder,
      graphPrototxt: String,
      ctx: Option[Context[T]] = None,
      generatedBackward: Boolean = true
    )(implicit ev: TensorNumeric[T]): Module[T] = {
    import scala.collection.JavaConverters._

    // Map from tensorflow node to the converted BigDL node
    val convertedNode = new mutable.HashMap[Node[NodeDef],
      Node[AbstractModule[Activity, Activity, T]]]()
    val nameToNode =
      new mutable.HashMap[String, Node[AbstractModule[Activity, Activity, T]]]()

    val moduleToInputNodes =
      new mutable.HashMap[Node[AbstractModule[Activity, Activity, T]], Seq[Node[NodeDef]]]()
    val moduleToAllNodes =
      new mutable.HashMap[Node[AbstractModule[Activity, Activity, T]], Set[Node[NodeDef]]]()
    val context = ctx.getOrElse(new Context[T])

    // BFS to keep the input order same
    tfGraph.BFS.foreach(n => {
      if (n.element == null) {
        // Dummy node, skip
      } else if (convertedNode.get(n).isDefined) {
        // converted node, skip
      } else {
        val errorMsg =
          s"""
            | Cannot convert the given tensorflow operation graph to BigDL model. The convert fails
            | at node ${n.element.getName}. Operation type is ${n.element.getOp}
          """.stripMargin

        val (module, nodes, inputNodes) =
          extract[T](n.graph(reverse = true), context, byteOrder).getOrElse({
            try {
              val cls = Class.forName("com.intel.analytics.bigdl.utils.tf.loaders." +
                n.element.getOp)
              val builder = cls.getConstructors()(0).newInstance().asInstanceOf[TensorflowOpsLoader]
              (builder.build[T](n.element, byteOrder, context), Seq(n).asJava, Seq(n))
            } catch {
              case e: Throwable =>
                throw new UnsupportedOperationException(errorMsg, e)
            }
          })

        // set name
        if (nodes.size() == 1) {
          // Use tf operation name if one to one map
          module.setName(removeColon(nodes.get(0).element.getName()))
        } else {
          // Many to one map
          val name = removeColon(findCommonPrefix(nodes.asScala.map(_.element.getName)))
          if (name == "") {
            // Use a name combine nodes
            module.setName(s"[${nodes.asScala.map(_.element.getName).map(_.replaceAll("/", "\\\\"))
              .map(removeColon(_)).mkString(", ")}]")
          } else {
            // Use the common name
            module.setName(name + "/" + module.getName())
          }
        }
        val node = module match {
          case _: SwitchOps[_] => new SwitchControlNode(module)
          case _ => Node(module)
        }

        nodes.asScala.foreach(m => {
          convertedNode(m) = node
          nameToNode(m.element.getName) = node
        })

        moduleToInputNodes(node) = inputNodes
        moduleToAllNodes(node) = nodes.asScala.toSet

      }
    })

    /**
     * Go through all tensorflow nodes
     * @param outputModuleNode
     */
    def connect(outputModuleNode: Seq[Node[AbstractModule[Activity, Activity, T]]]) = {
      val queue = new mutable.Queue[Node[AbstractModule[Activity, Activity, T]]]()
      val visited = mutable.Set[Node[AbstractModule[Activity, Activity, T]]]()
      queue.enqueue(outputModuleNode: _*)

      while (queue.nonEmpty) {
        val currNode = queue.dequeue()
        if (!visited(currNode)) {
          visited += currNode
          val inputNodes = moduleToInputNodes(currNode)
          val allNodes = moduleToAllNodes(currNode)
          val inputModuleNodes = inputNodes.flatMap(_.prevNodesAndEdges)
            .filterNot(n => context.containsTensor(n._1.element.getName) &&
              n._1.element.getOp() != "VariableV2")
            .filterNot(n => allNodes(n._1))
            .map(n => (convertedNode(n._1), n._2.newInstance())).filter(n => n._1 != currNode)
          inputModuleNodes.foreach(n => n._1.add(currNode, n._2))
          queue.enqueue(inputModuleNodes.map(_._1): _*)
        }
      }
    }

    val outputModules = tfGraph.source.prevNodes.map(_.element.getName).map(nameToNode)

    connect(outputModules)

    val inputNodes = inputs
      .map(n => nameToNode.getOrElse(n, throw new IllegalArgumentException(s"Can't find node $n")))
    val outputNodes = outputs
      .map(n => nameToNode.getOrElse(n, throw new IllegalArgumentException(s"Can't find node $n")))


    val weights = ArrayBuffer[Tensor[T]]()
    val gradients = ArrayBuffer[Tensor[T]]()
    for ((weight, grad, _) <- context.tensors) {
      weights += weight
      gradients += grad
    }

    // Append assign nodes
    val adjustOutputs = if (context.assignGrads.isDefined) {
      outputNodes.map(n => {
        val matchNode = context.assignGrads.get.filter(_._2 == n.element.getName())
        require(matchNode.size <= 1, "Invalid gradients output")
        if (matchNode.size == 1) {
          new AssignGrad[T](context(matchNode.head._1)._2).inputs(n)
        } else {
          n
        }
      })
    } else {
      outputNodes
    }

    Graph.dynamic(inputNodes.toArray, adjustOutputs.toArray,
      Some((weights.toArray, gradients.toArray)),
      generatedBackward)
  }

  /**
   * Extract one module and the corresponding node list from the given graph
   * @param graph
   * @return
   */
  private[bigdl] def extract[T: ClassTag](graph: DirectedGraph[NodeDef],
      context: Context[T], byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): Option[(
    AbstractModule[Activity, Activity, T],
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
    inputs.append(graph.source)

    pattern.BFS.foreach(patternNode => {
      if (patternNode.element != N_INPUT_PLACEHOLDER && patternNode.element != INPUT_PLACEHOLDER) {
        // Normal operation node
        if (patternToGraph.get(patternNode).isEmpty) return (util.Collections.emptyList(), Seq())

        val graphNode = patternToGraph(patternNode)
        // Operation type should match
        if (patternNode.element != graphNode.element.getOp) return (
          util.Collections.emptyList(), Seq())

        // Prev nodes number should be same except for the Ninput case
        val patternInputLength = patternNode.prevNodes.length
        val graphInputLength = graphNode.prevNodes.
          filterNot(_.element.getOp == "DependencyNode").length
        if (patternInputLength != graphInputLength &&
          !patternNode.prevNodes.exists(_.element == N_INPUT_PLACEHOLDER)) {
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

  private def findCommonPrefix(data: Seq[String]): String = {
    if (data.length == 0) return ""
    var shortest = data(0).length
    data.foreach(s => if (s.length < shortest) shortest = s.length)
    var prefix = ""
    var i = 0
    while(i < shortest) {
      var c = data(0).charAt(i)
      data.foreach(s => if (c != s.charAt(i)) return removeLast(prefix))
      prefix += c
      i += 1
    }

    return removeLast(prefix)
  }

  private def removeLast(s: String): String = {
    if (s.length == 0) return s
    if (s.charAt(s.length - 1) == '/') s.substring(0, s.length - 1) else s
  }

  private def removeColon(s: String): String = {
    s.replaceAll(":", "")
  }
}
