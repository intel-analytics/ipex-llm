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

import java.util

import scala.collection.JavaConverters._
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.nn.tf.{ControlDependency, WithoutInput}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer.{ContainerSerializable, DataConverter, ModuleData, ModuleSerializer}
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.utils.tf.{BigDLToTensorflow, Tensorflow, TensorflowSaver}
import serialization.Bigdl.{AttrValue, BigDLModule}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import com.intel.analytics.bigdl.visualization.tensorboard.{FileWriter => TFFileWriter}
import org.tensorflow.framework.GraphDef

/**
 * A graph container. Each node can have multiple inputs. The output of the node should be a tensor.
 * The output tensor can be connected to multiple nodes. So the module in each node can have a
 * tensor or table input, and should have a tensor output.
 *
 * The graph container can have multiple inputs and multiple outputs. If there's one input, the
 * input data fed to the graph module should be a tensor. If there're multiple inputs, the input
 * data fed to the graph module should be a table, which is actually an sequence of tensor. The
 * order of the input tensors should be same with the order of the input nodes. This is also
 * applied to the gradient from the module in the back propagation.
 *
 * All of the input modules must accept a tensor input. If your input module accept multiple
 * tensors as input, you should add some Input module before it as input nodes and connect the
 * output of the Input modules to that module.
 *
 * If there's one output, the module output is a tensor. If there're multiple outputs, the module
 * output is a table, which is actually an sequence of tensor. The order of the output tensors is
 * same with the order of the output modules. This is also applied to the gradient passed to the
 * module in the back propagation.
 *
 * All inputs should be able to connect to outputs through some paths in the graph. It is
 * allowed that some successors of the inputs node are not connect to outputs. If so, these nodes
 * will be excluded in the computation.
 *
 * @param inputs input nodes
 * @param outputs output nodes
 * @param variables an Array of tensor containing all the weights and biases of this graph,
 *                used when different nodes of this graph may share the same weight or bias.
 * @tparam T Numeric type. Only support float/double now
 */
@SerialVersionUID(- 2896121321564992779L)
class Graph[T: ClassTag](val inputs : Seq[ModuleNode[T]],
    private val outputs : Seq[ModuleNode[T]],
    private val variables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None)
    (implicit ev: TensorNumeric[T])
    extends Container[Activity, Activity, T]{

  type absModule = AbstractModule[_ <: Activity, _ <: Activity, T]

  override def updateOutput(input: Activity): Activity = {
    var i = 0
    while(i < forwardExecutions.length) {
      val node = forwardExecutions(i)
      val nodeInput = if (node.prevNodes.isEmpty && !node.element.isInstanceOf[WithoutInput]) {
        inputData(node, input)
      } else {
        val prevActivities = node.prevNodesAndEdges
          .filterNot(n => n._1.element.isInstanceOf[ControlDependency[T]])
          .map(n => {
          n._2.fromIndex match {
            case Some(i) => n._1.element.output.toTable.apply[Activity](i)
            case None => n._1.element.output
          }
        })
        if (prevActivities.length == 1) {
          prevActivities.head
        } else {
          seqToTable(prevActivities)
        }
      }
      node.element.forward(nodeInput)
      inputsBP.put(node.element.getName(), nodeInput)
      i += 1
    }

    output = dummyOutput.element.output
    output
  }

  override def backward(input: Activity, gradOutput: Activity): Activity = {
    val before = System.nanoTime()
    dummyOutputGrad.element.gradInput = gradOutput

    var i = 0
    while (i < backwardExecutions.length) {
      val curNode = backwardExecutions(i)
      var curGradOutput : Activity = null

      curNode.nextNodesAndEdges.filterNot(n => n._1.element.isInstanceOf[ControlDependency[T]])
        .foreach(n => {
        val otherActivity = if (n._1.element.gradInput.isTensor || n._1.prevEdges.length == 1) {
          n._1.element.gradInput
        } else {
          val index = n._1.prevEdges.indexOf(n._2) + 1
          n._1.element.gradInput.toTable.apply[Activity](index)
        }

        n._2.fromIndex match {
          case Some(i) =>
            if (curNode.element.output.isTable && curGradOutput == null) {
              curGradOutput = T()
            }
            val curActivity = curGradOutput.toTable.getOrElse[Activity](i, null)
            curGradOutput.toTable(i) = accActivity(curActivity, otherActivity)
          case None =>
            curGradOutput = accActivity(curGradOutput, otherActivity)
        }
      })

      gradOutputBP(i) = curGradOutput
      if (!isStopGradient(curNode.element)) {
        curNode.element.backward(inputsBP.get(curNode.element.getName()), curGradOutput)
      } else {
        curNode.element.accGradParameters(inputsBP.get(curNode.element.getName()), curGradOutput)
      }
      i += 1
    }

    gradInput = if (inputs.length == 1) {
      inputs.head.element.gradInput
    } else {
      seqToTable(inputs.map(n => n.element.gradInput))
    }
    backwardTime = System.nanoTime() - before
    gradInput
  }

  private def calcSumTimesOfAllNodes(timesOfAllNodes: Array[(absModule, Long, Long)])
  : (Long, Long) = {
    var sumForward = 0L
    var sumBackward = 0L
    timesOfAllNodes.foreach(x => {
      sumForward += x._2
      sumBackward += x._3
    })
    (sumForward, sumBackward)
  }

  override def getTimes():
  Array[(AbstractModule[_ <: Activity, _ <: Activity, T], Long, Long)] = {
    val timesOfAllNodes = this.modules.flatMap(_.getTimes()).toArray
    val (sumForward, sumBackward) = calcSumTimesOfAllNodes(timesOfAllNodes)
    timesOfAllNodes ++ Array((this, this.forwardTime - sumForward, this.backwardTime - sumBackward))
  }

  override def resetTimes(): Unit = {
    this.forwardTime = 0L
    this.backwardTime = 0L
    modules.foreach(_.resetTimes())
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    dummyOutputGrad.element.gradInput = gradOutput

    var i = 0
    while (i < backwardExecutions.length) {
      val curNode = backwardExecutions(i)
      var curGradOutput : Activity = null
      if (curNode.element.output.isTable) {
        curGradOutput = T()
      }

      curNode.nextNodesAndEdges
        .filterNot(n => n._1.element.isInstanceOf[ControlDependency[T]])
        .foreach(n => {
        val otherActivity = if (n._1.element.gradInput.isTensor || n._1.prevEdges.length == 1) {
          n._1.element.gradInput
        } else {
          val index = n._1.prevEdges.indexOf(n._2) + 1
          n._1.element.gradInput.toTable.apply[Activity](index)
        }

        n._2.fromIndex match {
          case Some(i) =>
            val curActivity = curGradOutput.toTable.getOrElse[Activity](i, null)
            curGradOutput.toTable(i) = accActivity(curActivity, otherActivity)
          case None =>
            curGradOutput = accActivity(curGradOutput, otherActivity)
        }
      })

      gradOutputBP(i) = curGradOutput

      if (!isStopGradient(curNode.element)) {
        curNode.element.updateGradInput(inputsBP.get(curNode.element.getName()), curGradOutput)
      }
      i += 1
    }

    gradInput = if (inputs.length == 1) {
      inputs.head.element.gradInput
    } else {
      seqToTable(inputs.map(n => n.element.gradInput))
    }
    gradInput
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    var i = 0
    while (i < backwardExecutions.length) {
      val curNode = backwardExecutions(i)
      curNode.element.accGradParameters(inputsBP.get(curNode.element.getName()), gradOutputBP(i))
      i += 1
    }
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    variables match {
      case None => super.parameters()
      case Some((weights, gradients)) => (weights, gradients)
    }
  }

  override def add(module: AbstractModule[_ <: Activity, _ <: Activity, T]): Graph.this.type = {
    throw new IllegalArgumentException("Graph: Please don't use add method in Graph container. " +
      "A graph container should not be changed after it is constructed")
  }

  // Add a dummy output node, to get an one end graph. So the nodes that are not dependent by
  // the outputs will be excluded
  private val dummyOutput = new ModuleNode[T](new Identity[T]())
  // Add a dummy output node for backward graph,
  // dummyOutputGrad has the same function as dummyOutput
  // used to construct a backward graph
  private var dummyOutputGrad: ModuleNode[T] = _
  outputs.foreach(_ -> dummyOutput)

  /**
   * Computing backgraph
   */
  private val backGraph = dummyOutput.graph(reverse = true)
  private var gradGraph: DirectedGraph[AbstractModule[Activity, Activity, T]] = null

  /**
   * Execution plan
   */
  private val forwardExecutions = backGraph.topologySort
    .filterNot(_.element.isInstanceOf[ControlDependency[T]]).reverse
  private var backwardExecutions: Array[Node[AbstractModule[Activity, Activity, T]]] = null

  modules.appendAll(forwardExecutions.filter(n => !n.eq(dummyOutput)).map(_.element))

  /**
   * build is needed when the stopGrad is changed
   */
  private[bigdl] def build(): this.type = {
    val gradGraph = backGraph.cloneGraph()
    dummyOutputGrad = gradGraph.source
    val nodes = gradGraph.DFS
    nodes.filter(x => isStopGradient(x.element)).foreach(_.removePrevEdges())
    backwardExecutions = gradGraph.topologySort.filter(n => !n.eq(dummyOutputGrad))
      .filterNot(_.element.isInstanceOf[ControlDependency[T]])
    clearState()
    this
  }


  private val inputsBP = new util.HashMap[String, Activity]()

  // Check all inputs of the graph should be passed in
  checkRoots
  build

  private val gradOutputBP = new Array[Activity](forwardExecutions.length - 1)

  private def checkRoots: Unit = {
    val roots = forwardExecutions.filter(_.prevNodes.size == 0)
      .filter(node => !node.element.isInstanceOf[WithoutInput])
    require(roots.size == inputs.length,
      s"There're ${inputs.length} inputs, but graph has ${roots.size} roots")
    inputs.foreach(n =>
      require(roots.contains(n), "inputs and graph roots are not match")
    )
  }

  private[nn] def shift[B](data : Array[B], from : Int, to : Int): Array[B] = {
    require(from < data.length && from >= 0, s"invalid from $from array length is ${data.length}")
    require(to < data.length && to >= 0, s"invalid to $to array length is ${data.length}")
    if (from == to) {
      data
    } else if (from < to) {
      var i = from
      while(i < to) {
        val tmp = data(i)
        data(i) = data(i + 1)
        data(i + 1) = tmp
        i += 1
      }
      data
    } else {
      var i = from
      while(i > to) {
        val tmp = data(i)
        data(i) = data(i - 1)
        data(i - 1) = tmp
        i -= 1
      }
      data
    }
  }

  private def seqToTable(inputs: Seq[Activity]) : Table = {
    val t = T()
    var j = 1
    inputs.foreach(tensor => {
      t(j) = tensor
      j += 1
    })
    t
  }

  private def inputData(
      node: Node[AbstractModule[Activity, Activity, T]],
      input: Activity
  ): Activity = {
    if (inputs.length == 1) {
      require(inputs(0).eq(node), "input node is not in the input list")
      input.toTensor
    } else {
      val i = inputs.indexOf(node)
      require(i != -1, "input node is not in the input list")
      input.toTable[Tensor[T]](i + 1)
    }
  }

  /**
   * set an array of layers that match the given ```names``` to be "freezed",
   * i.e. their parameters(weight/bias, if exists) are not changed in training process
   * @param names an array of layer names
   * @return current graph model
   */
  def freeze(names: Array[String]): this.type = {
    names.foreach(name => {
      val layer = this (name)
      require(layer.isDefined, s"cannot find layer match ${name}")
      layer.get.setScaleW(0)
      layer.get.setScaleB(0)
    })
    this
  }

  private var stopGradientLayers: util.HashSet[String] = _

  /**
   * whether stop propagating gradInput back
   * @return
   */
  private def isStopGradient(module: AbstractModule[_ <: Activity, _ <: Activity, T]): Boolean = {
    null != stopGradientLayers && stopGradientLayers.contains(module.getName())
  }

  /**
   * stop the input gradient of layers that match the given ```names```
   * their input gradient are not computed.
   * And they will not contributed to the input gradient computation of
   * layers that depend on them.
   * @param names an array of layer names
   * @return current graph model
   */
  def stopGradient(names: Array[String]): this.type = {
    names.foreach(name => {
      val layer = this (name)
      require(layer.isDefined, s"cannot find layer match ${name}")
      if (stopGradientLayers == null) stopGradientLayers =
        new util.HashSet[String]()
      stopGradientLayers.add(layer.get.getName())
    })
    build()
    this
  }


  override def reset(): Unit = {
    if (null != stopGradientLayers) stopGradientLayers.clear()
    unFreeze()
    build()
  }

  /**
   * get forward executions, the dummy node will be filtered
   * @return
   */
  def getForwardExecutions: Array[Node[AbstractModule[Activity, Activity, T]]] = {
    forwardExecutions.filter(n => !n.eq(dummyOutput))
  }

  @inline
  private def accActivity(activity: Activity, other: Activity): Activity = {
    if (activity == null) {
      other
    } else {
      if (other.isTensor) {
        require(activity.isTensor, "Cannot add a table to a tensor")
        activity.toTensor[T].add(other.toTensor[T])
      } else {
        val actTable = activity.toTable
        val actLen = actTable.length()
        val otherTable = other.toTable
        val otherLen = otherTable.length()
        require(actLen == otherLen, "table length is not equal")
        var i = 1
        while(i <= actLen) {
          require(actTable[Activity](i) != null, "Invalid table element")
          accActivity(actTable[Activity](i), otherTable[Activity](i))
          i += 1
        }
        actTable
      }
    }
  }

  /**
   * Save current model graph to a folder, which can be display in tensorboard by running
   *   tensorboard --logdir logPath
   * @param logPath
   * @return
   */
  def saveGraphTopology(logPath: String): this.type = {
    val writer = new TFFileWriter(logPath)
    val graphBuilder = GraphDef.newBuilder()
    forwardExecutions.map(m => {
      val nodeDef = Tensorflow.bigdlModule(m.element, m.nextNodes.map(_.element.getName()).asJava)
      graphBuilder.addNode(nodeDef)
    })
    writer.addGraphDef(graphBuilder.build())
    writer.close()
    this
  }
}

object Graph extends ContainerSerializable {
  /**
   * Node for graph container. The module should have a tensor/table input while a tensor output
   * @tparam T
   */
  type ModuleNode[T] = Node[AbstractModule[Activity, Activity, T]]

  /**
   * Build multiple inputs, multiple outputs graph container.
   * @param input input node
   * @param output output node
   * @return a graph container
   */
  def apply[T: ClassTag](input : Array[ModuleNode[T]], output : Array[ModuleNode[T]],
      variables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None)
      (implicit ev: TensorNumeric[T]) : Graph[T] = {
    new Graph[T](input, output, variables)
  }

  /**
   * Build a single input, multiple outputs graph container
   * @param input input node
   * @param output output nodes
   * @return a graph container
   */
  def apply[T: ClassTag](input : ModuleNode[T], output : Array[ModuleNode[T]])
    (implicit ev: TensorNumeric[T]) : Graph[T] = {
    new Graph[T](Array(input), output)
  }

  /**
   * Build a multiple inputs, single output graph container
   * @param input input nodes
   * @param output output node
   * @return a graph container
   */
  def apply[T: ClassTag](input : Array[ModuleNode[T]], output : ModuleNode[T])
    (implicit ev: TensorNumeric[T]) : Graph[T] = {
    new Graph[T](input, Array(output))
  }

  /**
   * Build a single input, single output graph container
   * @param input input nodes
   * @param output output nodes
   * @return a graph container
   */
  def apply[T: ClassTag](input : ModuleNode[T], output : ModuleNode[T])
    (implicit ev: TensorNumeric[T]) : Graph[T] = {
    new Graph[T](Array(input), Array(output))
  }

  override def doLoadModule[T: ClassTag](module : BigDLModule)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {

    val subModules = module.getSubModulesList.asScala

    val attributes = module.getAttrMap
    val inputNames = new ArrayBuffer[String]
    val outputNames = new ArrayBuffer[String]
    DataConverter.getAttributeValue(attributes.get("inputNames"))
      .asInstanceOf[Array[String]].map(name => inputNames.append(name))
    DataConverter.getAttributeValue(attributes.get("outputNames"))
      .asInstanceOf[Array[String]].map(name => outputNames.append(name))

    val inputs = new ArrayBuffer[ModuleNode[T]]
    val outputs = new ArrayBuffer[ModuleNode[T]]

    // layer name to layer node mapping
    val layerMap = new mutable.HashMap[String, ModuleNode[T]]()
    subModules.foreach(subModule => {
      val bigDLModule = ModuleSerializer.load(subModule)
      val moduleNode = bigDLModule.module.inputs()
      val preNodes = bigDLModule.pre
      preNodes.foreach(pre => {
        if (layerMap.contains(pre)) {
          layerMap(pre) -> moduleNode
        }
      })
      val nextNodes = bigDLModule.next
      layerMap(bigDLModule.module.getName) = moduleNode
    })

    inputNames.foreach(inputName => inputs.append(layerMap(inputName)))
    outputNames.foreach(outputName => outputs.append(layerMap(outputName)))

    var sharedVariables : Option[(Array[Tensor[T]], Array[Tensor[T]])] = None
    if (attributes.containsKey("sharedWeight") && attributes.containsKey("sharedBias")) {
      val weights = attributes.get("sharedWeight")
      val biases = attributes.get("sharedBias")
      val weightArray = DataConverter.getAttributeValue(weights).asInstanceOf[Array[Tensor[T]]]
      val biasArray = DataConverter.getAttributeValue(biases).asInstanceOf[Array[Tensor[T]]]
      sharedVariables = Some(weightArray, biasArray)
    }
    Graph[T](inputs.toArray, outputs.toArray, sharedVariables)
  }

  override def doSerializeModule[T: ClassTag](module : ModuleData[T],
                                              graphBuilder : BigDLModule.Builder)
                                           (implicit ev: TensorNumeric[T]) : Unit = {

    module.next.foreach(_ => graphBuilder.addAllPreModules(_))
    module.pre.foreach(_ => graphBuilder.addAllNextModules(_))
    val graph = module.module.asInstanceOf[Graph[T]]
    val inputsNames = graph.inputs.map(_.element.getName).toArray
    val outputsNames = graph.outputs.map(_.element.getName).toArray
    graph.getForwardExecutions.foreach(execution => {
      val preNodes = execution.prevNodes.map(_.element.getName)
      val nextNodes = execution.nextNodes.map(_.element.getName)
      val currNode = execution.element
        .asInstanceOf[AbstractModule[Activity, Activity, T]]
      val subModel = ModuleSerializer.serialize(ModuleData(currNode, preNodes, nextNodes))
      graphBuilder.addSubModules(subModel)
    })
    if (graph.variables.isDefined) {
      val (weights, bias) = graph.variables.get
      val weightAttrBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(weightAttrBuilder, weights)
      graphBuilder.putAttr("sharedWeight", weightAttrBuilder.build)

      val biasAttrBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(biasAttrBuilder, bias)
      graphBuilder.putAttr("sharedBias", biasAttrBuilder.build)
    }

    val inputNamesAttrBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(inputNamesAttrBuilder, inputsNames)
    graphBuilder.putAttr("inputNames", inputNamesAttrBuilder.build)

    val outputNamesBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(outputNamesBuilder, outputsNames)
    graphBuilder.putAttr("outputNames", outputNamesBuilder.build)

  }
}
