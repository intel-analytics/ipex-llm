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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.tf._
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.utils.serializer._
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.utils.tf.Tensorflow
import com.intel.analytics.bigdl.visualization.tensorboard.{FileWriter => TFFileWriter}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.reflect.runtime.universe
import scala.language.existentials
import scala.collection.JavaConverters._
import org.tensorflow.framework.GraphDef

/**
 * A graph container. The modules in the container are connected as a directed Graph. Each module
 * can output one tensor or multiple tensors(as table). The edges between modules in the graph
 * define how these tensors are passed. For example, if a module outputs two tensors, you can
 * pass these two tensors together to its following module, or pass only one of them
 * to its following module. If a tensor in the module output is connected to multiple modules, in
 * the back propagation, the gradients from multiple connection will be accumulated. If multiple
 * edges point to one module, the tensors from these edges will be stack as a table, then pass to
 * that module. In the back propagation, the gradients will be splited based on how the input
 * tensors stack.
 *
 * The graph container has multiple inputs and multiple outputs. The order of the input tensors
 * should be same with the order of the input nodes when you construct the graph container. In the
 * back propagation, the order of the gradients tensors should be the same with the order of the
 * output nodes.
 *
 * If there's one output, the module output is a tensor. If there're multiple outputs, the module
 * output is a table, which is actually an sequence of tensor. The order of the output tensors is
 * same with the order of the output modules.
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
abstract class Graph[T: ClassTag](
  val inputs : Seq[ModuleNode[T]],
  private[bigdl] val outputs : Seq[ModuleNode[T]],
  private[bigdl] val variables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None
)(implicit ev: TensorNumeric[T])
  extends Container[Activity, Activity, T] with MklInt8Convertible {

  /**
   * For a multi-tensor output module, some output tensors may not contributed to the final forward
   * result. So in the back propagation, the gradient on these positions are missing. And we use
   * zero tensor to populate.
   *
   * @param output
   * @param gradOutput
   */
  protected def addZeroTensorToMissingGradOutput(output: Table, gradOutput: Table): Unit = {
    var i = 0
    while (i < output.length()) {
      if (!gradOutput.contains(i + 1)) {
        val tensor = output[Tensor[T]](i + 1)
        val zero = Tensor(tensor.size())
        gradOutput(i + 1) = zero
      }
      i = i + 1
    }
  }

  private def calcSumTimesOfAllNodes(
    timesOfAllNodes: Array[(AbstractModule[_ <: Activity, _ <: Activity, T], Long, Long)])
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

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    variables match {
      case None => super.parameters()
      case Some((weights, gradients)) => (weights, gradients)
    }
  }

  // todo: expand the graph
  override def toGraph(startNodes: ModuleNode[T]*): Graph[T] = this

  /**
   * Return the corresponding node has the given name. If the given name doesn't match any node,
   * NoSuchElementException will be thrown
   * @param name
   * @return
   */
  def node(name: String): ModuleNode[T] = {
    val matchNodes = forwardNodes.filter(_.element.getName() == name).toArray
    if (matchNodes.length == 0) {
      throw new NoSuchElementException(s"Can not find node with name $name")
    } else {
      return matchNodes.head
    }
  }

  // Add a dummy output node, to get an one end forward graph. So the nodes that are not dependent
  // by the outputs will be excluded
  protected val dummyOutput = new ModuleNode[T](new Identity[T]())
  outputs.foreach(_ -> dummyOutput)
  protected val forwardGraph = dummyOutput.graph(reverse = true)
  protected val forwardNodes = forwardGraph.DFS.toArray

  populateModules()

  // Check all inputs of the graph should be passed in
  checkRoots

  protected def populateModules(): Unit

  // Check if the graph is correct
  private def checkRoots: Unit = {
    def duplicatedNames(names: Seq[String]): mutable.Set[String] = {
      names.sortWith(_ < _)
      val buffer = new mutable.HashSet[String]()
      var i = 1
      while(i < names.length) {
        if (names(i) == names(i - 1)) buffer.add(names(i))
        i += 1
      }
      buffer
    }

    require(forwardNodes.map(_.element.getName()).distinct.length == forwardNodes.length,
      s"the name of node in the graph should be unique, but find duplicated name " +
        s"${duplicatedNames(forwardNodes.map(_.element.getName())).mkString(", ")}")

    val roots = forwardNodes.filter(_.prevNodes.size == 0)
      .filterNot(_.element.isInstanceOf[WithoutInput])
      .filterNot(_.element.isInstanceOf[ControlDependency[_]])

    val realInputs = inputs.filterNot(_.element.isInstanceOf[WithoutInput])
    require(roots.size == realInputs.length, s"There're ${realInputs.length} inputs, " +
      s"but graph has ${roots.size} roots")

    realInputs.foreach(n =>
      require(roots.contains(n), "inputs and graph roots are not match")
    )
  }

  protected var dummyOutputGrad: ModuleNode[T] = _
  protected var backwardGraph: DirectedGraph[AbstractModule[Activity, Activity, T]] = _
  protected var backwardNodes: Array[Node[AbstractModule[Activity, Activity, T]]] = _
  // If the graph will generate gradInput for the input

  private var isGradInputAvailable: Array[Boolean] = _

  /**
   * Generate backward graph and apply the stopGrad
   */
  private[bigdl] def buildBackwardGraph(): this.type = {
    // Clone the forward graph and reverse the edge
    val gradGraph = forwardGraph.cloneGraph(reverseEdge = true)
    dummyOutputGrad = gradGraph.source
    gradGraph.DFS.filter(x => isStopGradient(x.element)).foreach(removeStopNodes(_))
    backwardNodes = gradGraph.DFS
      .filterNot(_.eq(dummyOutputGrad))
      .filterNot(_.element.isInstanceOf[ControlDependency[_]]).toArray

    val inputNames = inputs.map(_.element.getName()).toSet
    val dummyBackwardEnd = Identity().inputs()
    val backwardTargets = backwardNodes
      .filter(n => (n.element.parameters() != null && n.element.parameters()._1.length != 0)
        || inputNames.contains(n.element.getName()))
    backwardTargets.foreach(_ -> dummyBackwardEnd)
    backwardGraph = dummyBackwardEnd.graph(true)

    // Check if gradInput is empty for each input
    isGradInputAvailable = inputs.map(_ => false).toArray
    backwardGraph.DFS.foreach(curNode => {
      inputs.zipWithIndex.map { case (n, i) =>
        if (curNode.element.getName() == n.element.getName() && !isStopGradient(n.element)) {
          isGradInputAvailable(i) = true
        }
      }
    })

    clearState()
    this
  }

  private var stopGradientLayers: util.HashSet[String] = _

  def getStopGradientLayers(): util.HashSet[String] = stopGradientLayers

  /**
   * whether stop propagating gradInput back
   * @return
   */
  protected def isStopGradient(module: AbstractModule[_ <: Activity, _ <: Activity, T]): Boolean = {
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
    if (stopGradientLayers == null) stopGradientLayers = new util.HashSet[String]()

    names.foreach(name => {
      val layer = this (name)
      require(layer.isDefined, s"cannot find layer match ${name}")
      stopGradientLayers.add(layer.get.getName())
    })
    buildBackwardGraph()
    this
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

  private[bigdl] def removeStopNodes(n: Node[_]): Unit = {
    val nodes = n.nextNodes
    n.removeNextEdges()
    nodes.filter(_.prevNodes.length == 0).foreach(removeStopNodes(_))
  }


  protected def getInput(
    node: Node[AbstractModule[Activity, Activity, T]],
    input: Activity
  ): Activity = {
    if (inputs.length == 1) {
      require(inputs(0).eq(node), "input node is not in the input list")
      input
    } else {
      val i = inputs.indexOf(node)
      require(i != -1, "input node is not in the input list")
      input.toTable[Tensor[T]](i + 1)
    }
  }

  def findInput(node: ModuleNode[T], input: Activity): Activity = {
    if (node.element.isInstanceOf[WithoutInput]) return null

    val nodeInput = if (node.prevNodes.isEmpty) {
      getInput(node, input)
    } else {
      val prevActivities = node.prevNodesAndEdges
        .filterNot(n => n._1.element.isInstanceOf[ControlDependency[T]])
        .map(n => {
          n._2.fromIndex match {
            case Some(i) =>
              if (n._1.element.output == null || (i == 1 && n._1.element.output.isTensor)) {
                n._1.element.output
              } else {
                n._1.element.output.toTable.apply[Activity](i)
              }
            case None => n._1.element.output
          }
        })
      if (prevActivities.length == 1) {
        prevActivities.head
      } else {
        T.seq(prevActivities)
      }
    }
    nodeInput
  }

  protected def findGradOutput(curNode: ModuleNode[T], gradOutput: Activity): Activity = {
    var curGradOutput : Activity = if (curNode.eq(dummyOutputGrad)) gradOutput else null

    curNode.prevNodesAndEdges.filterNot(n => n._1.element.isInstanceOf[ControlDependency[T]])
      .foreach(n => {
        val otherActivity = if (n._1.element.gradInput.isTensor || n._1.nextEdges.length == 1) {
          n._1.element.gradInput
        } else {
          val index = n._1.nextEdges.indexOf(n._2) + 1
          n._1.element.gradInput.toTable.apply[Activity](index)
        }

        n._2.fromIndex match {
          case Some(i) =>
            if (i == 1 && curNode.element.output.isTensor) {
              curGradOutput = accActivity(curGradOutput, otherActivity)
            } else {
              if (curNode.element.output.isTable && curGradOutput == null) {
                curGradOutput = T()
              }
              val curActivity = curGradOutput.toTable.getOrElse[Activity](i, null)
              curGradOutput.toTable(i) = accActivity(curActivity, otherActivity)
            }
          case None =>
            curGradOutput = accActivity(curGradOutput, otherActivity)
        }
      })

    if (curNode.element.output.isTable) {
      addZeroTensorToMissingGradOutput(curNode.element.output.toTable, curGradOutput.toTable)
    }

    curGradOutput
  }

  protected def fetchModelGradInput(): Activity = {
    if (inputs.length == 1) {
      if (isGradInputAvailable.head) {
        inputs.head.element.gradInput
      } else {
        Activity.emptyGradInput(this.getName())
      }
    } else {
      var i = 0
      T.seq(inputs.zipWithIndex.map{ case(n, i) =>
        if (isGradInputAvailable(i)) {
          n.element.gradInput
        } else {
          Activity.emptyGradInput(this.getName())
        }
      })
    }
  }

  override def reset(): Unit = {
    if (null != stopGradientLayers) stopGradientLayers.clear()
    unFreeze()
    buildBackwardGraph()
  }

  /**
   * Get forward executions, the dummy node will be filtered.
   *
   * This method will output an unsorted executions.
   * @return
   */
  def getForwardExecutions(): Array[Node[AbstractModule[Activity, Activity, T]]] = {
    forwardNodes.filterNot(_.eq(dummyOutput))
  }

  /**
   * Get forward executions, the dummy nodes and control dependency nodes will be filtered.
   *
   * This method will output a sorted executions. If the graph contains loop, it will throw an
   * exception
   * @return
   */
  def getSortedForwardExecutions(): Array[ModuleNode[T]] = {
    forwardGraph.topologySort
      // todo: convert control dep node to edge
      .filterNot(_.element.isInstanceOf[ControlDependency[T]]).reverse
      .filter(n => !n.eq(dummyOutput))
  }

  @inline
  protected def accActivity(activity: Activity, other: Activity): Activity = {
    if (activity == null) {
      other
    } else {
      if (other.isTensor) {
        require(activity.isTensor, "Cannot add a table to a tensor")
        activity.toTensor[T].add(other.toTensor[T])
      } else {
        // if 'activity' and 'other' are both table, we need to merge 'other' to 'activity'
        // if 'other' and 'activity' both contains the index, update 'activity' by sum
        // if 'other' contains the index while 'activity' does not,
        // just insert the corresponding tensor of 'other' to 'activity'
        val actTable = activity.toTable
        val otherTable = other.toTable
        otherTable.keySet.foreach(index => {
          if (actTable.contains(index)) {
            accActivity(actTable[Activity](index), otherTable[Activity](index))
          } else {
            actTable.insert(index.asInstanceOf[Int], otherTable(index))
          }
        })
        actTable
      }
    }
  }

  /**
   * Save current model graph to a folder, which can be display in tensorboard by running
   *   tensorboard --logdir logPath
   * @param logPath
   * @param backward Draw backward graph instead of forward
   * @return
   */
  def saveGraphTopology(logPath: String, backward: Boolean = false): this.type = {
    val writer = new TFFileWriter(logPath)
    val graphBuilder = GraphDef.newBuilder()
    val nodes = if (backward) {
      backwardNodes.filter(n => !n.eq(dummyOutputGrad))
    } else {
      forwardNodes.filter(n => !n.eq(dummyOutput))
    }
    nodes.map(m => {
      val nodeDef = Tensorflow.bigdlModule(m.element, m.prevNodes.map(_.element.getName()).asJava)
      graphBuilder.addNode(nodeDef)
    })

    writer.addGraphDef(graphBuilder.build())
    writer.close()
    this
  }

  /**
   * Clear the original module and reset with module in the graph
   */
  def resetModules(): Unit = {
    modules.clear()
    modules.appendAll(forwardGraph.DFS.toArray
      .filterNot(_.element.isInstanceOf[ControlDependency[T]])
      .filter(n => !n.eq(dummyOutput)).map(_.element)
      // Some tests compare the paramerters between sequential and graph,add a reverse makes
      // it's eaiser to compare
      .reverse
    )
  }
}

object Graph extends GraphSerializable {
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
  def apply[T: ClassTag](
    input: Array[ModuleNode[T]],
    output: Array[ModuleNode[T]],
    variables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None
  )(implicit ev: TensorNumeric[T]): Graph[T] = {
    new StaticGraph[T](input, output, variables)
  }

  def apply[T: ClassTag](preprocessor: Module[T], trainable: Module[T])
    (implicit ev: TensorNumeric[T]): Graph[T] = {
    val preprocessorNode = preprocessor.inputs()
    val stopGradients = Identity[T]().inputs(preprocessorNode)
    val trainableNode = trainable.inputs(stopGradients)
    val graph = apply[T](preprocessorNode, trainableNode)
    graph.stopGradient(Array(stopGradients.element.getName()))
    graph
  }

  private[bigdl] def dynamic[T: ClassTag](
    input : Array[ModuleNode[T]],
    output : Array[ModuleNode[T]],
    variables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None,
    generateBackward: Boolean = true)(implicit ev: TensorNumeric[T]): Graph[T] = {
    new DynamicGraph[T](input, output, variables, generateBackward)
  }

  /**
   * Build a single input, multiple outputs graph container
   * @param input input node
   * @param output output nodes
   * @return a graph container
   */
  def apply[T: ClassTag](input: ModuleNode[T], output: Array[ModuleNode[T]])
    (implicit ev: TensorNumeric[T]): Graph[T] = {
    new StaticGraph[T](Seq(input), output)
  }

  private[bigdl] def dynamic[T: ClassTag](input : ModuleNode[T], output : Array[ModuleNode[T]])
    (implicit ev: TensorNumeric[T]) : Graph[T] = {
    new DynamicGraph[T](Array(input), output, None, true)
  }

  /**
   * Build a multiple inputs, single output graph container
   * @param input input nodes
   * @param output output node
   * @return a graph container
   */
  def apply[T: ClassTag](input: Array[ModuleNode[T]], output: ModuleNode[T])
    (implicit ev: TensorNumeric[T]): Graph[T] = {
    new StaticGraph[T](input, Seq(output))
  }

  private[bigdl] def dynamic[T: ClassTag](input : Array[ModuleNode[T]], output : ModuleNode[T])
    (implicit ev: TensorNumeric[T]) : Graph[T] = {
    new DynamicGraph[T](input, Array(output), None, true)
  }

  /**
   * Build a single input, single output graph container
   * @param input input nodes
   * @param output output nodes
   * @return a graph container
   */
  def apply[T: ClassTag](input: ModuleNode[T], output: ModuleNode[T])
    (implicit ev: TensorNumeric[T]): Graph[T] = {
    new StaticGraph[T](Seq(input), Seq(output))
  }

  private[bigdl] def dynamic[T: ClassTag](input : ModuleNode[T], output : ModuleNode[T])
    (implicit ev: TensorNumeric[T]) : Graph[T] = {
    new DynamicGraph[T](Array(input), Array(output), None, true)
  }
}

trait GraphSerializable extends ContainerSerializable {

  private[bigdl] def prepareLoadModule[T: ClassTag](context: DeserializeContext)
                                                   (implicit ev: TensorNumeric[T]) = {

    val module = context.bigdlModule
    val subModules = module.getSubModulesList.asScala

    val attributes = module.getAttrMap
    val inputNames = new ArrayBuffer[String]
    val outputNames = new ArrayBuffer[String]
    DataConverter.getAttributeValue(context, attributes.get("inputNames"))
      .asInstanceOf[Array[String]].map(name => inputNames.append(name))
    DataConverter.getAttributeValue(context, attributes.get("outputNames"))
      .asInstanceOf[Array[String]].map(name => outputNames.append(name))

    val inputs = new ArrayBuffer[ModuleNode[T]]
    val outputs = new ArrayBuffer[ModuleNode[T]]

    // layer name to layer node mapping
    val layerMap = new mutable.HashMap[String, (ModuleNode[T], Seq[String])]()
    subModules.foreach(subModule => {
      val bigDLModule = ModuleSerializer.load(DeserializeContext(subModule,
        context.storages, context.storageType))
      val moduleNode = bigDLModule.module match {
        case controlOps: ControlOps[T] => createControlNode(controlOps)
        case _ => new ModuleNode[T](bigDLModule.module)
      }
      val preNodes = bigDLModule.pre
      layerMap(bigDLModule.module.getName) = (moduleNode, preNodes)
    })

    layerMap.values.foreach(moduleNode => {
      val edges = DataConverter.getAttributeValue(context,
        attributes.get(s"${moduleNode._1.element.getName}_edges")).
        asInstanceOf[mutable.HashMap[String, mutable.HashMap[String, Int]]]
      val edgeMap = edges.get(moduleNode._1.element.getName).get
      moduleNode._2.foreach(pre => {
        if (layerMap.contains(pre)) {
          val edge: Edge = edgeMap.get(pre).get match {
            case -1 => Edge()
            case index: Int => Edge(index)
          }
          layerMap(pre)._1.add(moduleNode._1, edge)
        }
      })
    })

    inputNames.foreach(inputName => inputs.append(layerMap(inputName)._1))
    outputNames.foreach(outputName => outputs.append(layerMap(outputName)._1))

    var sharedVariables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None
    if (attributes.containsKey("sharedWeight") && attributes.containsKey("sharedBias")) {
      val weights = attributes.get("sharedWeight")
      val biases = attributes.get("sharedBias")
      val weightArray = DataConverter.getAttributeValue(context, weights)
        .asInstanceOf[Array[Tensor[T]]]
      val biasArray = DataConverter.getAttributeValue(context, biases)
        .asInstanceOf[Array[Tensor[T]]]
      sharedVariables = Some(weightArray, biasArray)
    }

    val generateBackwardValue = attributes.get("generateBackward")
    (module, inputs, outputs, generateBackwardValue, sharedVariables)
  }

  override def doLoadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {
    val (module, inputs, outputs, generateBackwardValue, sharedVariables) =
      prepareLoadModule(context)
    val attributes = module.getAttrMap
    val graph = if (generateBackwardValue != null) {
      val generateBackward = DataConverter.getAttributeValue(context, generateBackwardValue)
        .asInstanceOf[Boolean]
      Graph.dynamic[T](inputs.toArray, outputs.toArray, sharedVariables, generateBackward)
    } else {
      new StaticGraph[T](inputs, outputs, sharedVariables, false)
    }
    var serializedStopGradientLayers : Array[String] = null
    // this is to keep backward compatible
    if (attributes.containsKey("stopGradientLayers")) {
      val stopGradientLayers = attributes.get("stopGradientLayers")
      serializedStopGradientLayers = DataConverter.
        getAttributeValue(context, stopGradientLayers).asInstanceOf[Array[String]]
    }
    if (serializedStopGradientLayers != null) {
      graph.stopGradient(serializedStopGradientLayers)
    }
    graph
  }

  private def createControlNode[T: ClassTag](controlOps: ControlOps[T]): ModuleNode[T] = {
    controlOps match {
      case switchOps: SwitchOps[T] => new SwitchControlNode[Module[T]](switchOps)
      case mergeOps: MergeOps[T] => new MergeControlNode[Module[T]](mergeOps)
      case _ => new Node[Module[T]](controlOps)
    }
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
      graphBuilder: BigDLModule.Builder)
    (implicit ev: TensorNumeric[T]): Unit = {
    val module = context.moduleData
    module.next.foreach(_ => graphBuilder.addAllPreModules(_))
    module.pre.foreach(_ => graphBuilder.addAllNextModules(_))
    val graph = module.module.asInstanceOf[Graph[T]]
    val inputsNames = graph.inputs.map(_.element.getName).toArray
    val outputsNames = graph.outputs.map(_.element.getName).toArray
    graph.getForwardExecutions.foreach(execution => {

      val edgeMap = new mutable.HashMap[String, mutable.Map[String, Int]]

      val preNodesAndEdges = execution.prevNodesAndEdges
      val preNodes = preNodesAndEdges.map(_._1.element.getName)
      val nextNodes = preNodesAndEdges.map(_._1.element.getName)
      val currNode = execution.element
        .asInstanceOf[AbstractModule[Activity, Activity, T]]
      val subModel = ModuleSerializer.serialize(SerializeContext(
        ModuleData(currNode, preNodes, nextNodes), context.storages, context.storageType))
      // add edges
      val preNodeEdges = new mutable.HashMap[String, Int]()

      preNodesAndEdges.foreach(pre => {
        val preNodeName = pre._1.element.getName
        val preEdgeIndex = pre._2.fromIndex match {
          case Some(i) => i
          case None => -1
        }
        preNodeEdges(preNodeName) = preEdgeIndex
      })
      edgeMap(execution.element.getName) = preNodeEdges
      val attriBulder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context, attriBulder, edgeMap)

      graphBuilder.putAttr(s"${execution.element.getName}_edges", attriBulder.build)
      graphBuilder.addSubModules(subModel.bigDLModule)
    })


    if (graph.variables.isDefined) {
      val (weights, bias) = graph.variables.get
      val weightAttrBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context, weightAttrBuilder, weights,
        universe.typeOf[Array[Tensor[_ <: scala.Any]]])
      graphBuilder.putAttr("sharedWeight", weightAttrBuilder.build)

      val biasAttrBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context, biasAttrBuilder, bias,
        universe.typeOf[Array[Tensor[_ <: scala.Any]]])
      graphBuilder.putAttr("sharedBias", biasAttrBuilder.build)
    }

    val inputNamesAttrBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, inputNamesAttrBuilder,
      inputsNames, universe.typeOf[Array[String]])
    graphBuilder.putAttr("inputNames", inputNamesAttrBuilder.build)

    val outputNamesBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, outputNamesBuilder,
      outputsNames, universe.typeOf[Array[String]])
    graphBuilder.putAttr("outputNames", outputNamesBuilder.build)

    if (graph.isInstanceOf[DynamicGraph[_]]) {
      val generateBackwardBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context, generateBackwardBuilder,
        graph.asInstanceOf[DynamicGraph[_]].generateBackward, universe.typeOf[Boolean])
      graphBuilder.putAttr("generateBackward", generateBackwardBuilder.build)
    }

    val stopGradientLayers = graph.getStopGradientLayers

    if (stopGradientLayers != null && stopGradientLayers.size > 0) {
      val stopGradientLayersBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context, stopGradientLayersBuilder,
        stopGradientLayers.toArray(new Array[String](stopGradientLayers.size)),
        universe.typeOf[Array[String]])
      graphBuilder.putAttr("stopGradientLayers", stopGradientLayersBuilder.build)
    }
  }
}
