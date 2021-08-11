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
package com.intel.analytics.bigdl.nn.tf

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.Identity
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.ops.Operation
import com.intel.analytics.bigdl.tensor.{BooleanType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Edge, Node, T}

import scala.reflect.ClassTag

/**
 *  Control flow related operations
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now
 */
sealed abstract class ControlOps[T: ClassTag]()(implicit ev: TensorNumeric[T])
  extends Operation[Activity, Activity, T] {
  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    throw new UnsupportedOperationException("Operation does not support updateGradInput() method")
  }
}

/**
 * Control flow related operations, and they just pass the input without modifying them
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now
 */
abstract class IdentityControl[T: ClassTag]()(implicit ev: TensorNumeric[T]) extends ControlOps[T] {
  override def updateOutput(input: Activity): Activity = {
    output = input
    output
  }
}


/**
 * Switch the control flow. It will construct a table. It accepts a table input containing two
 * elements. The first element is a boolean scalar. The second element is the data.
 * It produces a table output containing two elements. If the boolean scalar is true, the first
 * element of the output is data and the second one is null; if the boolean scala is false, the
 * position is exchanged.
 *
 * When connect to some other node. You should never connect the whole output to other node. You
 * should always use SwitchNodeOutput(1) and SwitchNodeOutput(2). Or there will be run time failure.
 *
 * User should use ControlNodes.whileLoop or ControlNodes.switch to use this operation
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now
 */
private[bigdl] class SwitchOps[T: ClassTag]()(implicit ev: TensorNumeric[T]) extends ControlOps[T] {
  override def updateOutput(input: Activity): Activity = {
    val condition = input.toTable[Tensor[Boolean]](2)
    val data = input.toTable[Activity](1)
    if (condition.value()) {
      this.output = T(null, data)
    } else {
      this.output = T(data, null)
    }
    this.output
  }
}

/**
 * MergeOps will run as soon as one of the node dependency is ready. and pass the data from that
 * node. If the MergeOps is not in a loop, it should only be executed once. If it's in a loop, it
 * should be still executed once in a iteration.
 *
 * User should use ControlNodes.whileLoop or ControlNodes.merge to use this operation
 * @param switch which dependency node is avaliable
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now
 */
private[bigdl] class MergeOps[T: ClassTag](private var switch : Int = 1)(
  implicit ev: TensorNumeric[T]) extends ControlOps[T] {

  def setSwitch(s: Int) : this.type = {
    this.switch = s
    this
  }

  override def updateOutput(input: Activity): Activity = {
    this.output = input.toTable[Activity](switch)
    this.output
  }

  override def toString(): String = getPrintName() + s"($switch)"
}

/**
 * A wrapper of node for switch operation. Make code easy to read.
 *
 * @param element element
 * @tparam T element type
 */
sealed class  SwitchControlNode[T] (element: T) extends Node[T](element) {

  /**
   * The output edge which will be run when condition scalar is true. You should not connect one
   * node with both type edges.
   * @return
   */
  def trueEdge() : ((Node[T], Int)) = (this, 2)

  /**
   * The output edge which will be run when condition scalar is false. You should not connect one
   * node with both type edges.
   * @return
   */
  def falseEdge() : ((Node[T], Int)) = (this, 1)

  /**
   * Return nodes triggered by current node
   * @return
   */
  def availableNodes() : Seq[Node[T]] = {
    val bothNodes = this.nextNodesAndEdges.filter(_._2.fromIndex.isEmpty).map(_._1).distinct
    require(bothNodes.length == 0, "You should not connect to one node with both type of edges")

    val trueNodes = this.nextNodesAndEdges.filter(_._2.fromIndex.get == 2).map(_._1).distinct
    val falseNodes = this.nextNodesAndEdges.filter(_._2.fromIndex.get == 1).map(_._1).distinct
    trueNodes.foreach( n =>
      require(!falseNodes.contains(n),
        "You should not connect to one node with both type of edges")
    )

    val switch = element.asInstanceOf[SwitchOps[T]]
    if (switch.output.toTable(1) != null) {
      falseNodes
    } else {
      trueNodes
    }
  }
}

/**
 * A wrapper of node for merge operation.
 *
 * @param element element
 * @tparam T element type
 */
sealed class MergeControlNode[T] private[bigdl] (element: T) extends Node[T](element) {

  /**
   * Add another dependency node
   * @param dependency
   * @return
   */
  def append(dependency: Node[T]): this.type = {
    dependency -> this
    this
  }

  /**
   * Add another dependency node with edge
   * @param dependencyIndex
   * @return
   */
  def append(dependencyIndex: (Node[T], Int)): this.type = {
    dependencyIndex._1.add(this, Edge(dependencyIndex._2))
    this
  }
}

/**
 * Mark start of next iteration. User should use ControlNodes.whileLoop to use such operation.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now
 */
sealed private[bigdl] class NextIteration[T: ClassTag, D: ClassTag] private[bigdl]()
  (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends Operation[Tensor[D], Tensor[D], T] {
  output = Tensor[D]()

  override def updateOutput(input: Tensor[D]): Tensor[D] = {
    output.resizeAs(input).copy(input)
  }

  override def getClassTagNumerics(): (Array[ClassManifest[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}

/**
 * Mark start of a loop. User should use ControlNodes.whileLoop to use such operation.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now
 */
sealed private[bigdl] class Enter[T: ClassTag] private[bigdl](val frame: String)
  (implicit ev: TensorNumeric[T]) extends IdentityControl[T]

/**
 * Mark this dataflow is condition flow. It will erase the iteration status.
 * User should use ControlNodes.whileLoop to use such operation.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now
 */
sealed private[bigdl] class LoopCondition[T: ClassTag] private[bigdl]()
  (implicit ev: TensorNumeric[T]) extends IdentityControl[T] {

  /**
   * If current loop continue running
   * @return
   */
  private[bigdl] def continue() : Boolean = {
    require(this.output.isTensor, "loop condition result should be a tensor")
    val t = this.output.asInstanceOf[Tensor[Boolean]]
    require((t.isScalar || t.nElement() == 1) && t.getType() == BooleanType,
      "loop condition result should be a boolean scalar or one element tensor")
    t.storage().apply(t.storageOffset() - 1)
  }
}

/**
 * Mark end of a loop. User should use ControlNodes.whileLoop to use such operation.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now
 */
sealed private[bigdl] class Exit[T: ClassTag] private[bigdl]()(implicit ev: TensorNumeric[T])
  extends IdentityControl[T]

/**
 * Factory method of control flow related nodes
 */
private[bigdl] object ControlNodes {

  /**
   * Create a switch node
   * @param data data to pass down
   * @param condition condition node, should pass in a boolean scalar
   * @param ev
   * @tparam T
   * @return
   */
  def switch[T: ClassTag](data: ModuleNode[T], condition: ModuleNode[T]
  )(implicit ev: TensorNumeric[T]): SwitchControlNode[Module[T]] = {
    val curNode = new SwitchControlNode[Module[T]](new SwitchOps())
    condition -> curNode
    data -> curNode
    curNode
  }

  /**
   * Create a merge node
   * @param first dependency node, for method overload
   * @param nodesWithIndex dependency nodes
   * @param ev
   * @tparam T
   * @return
   */
  def merge[T: ClassTag](first: (ModuleNode[T], Int), nodesWithIndex : (ModuleNode[T], Int)*)(
    implicit ev: TensorNumeric[T]): MergeControlNode[Module[T]] = {
    val curNode = new MergeControlNode[Module[T]](new MergeOps())
    first._1.add(curNode, Edge(first._2))
    nodesWithIndex.foreach(nodeWithIndex => {
      nodeWithIndex._1.add(curNode, Edge(nodeWithIndex._2))
    })
    curNode
  }

  /**
   * Create a merge node
   * @param nodes dependency nodes
   * @param ev
   * @tparam T
   * @return
   */
  def merge[T: ClassTag](nodes : ModuleNode[T]*)(
    implicit ev: TensorNumeric[T]): MergeControlNode[Module[T]] = {
    val curNode = new MergeControlNode[Module[T]](new MergeOps())
    nodes.foreach(node => {
      node.add(curNode, Edge())
    })
    curNode
  }

  /**
   * Constructor a while loop in the graph
   * @param condition a sub graph produce a boolean scalar
   * @param body while body, input/output tuple. body length is seq of nodes with same length of
   *             loopVars
   * @param loopVars loop vars
   * @tparam T
   * @return a seq of nodes with same length of loopVars
   */
  def whileLoop[T: ClassTag](
    condition: (Seq[ModuleNode[T]], ModuleNode[T]),
    body: Seq[(ModuleNode[T], ModuleNode[T])],
    loopVars: (Seq[ModuleNode[T]]),
    name: String = null
  )(implicit ev: TensorNumeric[T]): Seq[ModuleNode[T]] = {
    val lc = new LoopCondition[T]().inputs(condition._2)
    if (name != null) lc.element.setName(s"$name/loopCondition")

    loopVars.zip(condition._1).zip(body).zipWithIndex.map(tuple => {
      val (((input, cond), update), indexBase0) = tuple
      val index = indexBase0 + 1
      val enter = new Enter[T]("test_frame").inputs(input)
      if (name != null) enter.element.setName(s"$name/enter$index")
      val mergeNode = merge[T](enter)
      if (name != null) mergeNode.element.setName(s"$name/merge$index")
      mergeNode -> cond
      val switchNode = switch[T](lc, mergeNode)
      if (name != null) switchNode.element.setName(s"$name/switch$index")
      val exitNode = new Exit[T]().inputs(switchNode.trueEdge())
      if (name != null) exitNode.element.setName(s"$name/exit$index")
      val identity = Identity[T]().inputs(switchNode.falseEdge())
      if (name != null) identity.element.setName(s"$name/switchFalse$index")
      identity -> update._1
      val nextIteration = new NextIteration[T, T].inputs(update._2)
      if (name != null) nextIteration.element.setName(s"$name/nextIteration$index")
      mergeNode.append(nextIteration)
      exitNode
    })
  }
}
