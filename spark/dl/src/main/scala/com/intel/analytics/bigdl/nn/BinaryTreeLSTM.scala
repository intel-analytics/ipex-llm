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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer._
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.utils.{T, Table}
import serialization.Bigdl.{AttrValue, BigDLModule}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.reflect.runtime.universe
import scala.util.control.Breaks._

/**
 * This class is an implementation of Binary TreeLSTM (Constituency Tree LSTM).
 * @param inputSize input units size
 * @param hiddenSize hidden units size
 * @param gateOutput whether gate output
 * @param withGraph whether create lstms with [[Graph]], the default value is true.
 */
class BinaryTreeLSTM[T: ClassTag](
  inputSize: Int,
  hiddenSize: Int,
  gateOutput: Boolean = true,
  withGraph: Boolean = true
)(implicit ev: TensorNumeric[T])
  extends TreeLSTM[T](inputSize, hiddenSize) {
  private var composer: Module[T] = createComposer()
  private var leafModule: Module[T] = createLeafModule()
  val composers: ArrayBuffer[Module[T]] = ArrayBuffer[Module[T]](composer)
  val leafModules: ArrayBuffer[Module[T]] = ArrayBuffer[Module[T]](leafModule)
  val cells: ArrayBuffer[ArrayBuffer[Module[T]]] = ArrayBuffer[ArrayBuffer[Module[T]]]()

  def createLeafModule(): Module[T] = {
    if (withGraph) createLeafModuleWithGraph()
    else createLeafModuleWithSequential()
  }

  def createComposer(): Module[T] = {
    if (withGraph) createComposerWithGraph()
    else createComposerWithSequential()
  }

  def createLeafModuleWithGraph(): Module[T] = {
    val input = Input()
    val c = Linear(inputSize, hiddenSize).inputs(input)
    val h: ModuleNode[T] = if (gateOutput) {
      val o = Sigmoid().inputs(Linear(inputSize, hiddenSize).inputs(input))
      CMulTable().inputs(o, Tanh().inputs(c))
    } else {
      Tanh().inputs(c)
    }

    val leafModule = Graph(Array(input), Array(c, h))

    if (this.leafModule != null) {
      shareParams(leafModule, this.leafModule)
    }

    leafModule
  }

  def createComposerWithGraph(): Module[T] = {
    val (lc, lh) = (Input(), Input())
    val (rc, rh) = (Input(), Input())

    def newGate(): ModuleNode[T] = CAddTable().inputs(
      Linear(hiddenSize, hiddenSize).inputs(lh),
      Linear(hiddenSize, hiddenSize).inputs(rh)
    )

    val i = Sigmoid().inputs(newGate())
    val lf = Sigmoid().inputs(newGate())
    val rf = Sigmoid().inputs(newGate())
    val update = Tanh().inputs(newGate())
    val c = CAddTable().inputs(
      CMulTable().inputs(i, update),
      CMulTable().inputs(lf, lc),
      CMulTable().inputs(rf, rc)
    )

    val h = if (this.gateOutput) {
      val o = Sigmoid().inputs(newGate())
      CMulTable().inputs(o, Tanh().inputs(c))
    } else {
      Tanh().inputs(c)
    }

    val composer = Graph(Array(lc, lh, rc, rh), Array(c, h))

    if (this.composer != null) {
      shareParams(composer, this.composer)
    }

    composer
  }

  def createLeafModuleWithSequential(): Module[T] = {
    val gate = ConcatTable()
      .add(Sequential()
        .add(Linear(inputSize, hiddenSize))
        .add(ConcatTable()
          .add(Identity())
          .add(Tanh())))
      .add(Sequential()
        .add(Linear(inputSize, hiddenSize))
        .add(Sigmoid()))


    val leafModule = Sequential()
      .add(gate)
      .add(FlattenTable())
      .add(ConcatTable()
        .add(SelectTable(1))
        .add(Sequential()
          .add(NarrowTable(2, 2))
          .add(CMulTable())))


    if (this.leafModule != null) {
      shareParams(leafModule, this.leafModule)
    }

    leafModule
  }

  def createComposerWithSequential(): Module[T] = {
    def newGate(): Module[T] =
      Sequential()
        .add(ParallelTable()
          .add(Linear(hiddenSize, hiddenSize))
          .add(Linear(hiddenSize, hiddenSize)))
        .add(CAddTable())

    val gates = Sequential()
      .add(ConcatTable()
        .add(SelectTable(2))
        .add(SelectTable(4)))
      .add(ConcatTable()
        .add(Sequential()
          .add(newGate())
          .add(Sigmoid())) // i
        .add(Sequential()
        .add(newGate())
        .add(Sigmoid())) // lf
        .add(Sequential()
        .add(newGate())
        .add(Sigmoid())) // rf
        .add(Sequential()
        .add(newGate())
        .add(Tanh()))    // update
        .add(Sequential()
        .add(newGate())
        .add(Sigmoid()))) // o

    val i2c = Sequential()
      .add(ConcatTable()
        .add(Sequential()
          .add(ConcatTable()
            .add(SelectTable(3))  // i
            .add(SelectTable(6))) // update
          .add(CMulTable()))
        .add(Sequential()
          .add(ConcatTable()
            .add(SelectTable(4))  // lf
            .add(SelectTable(1))) // lc
          .add(CMulTable()))
        .add(Sequential()
          .add(ConcatTable()
            .add(SelectTable(5))  // rf
            .add(SelectTable(2))) // rc
          .add(CMulTable())))
      .add(CAddTable())

    val composer = Sequential()
      .add(ConcatTable()
        .add(SelectTable(1)) // lc
        .add(SelectTable(3)) // rc
        .add(gates))
      .add(FlattenTable())
      .add(ConcatTable()
        .add(i2c)
        .add(SelectTable(7))) // o
      .add(ConcatTable()
      .add(SelectTable(1)) // c
      .add(Sequential()
      .add(ParallelTable()
        .add(Tanh())     // Tanh(c)
        .add(Identity()))// o
      .add(CMulTable())))// h


    if (this.composer != null) {
      shareParams(composer, this.composer)
    }

    composer
  }

  override def updateOutput(input: Table): Tensor[T] = {
    cells.clear()
    val inputs = input[Tensor[T]](1)
    val trees = input[Tensor[T]](2)
    val batchSize = inputs.size(1)
    val nodeSize = trees.size(2)
    output.resize(batchSize, nodeSize, hiddenSize)
    output.zero()

    for (b <- 1 to batchSize) {
      cells.append(ArrayBuffer[Module[T]]())
    }

    var leafIndex = 0
    var composerIndex = 0
    for (b <- 1 to batchSize) {
      val tensorTree = new TensorTree[T](trees(b))
      for (i <- 1 to tensorTree.nodeNumber) {
        if (tensorTree.noChild(i)) {
          if (leafIndex > leafModules.length - 1) {
            val leafModule = createLeafModule()
            cells(b - 1).append(leafModule)
            leafModules.append(leafModule)
          } else {
            cells(b - 1).append(leafModules(leafIndex))
          }
          leafIndex += 1
        } else if (tensorTree.hasChild(i)) {
          if (composerIndex > composers.length - 1) {
            val composer = createComposer()
            cells(b - 1).append(composer)
            composers.append(composer)
          } else {
            cells(b - 1).append(composers(composerIndex))
          }
          composerIndex += 1
        }
      }
      recursiveForward(b, inputs.select(1, b), tensorTree, tensorTree.getRoot)
      for (i <- 1 to cells(b - 1).size) {
        output(b)(i).copy(unpackState(cells(b - 1)(i - 1).output.toTable)._2)
      }
    }
    output
  }

  def recursiveForward(
    batch: Int,
    input: Tensor[T],
    tree: TensorTree[T],
    nodeIndex: Int): Table = {
    val out = if (tree.noChild(nodeIndex)) {
      cells(batch - 1)(nodeIndex - 1)
        .forward(input.select(1, tree.leafIndex(nodeIndex))).toTable
    } else {
      val leftOut = recursiveForward(batch, input, tree, tree.children(nodeIndex)(0))
      val rightOut = recursiveForward(batch, input, tree, tree.children(nodeIndex)(1))
      val (lc, lh) = unpackState(leftOut)
      val (rc, rh) = unpackState(rightOut)
      val cell = cells(batch - 1)(nodeIndex - 1)
      cell.forward(T(lc, lh, rc, rh)).toTable
    }
    out
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    if (!(gradInput.contains(1) || gradInput.contains(2))) {
      gradInput = T(Tensor(), Tensor())
    }

    val inputs = input[Tensor[T]](1)
    val trees = input[Tensor[T]](2)

    gradInput[Tensor[T]](1).resizeAs(inputs)
    gradInput[Tensor[T]](2).resizeAs(trees)

    val batchSize = inputs.size(1)

    for (b <- 1 to batchSize) {
      val tensorTree = new TensorTree[T](trees(b))
      recursiveBackward(
        b,
        inputs(b),
        tensorTree,
        gradOutput(b),
        T(memZero, memZero),
        tensorTree.getRoot)
    }
    gradInput
  }

  def recursiveBackward(
    batch: Int,
    inputs: Tensor[T],
    tree: TensorTree[T],
    outputGrads: Tensor[T],
    gradOutput: Table,
    nodeIndex: Int
  ): Unit = {
    val outputGrad = outputGrads(nodeIndex)

    if (tree.noChild(nodeIndex)) {
      gradInput[Tensor[T]](1)(batch)(tree.leafIndex(nodeIndex))
        .copy(
          cells(batch - 1)(nodeIndex - 1)
            .backward(
              inputs.select(1, tree.leafIndex(nodeIndex)),
              T(gradOutput(1), gradOutput[Tensor[T]](2) + outputGrad)
            ).toTensor)

    } else {
      val children = tree.children(nodeIndex)
      val (lc, lh) = unpackState(cells(batch - 1)(children(0) - 1).output.toTable)
      val (rc, rh) = unpackState(cells(batch - 1)(children(1) - 1).output.toTable)
      val composerGrad = cells(batch - 1)(nodeIndex - 1)
        .backward(T(lc, lh, rc, rh), T(gradOutput(1), gradOutput[Tensor[T]](2) + outputGrad))
        .toTable

      recursiveBackward(
        batch,
        inputs,
        tree,
        outputGrads,
        T(composerGrad[Tensor[T]](1),
          composerGrad[Tensor[T]](2)),
        children(0))
      recursiveBackward(
        batch,
        inputs,
        tree,
        outputGrads,
        T(composerGrad[Tensor[T]](3),
          composerGrad[Tensor[T]](4)),
        children(1))
    }
  }

  def unpackState(state: Table): (Tensor[T], Tensor[T]) = {
    if (state.length() == 0) {
      (memZero, memZero)
    } else {
      (state(1), state(2))
    }
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    val (cp, cg) = composer.parameters()
    val (lp, lg) = leafModule.parameters()
    (cp ++ lp, cg ++ lg)
  }

  override def getParametersTable(): Table = {
    val pt = T()
    val t1 = composer.getParametersTable()
    val t2 = leafModule.getParametersTable()
    t1.keySet.foreach(key => pt(key) = t1(key))
    t2.keySet.foreach(key => pt(key) = t2(key))
    pt
  }

  override def reset(): Unit = {
    composer.reset()
    leafModule.reset()
  }

  override def hashCode(): Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(super.hashCode(), composer, leafModule)
    state.map(getHashCode).foldLeft(0)((a, b) => 31 * a + b)
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[BinaryTreeLSTM[T]]

  override def equals(other: Any): Boolean = other match {
    case that: BinaryTreeLSTM[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        composer == that.composer &&
        leafModule == that.leafModule
    case _ => false
  }
}

object BinaryTreeLSTM extends ModuleSerializable {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputSize: Int,
    hiddenSize: Int,
    gateOutput: Boolean = true,
    withGraph: Boolean = true
  )(implicit ev: TensorNumeric[T]): BinaryTreeLSTM[T] =
    new BinaryTreeLSTM[T](inputSize, hiddenSize, gateOutput, withGraph)

  override def doLoadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {

    val binaryTreeLSTMModule = super.doLoadModule(context).asInstanceOf[BinaryTreeLSTM[T]]
    binaryTreeLSTMModule.composers.clear
    binaryTreeLSTMModule.leafModules.clear

    val attrMap = context.bigdlModule.getAttrMap

    DataConverter.getAttributeValue(context, attrMap.get("composers")).
      asInstanceOf[Array[Module[T]]].foreach(module => {
      binaryTreeLSTMModule.composers.append(module)
    })

    DataConverter.getAttributeValue(context, attrMap.get("leafModules")).
      asInstanceOf[Array[Module[T]]].foreach(module => {
      binaryTreeLSTMModule.leafModules.append(module)
    })

    binaryTreeLSTMModule.leafModule = DataConverter.
      getAttributeValue(context, attrMap.get("leafModule")).
      asInstanceOf[Module[T]]

    binaryTreeLSTMModule.composer = DataConverter.getAttributeValue(context,
      attrMap.get("composer")).
      asInstanceOf[Module[T]]

    binaryTreeLSTMModule
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
                                              binaryTreeLSTMBuilder : BigDLModule.Builder)
                                           (implicit ev: TensorNumeric[T]) : Unit = {
    super.doSerializeModule(context, binaryTreeLSTMBuilder)

    val binaryTreeLSTM = context.moduleData.module.asInstanceOf[BinaryTreeLSTM[T]]

    val composer = binaryTreeLSTM.composer
    val composerBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, composerBuilder, composer,
      ModuleSerializer.abstractModuleType)
    binaryTreeLSTMBuilder.putAttr("composer", composerBuilder.build)


    val leafModule = binaryTreeLSTM.leafModule
    val leafModuleBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, leafModuleBuilder, leafModule,
      ModuleSerializer.abstractModuleType)
    binaryTreeLSTMBuilder.putAttr("leafModule", leafModuleBuilder.build)

    val composers = binaryTreeLSTM.composers.toArray
    val composersBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, composersBuilder, composers,
      universe.
        typeOf[Array[_ <: AbstractModule[Activity, Activity, _ <: Any]]])
    binaryTreeLSTMBuilder.putAttr("composers", composersBuilder.build)

    val leafModules = binaryTreeLSTM.leafModules.toArray
    val leafModulesBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, leafModulesBuilder, leafModules, universe.
      typeOf[Array[_ <: AbstractModule[Activity, Activity, _ <: Any]]])
    binaryTreeLSTMBuilder.putAttr("leafModules", leafModulesBuilder.build)

  }
}

/**
 * [[TensorTree]] class is used to decode a tensor to a tree structure.
 * The given input `content` is a tensor which encodes a constituency parse tree.
 * The tensor should have the following structure:
 *
 * Each row of the tensor represents a tree node and the row number is node number
 * For each row, except the last column, all other columns represent the children
 * node number of this node. Assume the value of a certain column of the row is not zero,
 * the value `p` means this node has a child whose node number is `p` (lies in the `p`-th)
 * row. Each leaf has a leaf number, in the tensor, the last column represents the leaf number.
 * Each leaf does not have any children, so all the columns of a leaf except the last should
 * be zero. If a node is the root, the last column should equal to `-1`.
 *
 * Note: if any row for padding, the padding rows should be placed at the last rows with all
 * elements equal to `-1`.
 *
 * eg. a tensor represents a binary tree:
 *
 * [11, 10, -1;
 *  0, 0, 1;
 *  0, 0, 2;
 *  0, 0, 3;
 *  0, 0, 4;
 *  0, 0, 5;
 *  0, 0, 6;
 *  4, 5, 0;
 *  6, 7, 0;
 *  8, 9, 0;
 *  2, 3, 0;
 *  -1, -1, -1;
 *  -1, -1, -1]
 *
 * @param content the tensor to be encoded
 * @param ev  implicit tensor numeric
 * @tparam T  Numeric type [[Float]] or [[Double]]
 */
class TensorTree[T: ClassTag](val content: Tensor[T])
  (implicit ev: TensorNumeric[T]) extends Serializable {
  require(content.dim() == 2,
    "The content of TensorTree should be a two-dimensional tensor" +
      s"content dim(${content.dim()})")
  def size: Array[Int] = content.size()

  def nodeNumber: Int = size(0)

  def children(index: Int): Array[Int] =
    content.select(1, index).toBreezeVector().toArray.map(ev.toType[Int])

  def addChild(parent: Int, child: T): Unit = {
    breakable {
      for (i <- 1 until size(1)) {
        if (content(Array(parent, i)) == ev.zero) {
          content.setValue(parent, i, child)
          break()
        }
      }
    }
  }

  def markAsRoot(index: Int): Unit = {
    content.setValue(index, size(1), ev.negative(ev.one))
  }

  def getRoot: Int = {
    for (i <- 1 to size(0)) {
      if (ev.toType[Int](content(Array(i, size(1)))) == -1) {
        return i
      }
    }

    throw new RuntimeException("There is no root in the tensor tree")
  }

  def markAsLeaf(index: Int, leafIndex: Int): Unit = {
    content.setValue(index, size(1), ev.fromType(leafIndex))
  }

  def leafIndex(index: Int): Int = {
    ev.toType[Int](content(Array(index, size(1))))
  }

  def hasChild(index: Int): Boolean = {
    ev.toType[Int](content(Array(index, 1))) > 0
  }

  def noChild(index: Int): Boolean = {
    ev.toType[Int](content(Array(index, 1))) == 0
  }

  def exists(index: Int): Boolean = {
    index >= 1 && index <= size(0)
  }

  def isPadding(index: Int): Boolean = {
    ev.toType[Int](content(Array(index, 1))) == -1
  }
}
