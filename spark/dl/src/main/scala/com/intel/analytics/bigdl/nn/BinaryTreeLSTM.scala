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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.util.control.Breaks._

class BinaryTreeLSTM[T: ClassTag](
  inputSize: Int,
  hiddenSize: Int,
  gateOutput: Boolean = true
)(implicit ev: TensorNumeric[T])
  extends TreeLSTM[T](inputSize, hiddenSize) {
//  val modules: ArrayBuffer[Module[T]] = ArrayBuffer[Module[T]]()
  val composers: ArrayBuffer[Module[T]] = ArrayBuffer[Module[T]]()
  val leafModules: ArrayBuffer[Module[T]] = ArrayBuffer[Module[T]]()
  val composer: Module[T] = createComposer()
  val leafModule: Module[T] = createLeafModule()
  val cells = ArrayBuffer[Module[T]]()

  def createLeafModule(): Module[T] = {
    val input = Identity().apply()
    val c = Linear(inputSize, hiddenSize).apply(input)
    val h: ModuleNode[T] = if (gateOutput) {
      val o = Sigmoid().apply(Linear(inputSize, hiddenSize).apply(input))
      CMulTable().apply(o, Tanh().apply(c))
    } else {
      Tanh().apply(c)
    }

    val leafModule = Graph(Array(input), Array(c, h))

    if (this.leafModule != null) {
      shareParams(leafModule, this.leafModule)
    }

    leafModule
  }

  def createComposer(): Module[T] = {
    val (lc, lh) = (Identity().apply(), Identity().apply())
    val (rc, rh) = (Identity().apply(), Identity().apply())

    def newGate(): ModuleNode[T] = CAddTable().apply(
      Linear(hiddenSize, hiddenSize).apply(lh),
      Linear(hiddenSize, hiddenSize).apply(rh)
    )

    val i = Sigmoid().apply(newGate())
    val lf = Sigmoid().apply(newGate())
    val rf = Sigmoid().apply(newGate())
    val update = Tanh().apply(newGate())
    val c = CAddTable().apply(
      CMulTable().apply(i, update),
      CMulTable().apply(lf, lc),
      CMulTable().apply(rf, rc)
    )

    val h = if (this.gateOutput) {
      val o = Sigmoid().apply(newGate())
      CMulTable().apply(o, Tanh().apply(c))
    } else {
      Tanh().apply(c)
    }

    val composer = Graph(Array(lc, lh, rc, rh), Array(c, h))

    if (this.composer != null) {
      shareParams(composer, this.composer)
    }

    composer
  }

  override def updateOutput(input: Table): Tensor[T] = {
    val tensorTree = new TensorTree[T](input(2))
    for (i <- 1 to tensorTree.size(1)) {
      if (tensorTree.noChild(i)) {
        val numLeafModules = leafModules.size
        if (numLeafModules == 0) {
          cells.append(createLeafModule())
        } else {
          cells.append(leafModules.remove(numLeafModules - 1))
        }
      } else {
        val numComposers = composers.size
        if (numComposers == 0) {
          cells.append(createComposer())
        } else {
          cells.append(composers.remove(numComposers - 1))
        }
      }
    }
    recursiveForward(input(1), tensorTree, tensorTree.getRoot)
    output.resize(cells.size, hiddenSize)
    for (i <- 1 to cells.size) {
      output.select(1, i).copy(unpackState(cells(i - 1).output.toTable)._2)
    }
    output
  }

  def recursiveForward(input: Tensor[T], tree: TensorTree[T], nodeIndex: Int): Table = {
    if (tree.noChild(nodeIndex)) {
      cells(nodeIndex - 1)
        .forward(input.select(1, tree.leafIndex(nodeIndex))).toTable
    } else {
      val leftOut = recursiveForward(input, tree, tree.children(nodeIndex)(0))
      val rigthOut = recursiveForward(input, tree, tree.children(nodeIndex)(1))
      val (lc, lh) = unpackState(leftOut)
      val (rc, rh) = unpackState(rigthOut)

      cells(nodeIndex - 1).forward(T(lc, lh, rc, rh)).toTable
    }
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    val tensorTree = new TensorTree[T](input(2))
    recursiveBackward(input(1),
      tensorTree,
      gradOutput,
      T(memZero, memZero),
      tensorTree.getRoot)
    gradInput
  }

  def recursiveBackward(
    inputs: Tensor[T],
    tree: TensorTree[T],
    outputGrads: Tensor[T],
    gradOutput: Table,
    nodeIndex: Int
  ): Unit = {
    val outputGrad = outputGrads.select(1, nodeIndex)
    gradInput[Tensor[T]](1).resizeAs(inputs)

    if (tree.noChild(nodeIndex)) {
      gradInput[Tensor[T]](1)
        .select(1, tree.leafIndex(nodeIndex))
        .copy(
          cells(nodeIndex - 1)
            .backward(
              inputs.select(1, tree.leafIndex(nodeIndex)),
              T(gradOutput(1), gradOutput[Tensor[T]](2) + outputGrad)
            ).toTensor)

      leafModules.append(cells(nodeIndex - 1))
    } else {
      val children = tree.children(nodeIndex)
      val (lc, lh) = unpackState(cells(children(0) - 1).output.toTable)
      val (rc, rh) = unpackState(cells(children(1) - 1).output.toTable)
      val composerGrad = cells(nodeIndex - 1)
        .backward(T(lc, lh, rc, rh), T(gradOutput(1), gradOutput[Tensor[T]](2) + outputGrad))
        .toTable

      composers.append(cells(nodeIndex - 1))
      recursiveBackward(
        inputs,
        tree,
        outputGrads,
        T(composerGrad(1),
          composerGrad(2)),
        children(0))
      recursiveBackward(
        inputs,
        tree,
        outputGrads,
        T(composerGrad(3),
          composerGrad(4)),
        children(1))
    }

    cells.clear()
  }

  def unpackState(state: Table): (Tensor[T], Tensor[T]) = {
    if (state.length() == 0) {
      (memZero, memZero)
    } else {
      (state(1), state(2))
    }
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    val param = new ArrayBuffer[Tensor[T]]()
    val gradParam = new ArrayBuffer[Tensor[T]]()
    val (cp, cg) = composer.parameters()
    val (lp, lg) = leafModule.parameters()

    (cp ++ lg, cg ++ lg)
  }
}

object BinaryTreeLSTM {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputSize: Int,
    hiddenSize: Int,
    gateOutput: Boolean = true
  )(implicit ev: TensorNumeric[T]): BinaryTreeLSTM[T] =
    new BinaryTreeLSTM[T](inputSize, hiddenSize, gateOutput)
}

class TensorTree[T: ClassTag](val content: Tensor[T])
  (implicit ev: TensorNumeric[T]) extends Serializable {
  def size: Array[Int] = content.size()

  def children(index: Int): Array[Int] =
    content.select(1, index).toBreezeVector().toArray.map(ev.toType[Int])

  def addChild(parent: Int, child: T): Unit = {
    for (i <- 1 to size(1)) {
      if (content(Array(parent, i)) == 0) {
        content.setValue(parent, i, child)
        break()
      }
    }
  }

  def markAsRoot(index: Int): Unit = {
    content.setValue(index, size(1), ev.negative(ev.one))
  }

  def getRoot: Int = {
    for (i <- 1 to size(0)) {
      if (content(Array(i, size(1))) == -1) {
        return i
      }
    }

    -1
  }

  def markAsLeaf(index: Int, leafIndex: Int): Unit = {
    content.setValue(index, size(1), ev.fromType(leafIndex))
  }

  def leafIndex(index: Int): Int = {
    ev.toType[Int](content(Array(index, size(1))))
  }

  def hasChild(index: Int): Boolean = {
    content(Array(index, 1)) != 0
  }

  def noChild(index: Int): Boolean = {
    content(Array(index, 1)) == 0
  }

  def exists(index: Int): Boolean = {
    index >= 1 && index <= size(0)
  }
}

