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
  val cells: ArrayBuffer[ArrayBuffer[Module[T]]] = ArrayBuffer[ArrayBuffer[Module[T]]]()

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

  def createLeafModule1(): Module[T] = {
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

  def createComposer(): Module[T] = {
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

  def createComposer1(): Module[T] = {
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

  //  var flag = true
  override def updateOutput(input: Table): Tensor[T] = {
    val inputs = input[Tensor[T]](1)
    val trees = input[Tensor[T]](2)
    //    if (flag) {
    //      println(s"input is ${inputs(1)(2)}")
    //    }
    //    flag = false
    //
    //    println(s"tree is ${trees(1)}")
    val batchSize = inputs.size(1)
    val nodeSize = trees.size(2)
    output.resize(batchSize, nodeSize, hiddenSize)
    output.zero()

    for (b <- 1 to batchSize) {
      cells.append(ArrayBuffer[Module[T]]())
    }

    for (b <- 1 to batchSize) {
      val tensorTree = new TensorTree[T](trees(b))
      for (i <- 1 to tensorTree.size(0)) {
        if (tensorTree.noChild(i)) {
          val numLeafModules = leafModules.size
          if (numLeafModules == 0) {
            cells(b - 1).append(createLeafModule())
          } else {
            cells(b - 1).append(leafModules.remove(numLeafModules - 1))
          }
        } else if (tensorTree.hasChild(i)) {
          val numComposers = composers.size
          if (numComposers == 0) {
            cells(b - 1).append(createComposer())
          } else {
            cells(b - 1).append(composers.remove(numComposers - 1))
          }
        }
      }
      recursiveForward(b, inputs.select(1, b), tensorTree, tensorTree.getRoot)
      output(b).resize(nodeSize, hiddenSize)
      //      println("output size ", output(b).size(1))
      //      println("cells size ", cells(b - 1).size)
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
//      println("lc: ", lc(1)(1))
//      println("lh: ", lh(1)(1))
//      println("rc: ", rc(1)(1))
//      println("rh: ", rh(1)(1))
      cells(batch - 1)(nodeIndex - 1).forward(T(lc, lh, rc, rh)).toTable
      val (c, h) = unpackState(cells(batch - 1)(nodeIndex - 1).output.toTable)
//      println("c: ", c(1)(1))
//      println("h: ", h(1)(1))
      cells(batch - 1)(nodeIndex - 1).output.toTable
    }
    //    println("binary tree output", out[Tensor[T]](1)(1), out[Tensor[T]](2)(1))
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
    val nodeSize = trees.size(2)

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
    //    val p1 = cells(0)(0).getParameters()
    //    val p2 = cells(0)(1).getParameters()
    //    val p3 = cells(0)(2).getParameters()
    //    val p4 = cells(0)(3).getParameters()
    //    val p5 = cells(0)(4).getParameters()
    //    val p6 = cells(0)(5).getParameters()
    //    val p7 = cells(0)(6).getParameters()
    //    val p8 = cells(0)(7).getParameters()
    //    val p9 = cells(0)(8).getParameters()
    //    val p10 = cells(0)(9).getParameters()
    cells.clear()
//    println("lp: ", leafModule.parameters()._1(0)(1)(1))
//    println("lg: ", leafModule.parameters()._2(0)(1))
//    println("cp: ", composer.parameters()._1(0)(1))
//    println("cg: ", composer.parameters()._2(0))

    //    println(s"gradInput: ${leafModule.getParameters()._2(1)}")
    //    val (cp, cg) = composer.getParameters()
    //    val (lp, lg) = leafModule.getParameters()
    //    println("composer weight gradient", cg.select(1, 1))
    //    println("leafModule weight gradient", lg.select(1, 1))
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
//        println("gradOutput ", outputGrad(1))

    if (tree.noChild(nodeIndex)) {
      gradInput[Tensor[T]](1)(batch)(tree.leafIndex(nodeIndex))
        .copy(
          cells(batch - 1)(nodeIndex - 1)
            .backward(
              inputs.select(1, tree.leafIndex(nodeIndex)),
              T(gradOutput(1), gradOutput[Tensor[T]](2) + outputGrad)
            ).toTensor)

//      println("gradInput ", gradInput[Tensor[T]](1)(batch)(tree.leafIndex(nodeIndex))(1))
      leafModules.append(cells(batch - 1)(nodeIndex - 1))
    } else {
      val children = tree.children(nodeIndex)
      val (lc, lh) = unpackState(cells(batch - 1)(children(0) - 1).output.toTable)
      val (rc, rh) = unpackState(cells(batch - 1)(children(1) - 1).output.toTable)
      val composerGrad = cells(batch - 1)(nodeIndex - 1)
        .backward(T(lc, lh, rc, rh), T(gradOutput(1), gradOutput[Tensor[T]](2) + outputGrad))
        .toTable

      composers.append(cells(batch - 1)(nodeIndex - 1))
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
    val param = new ArrayBuffer[Tensor[T]]()
    val gradParam = new ArrayBuffer[Tensor[T]]()
    val (cp, cg) = composer.parameters()
    val (lp, lg) = leafModule.parameters()
//    cp.foreach(param += _)
//    lp.foreach(param += _)
//    cg.foreach(gradParam += _)
//    lg.foreach(gradParam += _)
//    param += cp
//    println("params", lp(1)(1))
//    println("params", leafModule.getParameters()._1(1))
//    println("grad_params", lg(1)(1))
    (cp ++ lp, cg ++ lg)
//    (param.toArray, gradParam.toArray)
  }


  override def zeroGradParameters(): Unit = {
    composer.zeroGradParameters()
    leafModule.zeroGradParameters()
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
    breakable {
      for (i <- 1 to size(1)) {
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

    -1
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

