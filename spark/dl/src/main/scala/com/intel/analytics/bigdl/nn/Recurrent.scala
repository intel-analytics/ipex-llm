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
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer._
import com.intel.analytics.bigdl.utils.serializer.{ContainerSerializable, ModuleSerializer}
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import serialization.Bigdl.{AttrValue, BigDLModule}

import scala.reflect.runtime.universe
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * [[Recurrent]] module is a container of rnn cells
 * Different types of rnn cells can be added using add() function
 *
 * The recurrent includes some mask mechanisms
 * if the `maskZero` variable is set to true, the `Recurrent` module will
 * not consider zero vector inputs. For each time step input, if a certain row is
 * a zero vector (all the elements of the vector equals zero), then output of certain row
 * of this time step would be a zero vector, and the hidden state of the certain row of
 * this time step would be the same as the corresponding row of the hidden state of the
 * previous step.
 *
 */
class Recurrent[T : ClassTag](
  var batchNormParams: BatchNormParams[T] = null,
  var maskZero: Boolean = false
)
  (implicit ev: TensorNumeric[T]) extends DynamicContainer[Tensor[T], Tensor[T], T] {

  protected var hidden: Activity = null
  protected var gradHidden: Activity = null
  protected var hiddenShape: Array[Int] = null
  protected var currentInput = T()
  protected val currentGradOutput = T()
  protected val gradInput2Cell = Tensor[T]()
  protected var input2Cell = Tensor[T]()
  protected var _input = T()
  protected val batchDim = Recurrent.batchDim
  protected val timeDim = Recurrent.timeDim
  protected val inputDim = 1
  protected val hidDim = 2
  protected var (batchSize, times) = (0, 0)
  protected var topology: Cell[T] = null
  protected val stepInput2CellBuf = Tensor[T]()
  protected val stepGradBuffer = Tensor[T]()
  protected var preTopology: AbstractModule[Activity, Activity, T] = null
  private val dropouts: ArrayBuffer[Array[Dropout[T]]] =
    new ArrayBuffer[Array[Dropout[T]]]
  private var layer: TensorModule[T] = null
  private var maskBuffer: Tensor[T] = Tensor()
  private var gradOutputBuff: Table = T()
  private var indexBuffer: Tensor[T] = Tensor()
  private var inputBuffer: Tensor[T] = Tensor()
  private var outputBuffers: ArrayBuffer[Tensor[T]] = ArrayBuffer(Tensor())
  private var minLength: Int = 0

  def getCell(): Cell[T] = topology

  /**
   *
   *  modules: -- preTopology
   *           |- BatchNormalization (optional)
   *           |- topology (cell)
   *
   * The topology (or cell) will be cloned for N times w.r.t the time dimension.
   * The preTopology will be execute only once before the recurrence.
   *
   * @param module module to be add
   * @return this container
   */
  override def add(module: AbstractModule[_ <: Activity, _ <: Activity, T]): this.type = {
    require(module.isInstanceOf[Cell[T]],
      "Recurrent: added module should be Cell type!")
    require(!module.isInstanceOf[MultiRNNCell[T]],
      "Recurrent: added module cannot be MultiRNNCell," +
        "use Sequential().add(Recurrent(cell)).add(Recurrent(cell))... instead!")

    topology = module.asInstanceOf[Cell[T]]
    preTopology = if (topology.preTopology != null) {
      TimeDistributed(topology.preTopology, maskZero = maskZero)
    } else topology.preTopology

    if (batchNormParams != null && preTopology == null) {
      throw new IllegalArgumentException(
        s"${topology.getName} does not support BatchNormalization." +
          s" Please add preTopology for it. You can simply using: " +
          s"override def preTopology: AbstractModule[Activity, Activity, T] = Identity()")
    }

    if (batchNormParams != null) {
      layer = batchNormalization(batchNormParams)
      preTopology = Sequential[T]().add(preTopology).add(layer)
    }

    if (preTopology != null) {
      modules += preTopology
    }
    modules += topology

    require((preTopology == null && modules.length == 1) ||
      (topology != null && preTopology != null && modules.length == 2),
      "Recurrent extend: should contain only one cell or plus a pre-topology" +
        " to process input")
    this
  }

  private def batchNormalization(batchNormParams: BatchNormParams[T]) = {
    TimeDistributed[T](BatchNormalization[T](
      nOutput = topology.hiddenSizeOfPreTopo,
      batchNormParams.eps,
      batchNormParams.momentum,
      affine = batchNormParams.affine,
      batchNormParams.initWeight,
      batchNormParams.initBias,
      batchNormParams.initGradWeight,
      batchNormParams.initGradBias))
  }

  // list of cell modules cloned from added modules
  protected val cells: ArrayBuffer[Cell[T]]
  = ArrayBuffer[Cell[T]]()

  /**
   * Clone N models; N depends on the time dimension of the input
   * @param sizes, the first element is hiddensize, the left is size of images
   */
  protected def initHidden(sizes: Array[Int]): Unit = {
    val stepShape = sizes
    if (hidden == null) {
      cells.clear()
      cells += topology
      val cell = cells.head

      // The cell will help initialize or resize the hidden variable.
      hidden = cell.hidResize(hidden = null, batchSize = batchSize, stepShape)

      /*
       * Since the gradHidden is only used as an empty Tensor or Table during
       * backward operations. We can reuse the hidden variable by pointing the
       * gradHidden to it.
       */
      gradHidden = hidden
    } else {
      cells.head.hidResize(hidden = hidden, batchSize = batchSize, stepShape)
      gradHidden = hidden
    }
  }

  protected def cloneCells(): Unit = {
    var t = cells.length
    if (t < times) {
      val cloneCell = cells.head.cloneModule()
      cloneCell.parameters()._1.map(_.set())
      cloneCell.parameters()._2.map(_.set())
      // preTopology's output is useless here, clear it.
      // Notice: preTopology is a merge output of all i2h,
      // it's a bigdl tensor, and shouldn't be cloned.
      if (cloneCell.preTopology != null) {
        cloneCell.preTopology.output.set()
      }
      while (t < times) {
        cells += cloneCell.cloneModule()
          .asInstanceOf[Cell[T]]
        outputBuffers.append(Tensor())
        t += 1
      }
      share(cells)
    }
  }

  /**
   * Sharing weights, bias, gradWeights across all the cells in time dim
   * @param cells
   */
  def share(cells: ArrayBuffer[Cell[T]]): Unit = {
    val params = cells.head.parameters()
    cells.foreach(c => {
      if (!c.parameters().eq(params)) {
        var i = 0
        while (i < c.parameters()._1.length) {
          c.parameters()._1(i).set(params._1(i))
          i += 1
        }
        i = 0
        while (i < c.parameters()._2.length) {
          c.parameters()._2(i).set(params._2(i))
          i += 1
        }

        dropouts.append(findDropouts(c))
      }
    })

    val stepLength = dropouts.length
    for (i <- dropouts.head.indices) {
      val head = dropouts.head(i)
      val noise = head.noise
      for (j <- 1 until stepLength) {
        val current = dropouts(j)(i)
        current.noise = noise
        current.isResampling = false
      }
    }
  }

  def findDropouts(cell: Cell[T]): Array[Dropout[T]] = {
    var result: Array[Dropout[T]] = null
    cell.cell match {
      case container: Container[_, _, T] =>
        result = container
          .findModules("Dropout")
          .toArray
          .map(_.asInstanceOf[Dropout[T]])
      case _ =>
    }

    result
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim == 3 || input.dim == 5 || input.dim == 6,
      "Recurrent: input should be a 3D/5D/6D Tensor, e.g [batch, times, nDim], " +
        s"current input.dim = ${input.dim}")

    batchSize = input.size(batchDim)
    times = input.size(timeDim)

    input2Cell = if (preTopology != null) {
      preTopology.forward(input).toTensor[T]
    } else {
      input
    }

    val hiddenSize = topology.hiddensShape(0)
    val outputSize = input.size()
    outputSize(2) = hiddenSize
    output.resize(outputSize)

    /**
     * currentInput forms a T() type. It contains two elements, hidden and input.
     * Each time it will feed the cell with T(hidden, input) (or T(input, hidden) depends on
     * your hidDim and inputDim), and the cell will give a table output containing two
     * identical elements T(output, output). One of the elements from the cell output is
     * the updated hidden. Thus the currentInput will update its hidden element with this output.
     */
    var i = 1
    // Clone N modules along the sequence dimension.
    initHidden(outputSize.drop(2))
    cloneCells()
    if (maskZero) {
      require(input.dim == 3,
        "If maskZero set to true, input should be a 3D Tensor, e.g [batch, times, nDim]")
      inputBuffer.resizeAs(input).abs(input).max(maskBuffer, indexBuffer, 3)
      minLength = ev.toType[Int](maskBuffer.sign().sum(2).min(1)._1(Array(1, 1, 1)))
    }

    currentInput(hidDim) = if (initHiddenState != null) initHiddenState
    else hidden

    while (i <= times) {
      currentInput(inputDim) = Recurrent.selectCopy(input2Cell, i, stepInput2CellBuf)
      cells(i - 1).forward(currentInput)
      val curOutput = cells(i - 1).output
      if (maskZero && i > minLength) {
        val curMask = maskBuffer.select(2, i)
        val curOut = curOutput[Table](hidDim)[Tensor[T]](1)
        // Copy output to a new new tensor as output, because for some cells
        // such as LSTM the hidden h and ouput o refer to the same tensor.
        // But in this case, we want h and o have difference values.
        curOutput.update(inputDim, outputBuffers(i - 1).resizeAs(curOut).copy(curOut))
        for (b <- 1 to curMask.size(1)) {
          if (curMask(Array(b, 1)) == ev.zero) {
            val newState = curOutput[Table](hidDim)
            val originState = currentInput[Table](hidDim)
            for (j <- 1 to newState.length()) {
              newState[Tensor[T]](j).select(1, b).copy(originState[Tensor[T]](j).select(1, b))
            }
            curOutput[Tensor[T]](inputDim).select(1, b).zero()
          }
        }
      }
      currentInput(hidDim) = curOutput[Table](hidDim)
      i += 1
    }

    Recurrent.copy(cells.map(x => x.output.toTable[Tensor[T]](inputDim)),
      output)
    output
  }

  // get hidden state at the last time step
  def getHiddenState(): Activity = {
    require(cells != null && cells(times - 1).output != null,
      "getHiddenState need to be called after updateOutput")
    cells(times - 1).output.toTable(hidDim)
  }

  // set hidden state at the first time step
  protected var initHiddenState: Activity = null
  def setHiddenState(hiddenState: Activity): Unit = {
    initHiddenState = hiddenState
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    currentGradOutput(hidDim) = gradHidden
    /**
     * Since we clone module along the time dimension, the output of each
     * iteration have been recorded by the cloned modules. Thus, we can
     * reuse these outputs during the backward operations by copying the
     * outputs to _input variable.
     *
     * The output of Cell(i-1) should be one of the elements fed to the inputs
     * of Cell(i)
     * The first module in the cells array accepts zero hidden parameter.
     */

    var i = times
    while (i >= 1) {
      currentGradOutput(inputDim) = Recurrent.selectCopy(gradOutput, i, stepGradBuffer)
      _input(hidDim) = if (i > 1) cells(i - 2).output.toTable(hidDim)
      else if (initHiddenState == null) hidden else initHiddenState
      _input(inputDim) = Recurrent.selectCopy(input2Cell, i, stepInput2CellBuf)

      if (i == 1) {
        cells(i - 1).regluarized(true)
      } else {
        cells(i - 1).regluarized(false)
      }

      if (maskZero && i > minLength) {
        val curMask = maskBuffer.select(2, i)
        if (gradOutputBuff.length() == 0) {
          Utils.recursiveResizeAs(gradOutputBuff, currentGradOutput)
        }
        Utils.recursiveCopy(gradOutputBuff, currentGradOutput)
        for (b <- 1 to curMask.size(1)) {
          if (curMask(Array(b, 1)) == ev.zero) {
            val originState = gradOutputBuff[Table](Recurrent.hidDim)
            for (j <- 1 to originState.length()) {
              originState[Tensor[T]](j).select(1, b).zero()
            }
          }
        }

        cells(i - 1).accGradParameters(_input, currentGradOutput)

        for (b <- 1 to curMask.size(1)) {
          if (curMask(Array(b, 1)) == ev.zero) {
            val newState = cells(i - 1).gradInput[Table](hidDim)
            val originState = currentGradOutput[Table](hidDim)
            for (j <- 1 to newState.length()) {
              newState[Tensor[T]](j).select(1, b).copy(originState[Tensor[T]](j).select(1, b))
            }
          }
        }
      } else {
        cells(i - 1).accGradParameters(_input, currentGradOutput)
      }
      currentGradOutput(hidDim) = cells(i - 1).gradInput.toTable(hidDim)
      i -= 1
    }
    if (preTopology != null) {
      preTopology.accGradParameters(input, gradInput2Cell)
    }
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {

    gradInput = if (preTopology != null) {
      /**
       * if preTopology is Sequential, it has not created gradInput.
       * Thus, it needs to create a new Tensor.
       */
      if (preTopology.gradInput == null) {
        preTopology.gradInput = Tensor[T]()
      }
      preTopology.gradInput.toTensor[T]
    } else {
      gradInput2Cell
    }
    gradInput2Cell.resizeAs(input2Cell)
    currentGradOutput(hidDim) = gradHidden
    var i = times
    while (i >= 1) {
      currentGradOutput(inputDim) = Recurrent.selectCopy(gradOutput, i, stepGradBuffer)
      _input(hidDim) = if (i > 1) cells(i - 2).output.toTable(hidDim)
        else if (initHiddenState == null) hidden else initHiddenState
      _input(inputDim) = Recurrent.selectCopy(input2Cell, i, stepInput2CellBuf)

      if (maskZero && i > minLength) {
        val curMask = maskBuffer.select(2, i)
        if (gradOutputBuff.length() == 0) {
          Utils.recursiveResizeAs(gradOutputBuff, currentGradOutput)
        }
        Utils.recursiveCopy(gradOutputBuff, currentGradOutput)
        for (b <- 1 to curMask.size(1)) {
          if (curMask(Array(b, 1)) == ev.zero) {
            val originState = gradOutputBuff[Table](Recurrent.hidDim)
            for (j <- 1 to originState.length()) {
              originState[Tensor[T]](j).select(1, b).zero()
            }
          }
        }

        cells(i - 1).updateGradInput(_input, currentGradOutput)

        for (b <- 1 to curMask.size(1)) {
          if (curMask(Array(b, 1)) == ev.zero) {
            val newState = cells(i - 1).gradInput[Table](hidDim)
            val originState = currentGradOutput[Table](hidDim)
            for (j <- 1 to newState.length()) {
              newState[Tensor[T]](j).select(1, b).copy(originState[Tensor[T]](j).select(1, b))
            }
          }
        }
      } else {
        cells(i - 1).updateGradInput(_input, currentGradOutput)
      }
      currentGradOutput(hidDim) = cells(i - 1).gradInput.toTable(hidDim)
      i -= 1
    }
    Recurrent.copy(cells.map(x => x.gradInput.toTable[Tensor[T]](inputDim)), gradInput2Cell)
    if (preTopology != null) {
      gradInput = preTopology.updateGradInput(input, gradInput2Cell).toTensor[T]
    }
    gradInput
  }

  override def backward(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val before = System.nanoTime
    currentGradOutput(hidDim) = gradHidden
    var i = times

    while (i >= 1) {
      currentGradOutput(inputDim) = Recurrent.selectCopy(gradOutput, i, stepGradBuffer)

      _input(hidDim) = if (i > 1) cells(i - 2).output.toTable(hidDim)
      else if (initHiddenState == null) hidden else initHiddenState

      _input(inputDim) = Recurrent.selectCopy(input2Cell, i, stepInput2CellBuf)

      if (i == 1) {
        cells(i - 1).regluarized(true)
      } else {
        cells(i - 1).regluarized(false)
      }

      if (maskZero && i > minLength) {
        val curMask = maskBuffer.select(2, i)
        if (gradOutputBuff.length() == 0) {
          Utils.recursiveResizeAs(gradOutputBuff, currentGradOutput)
        }
        Utils.recursiveCopy(gradOutputBuff, currentGradOutput)
        for (b <- 1 to curMask.size(1)) {
          if (curMask(Array(b, 1)) == ev.zero) {
            val originState = gradOutputBuff[Table](Recurrent.hidDim)
            for (j <- 1 to originState.length()) {
              originState[Tensor[T]](j).select(1, b).zero()
            }
          }
        }

        cells(i - 1).backward(_input, gradOutputBuff).toTable

        for (b <- 1 to curMask.size(1)) {
          if (curMask(Array(b, 1)) == ev.zero) {
            val newState = cells(i - 1).gradInput[Table](hidDim)
            val originState = currentGradOutput[Table](hidDim)
            for (j <- 1 to newState.length()) {
              newState[Tensor[T]](j).select(1, b).copy(originState[Tensor[T]](j).select(1, b))
            }
          }
        }
      } else {
        cells(i - 1).backward(_input, currentGradOutput)
      }
      currentGradOutput(hidDim) = cells(i - 1).gradInput.toTable(hidDim)
      i -= 1
    }

    gradInput = if (preTopology != null) {
      /**
       * if preTopology is Sequential, it has not created gradInput.
       * Thus, it needs to create a new Tensor.
       */
      if (preTopology.gradInput == null) {
        preTopology.gradInput = Tensor[T]()
      }
      preTopology.gradInput.toTensor[T]
    } else {
      gradInput2Cell
    }
    gradInput2Cell.resizeAs(input2Cell)
    Recurrent.copy(cells.map(x => x.gradInput.toTable[Tensor[T]](inputDim)), gradInput2Cell)

    if (preTopology != null) {
      gradInput = preTopology.backward(input, gradInput2Cell).toTensor[T]
    }

    this.backwardTime += System.nanoTime - before
    gradInput
  }

  override def getTimes(): Array[(AbstractModule[_ <: Activity, _ <: Activity, T], Long, Long)] = {
    val timeBuffer =
      new ArrayBuffer[(AbstractModule[_ <: Activity, _ <: Activity, T], Long, Long)]

    if (!cells.isEmpty) {
      timeBuffer.append(
        cells.flatMap(_.getTimes()).reduce((a, b) => (a._1, a._2 + b._2, a._3 + b._3)))
    }

    if (preTopology != null) {
      timeBuffer.appendAll(preTopology.getTimes())
    }

    val (bufferForward, bufferBackward) =
      timeBuffer.map(t => (t._2, t._3)).reduce((a, b) => (a._1 + b._1, a._2 + b._2))
    timeBuffer.append(
      (this,
        forwardTime - bufferForward,
        backwardTime - bufferBackward))
    timeBuffer.toArray
  }

  override def resetTimes(): Unit = {
    super.resetTimes()
    if (preTopology != null) {
      preTopology.resetTimes
    }
    cells.foreach(_.resetTimes())
  }

  override def clearState() : this.type = {
    super.clearState()
    hidden = null
    gradHidden = null
    hiddenShape = null
    gradInput2Cell.set()
    input2Cell.set()
    currentInput.clear()
    currentGradOutput.clear()
    _input.clear()
    cells.foreach(x => x.clearState())
    cells.clear()
    initHiddenState = null
    stepInput2CellBuf.set()
    stepGradBuffer.set()
    maskBuffer.set()
    gradOutputBuff.clear()
    inputBuffer.set()
    indexBuffer.set()
    outputBuffers.clear()
    minLength = 0
    this
  }

  override def reset(): Unit = {
    require((preTopology == null && modules.length == 1) ||
      (topology != null && preTopology != null && modules.length == 2),
      "Recurrent extend: should contain only one cell or plus a pre-topology" +
        " to process input.")
    require(topology.isInstanceOf[Cell[T]],
      "Recurrent: should contain module with Cell type")

    modules.foreach(_.reset())
    cells.clear()
    hidden = null
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[Recurrent[T]]

  override def equals(other: Any): Boolean = other match {
    case that: Recurrent[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        cells == that.cells
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), cells)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }

  override def toString(): String = s"${getPrintName}${modules}"
}

object Recurrent extends ContainerSerializable {

  private val batchDim = 1
  private val timeDim = 2
  val inputDim = 1
  val hidDim = 2

  def apply[@specialized(Float, Double) T: ClassTag](
    batchNormParams: BatchNormParams[T] = null,
    maskZero: Boolean = false
  )
    (implicit ev: TensorNumeric[T]) : Recurrent[T] = {
    new Recurrent[T](batchNormParams, maskZero = maskZero)
  }

  /**
   * set the cells' output and gradInput to recurrent's output and gradInput
   * to decrease the copy expense.
   * Copy src tensor to dst tensor along timeDime, default timeDime 2, batchDim 1
   * @param src
   * @param dst
   */
  private[bigdl] def copy[T: ClassTag](
    src: ArrayBuffer[Tensor[T]], dst: Tensor[T]): Unit = {
    val timeSize = dst.size(timeDim)
    var t = 1
    while (t <= timeSize) {
      copyToIndex(src(t -1), dst, t)
      t += 1
    }
  }

  /**
   * select srcIndex subset of the 2-th dimension from src, and copy to dst
   * @param src
   * @param srcIndex the index of 2-th dimension from src
   * @param dst
   */
  private[bigdl] def selectCopy[T: ClassTag](
    src: Tensor[T], srcIndex: Int, dst: Tensor[T]): Tensor[T] = {
    if (src.isContiguous() && dst.isContiguous()) {
      if ((dst.nElement() == 0) || (dst.nElement() != (src.nElement() / src.size(2)))) {
        dst.resizeAs(src.select(2, srcIndex))
      }

      val batchSize = src.size(batchDim)
      val timeSize = src.size(timeDim)
      val stepSize = src.nElement() / (batchSize * timeSize)

      val srcArr = src.storage().array()
      var srcOffset = src.storageOffset() - 1
      val dstArr = dst.storage().array()
      var dstOffset = dst.storageOffset() - 1

      val recordSize = timeSize * stepSize
      val indexSize = (srcIndex-1) * stepSize

      var b = 0
      while (b < batchSize) {
        System.arraycopy(srcArr, srcOffset + indexSize, dstArr, dstOffset, stepSize)
        srcOffset += recordSize
        dstOffset += stepSize
        b += 1
      }
    } else {
      val output = src.select(2, srcIndex)
      dst.resizeAs(output).copy(output)
    }
    dst
  }

  /**
   * copy src to be dst dstIndex subset of the 2-th dimension
   * @param src
   * @param dst
   * @param dstIndex the index of 2-th dimension from dst
   */
  private[bigdl] def copyToIndex[T: ClassTag](
    src: Tensor[T], dst: Tensor[T], dstIndex: Int): Tensor[T] = {
    if (src.isContiguous() && dst.isContiguous()) {
      val batchSize = dst.size(batchDim)
      val timeSize = dst.size(timeDim)
      val stepSize = dst.nElement() / (batchSize * timeSize)

      val dstArr = dst.storage().array()
      var dstOffset = dst.storageOffset() - 1
      val srcArr = src.storage().array()
      var srcOffset = src.storageOffset() - 1

      val recordSize = timeSize * stepSize
      val indexSize = (dstIndex - 1) * stepSize

      var b = 0
      while (b < batchSize) {
        System.arraycopy(srcArr, srcOffset, dstArr, dstOffset + indexSize, stepSize)
        srcOffset += stepSize
        dstOffset += recordSize
        b += 1
      }
    } else {
      dst.select(2, dstIndex).copy(src)
    }
    dst
  }

  override def doLoadModule[T: ClassTag](context : DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {

    val attrMap = context.bigdlModule.getAttrMap

    val flag = DataConverter
      .getAttributeValue(context, attrMap.get("bnorm"))
      .asInstanceOf[Boolean]
    val recurrent = if (flag) {
      Recurrent[T](BatchNormParams[T]())
    } else {
      Recurrent[T]()
    }

    val topologyAttr = attrMap.get("topology")
    recurrent.topology = DataConverter.getAttributeValue(context, topologyAttr).
      asInstanceOf[Cell[T]]

    val preTopologyAttr = attrMap.get("preTopology")
    recurrent.preTopology = DataConverter.getAttributeValue(context, preTopologyAttr).
      asInstanceOf[AbstractModule[Activity, Activity, T]]

    if (recurrent.preTopology != null) {
      recurrent.modules.append(recurrent.preTopology)
    }
    recurrent.modules.append(recurrent.topology)

    if (flag) {
      val bnormEpsAttr = attrMap.get("bnormEps")
      recurrent.batchNormParams.eps =
        DataConverter.getAttributeValue(context, bnormEpsAttr)
          .asInstanceOf[Double]

      val bnormMomentumAttr = attrMap.get("bnormMomentum")
      recurrent.batchNormParams.momentum =
        DataConverter.getAttributeValue(context, bnormMomentumAttr)
          .asInstanceOf[Double]

      val bnormInitWeightAttr = attrMap.get("bnormInitWeight")
      recurrent.batchNormParams.initWeight =
        DataConverter.getAttributeValue(context, bnormInitWeightAttr)
          .asInstanceOf[Tensor[T]]

      val bnormInitBiasAttr = attrMap.get("bnormInitBias")
      recurrent.batchNormParams.initBias =
        DataConverter.getAttributeValue(context, bnormInitBiasAttr)
          .asInstanceOf[Tensor[T]]

      val bnormInitGradWeightAttr = attrMap.get("bnormInitGradWeight")
      recurrent.batchNormParams.initGradWeight =
        DataConverter.getAttributeValue(context, bnormInitGradWeightAttr)
          .asInstanceOf[Tensor[T]]

      val bnormInitGradBiasAttr = attrMap.get("bnormInitGradBias")
      recurrent.batchNormParams.initGradBias =
        DataConverter.getAttributeValue(context, bnormInitGradBiasAttr)
          .asInstanceOf[Tensor[T]]

      val bnormAffineAttr = attrMap.get("bnormAffine")
      recurrent.batchNormParams.affine =
        DataConverter.getAttributeValue(context, bnormAffineAttr)
        .asInstanceOf[Boolean]
    }

    recurrent
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
                                            recurrentBuilder : BigDLModule.Builder)
                                           (implicit ev: TensorNumeric[T]) : Unit = {

    val recurrent = context.moduleData.module.asInstanceOf[Recurrent[T]]

    val topologyBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, topologyBuilder, recurrent.topology,
      ModuleSerializer.abstractModuleType)
    recurrentBuilder.putAttr("topology", topologyBuilder.build)

    val preTopologyBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, preTopologyBuilder,
      recurrent.preTopology, ModuleSerializer.abstractModuleType)
    recurrentBuilder.putAttr("preTopology", preTopologyBuilder.build)

    val flag = if (recurrent.batchNormParams != null) {

      val bnormEpsBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context, bnormEpsBuilder,
        recurrent.batchNormParams.eps, universe.typeOf[Double])
      recurrentBuilder.putAttr("bnormEps", bnormEpsBuilder.build)

      val bnormMomentumBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context, bnormMomentumBuilder,
        recurrent.batchNormParams.momentum, universe.typeOf[Double])
      recurrentBuilder.putAttr("bnormMomentum", bnormMomentumBuilder.build)

      val bnormInitWeightBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context, bnormInitWeightBuilder,
        recurrent.batchNormParams.initWeight, ModuleSerializer.tensorType)
      recurrentBuilder.putAttr("bnormInitWeight", bnormInitWeightBuilder.build)

      val bnormInitBiasBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context, bnormInitBiasBuilder,
        recurrent.batchNormParams.initBias, ModuleSerializer.tensorType)
      recurrentBuilder.putAttr("bnormInitBias", bnormInitBiasBuilder.build)

      val bnormInitGradWeightBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context, bnormInitGradWeightBuilder,
        recurrent.batchNormParams.initGradWeight, ModuleSerializer.tensorType)
      recurrentBuilder.putAttr("bnormInitGradWeight", bnormInitGradWeightBuilder.build)

      val bnormInitGradBiasBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context, bnormInitGradBiasBuilder,
        recurrent.batchNormParams.initGradBias, ModuleSerializer.tensorType)
      recurrentBuilder.putAttr("bnormInitGradBias", bnormInitGradBiasBuilder.build)

      val bnormAffineBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context, bnormAffineBuilder,
        recurrent.batchNormParams.affine, universe.typeOf[Boolean])
      recurrentBuilder.putAttr("bnormAffine", bnormAffineBuilder.build)

      true
    } else {
      false
    }

    val bNormBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, bNormBuilder,
      flag, universe.typeOf[Boolean])
    recurrentBuilder.putAttr("bnorm", bNormBuilder.build)

  }
}

case class BatchNormParams[T : ClassTag](
             var eps: Double = 1e-5, // avoid divde zero
             var momentum: Double = 0.1, // momentum for weight update
             var initWeight: Tensor[T] = null,
             var initBias: Tensor[T] = null,
             var initGradWeight: Tensor[T] = null,
             var initGradBias: Tensor[T] = null,
             var affine: Boolean = true)(implicit ev: TensorNumeric[T])
