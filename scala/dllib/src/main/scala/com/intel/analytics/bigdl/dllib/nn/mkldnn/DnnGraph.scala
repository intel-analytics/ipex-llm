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

package com.intel.analytics.bigdl.nn.mkldnn

import java.util

import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat}
import com.intel.analytics.bigdl.nn.tf.{ControlDependency, WithoutInput}
import com.intel.analytics.bigdl.nn.{Graph, mkldnn, MklInt8Convertible}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Node, T}

import scala.collection.mutable


class DnnGraph(
  private val _inputs : Seq[ModuleNode[Float]],
  private val _outputs : Seq[ModuleNode[Float]],
  private val _variables: Option[(Array[Tensor[Float]], Array[Tensor[Float]])] = None,
  private val enableExcludeChecking: Boolean = true)
  extends Graph[Float](_inputs, _outputs, _variables)
  with MklDnnLayer with MklInt8Convertible {
  private val forwardExecution = forwardGraph.topologySort.reverse
  private var backwardExecution: Array[Node[AbstractModule[Activity, Activity, Float]]] = _
  private var inputCache: Array[Activity] = _
  private var backId2ForwardId: Array[Int] = _
  private var skipPrimitiveId = new Array[Boolean](forwardExecution.length)

  /**
   * Batch size may change when model prediction, but output size of dnn layers will not be changed.
   * So we have to compare batchSize for input and output, and do some change if not the same.
   * @param input
   * @param output
   * @return
   */
  private def getRealOutput(input: Activity, output: Activity): Activity = {
    if (input.isTensor && output.isTensor) {
      val in = input.toTensor[Float]
      val out = output.toTensor[Float]
      // for grey image, input should be 3 dims and the first dim should be batch size
      // for non grey image, input should be 4 dims and the first dim should be batch size
      // for rnn model, input should be 2 dims and the first dim should be batch size
      require(in.nDimension() == 4 || in.nDimension() == 3 || in.nDimension() == 2,
        s"only support input with 4 dimension or 3 dimension, but get ${in.nDimension()}")
      if (in.size(1) != out.size(1)) out.narrow(1, 1, in.size(1)) else output
    } else output
  }

  @transient protected lazy val reorderManager = new ReorderManager()

  if (enableExcludeChecking) {
    excludeInvalidLayers(forwardExecution.map {_.element})
  }

  buildBackwardGraph()

  override def updateOutput(input: Activity): Activity = {
    var i = 0
    while(i < forwardExecution.length) {
      val node = forwardExecution(i)
      val nodeInput = if (skipPrimitiveId(i)) {
        findInput(node, input)
      } else {
        findDnnInput(node, input)
      }
      inputCache(i) = nodeInput
      val output = node.element.forward(nodeInput)
      // resize to heap size
      if (!skipPrimitiveId(i) && output.isTensor &&
        !node.element.isInstanceOf[BlasWrapper]) {
        output.toTensor[Float].resize(
          node.element.asInstanceOf[MklDnnLayer].outputFormats()(0).getHeapShape())
      }
      i += 1
    }
    output = getRealOutput(input, dummyOutput.element.output)
    output
  }

  override def backward(input: Activity, gradOutput: Activity): Activity = {
    val before = System.nanoTime()
    val gradients = updateGradInput(input, gradOutput)
    accGradParameters(input, gradOutput)
    backwardTime += System.nanoTime() - before
    gradients
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    dummyOutputGrad.element.gradInput = gradOutput
    var i = 0
    while (i < backwardExecution.length - 1) { // do not execute the dummy backward end
      val curNode = backwardExecution(i)
      val curGradOutput = findDnnGradOutput(curNode, gradOutput)
      // use input from forward
      val curInput = inputCache(backId2ForwardId(i))
      if (!isStopGradient(curNode.element)) {
        val gradInput = curNode.element.updateGradInput(curInput, curGradOutput)
        // resize to heap size
        if (!skipPrimitiveId(i) && gradInput.isTensor &&
          !curNode.element.isInstanceOf[BlasWrapper]) {
          gradInput.toTensor[Float].resize(
            curNode.element.asInstanceOf[MklDnnLayer].gradInputFormats()(0).getHeapShape())
        }
      }
      i += 1
    }
    gradInput = getRealOutput(input, fetchModelGradInput())
    gradInput
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    var i = 0
    while (i < backwardExecution.length - 1) {
      val curNode = backwardExecution(i)
      // use input from forward
      val curInput = inputCache(backId2ForwardId(i))
      val curGradOutput = findDnnGradOutput(curNode, gradOutput, true)
      curNode.element.accGradParameters(curInput, curGradOutput)
      curNode.element.asyncGradient()
      i += 1
    }
  }

  override def buildBackwardGraph(): this.type = {
    super.buildBackwardGraph()
    inputCache = new Array[Activity](forwardExecution.length)
    backwardExecution = backwardGraph.topologySort.reverse
    backId2ForwardId = new Array[Int](backwardExecution.length)

    var i = 0
    // do not execute the dummy backward end
    while(i < backwardExecution.length - 1) {
      var j = 0
      var find = false
      while(j < forwardExecution.length) {
        if (forwardExecution(j).element.getName() == backwardExecution(i).element.getName()) {
          val e = forwardExecution(j).element
          // when creating graph, there may add nn.Identity node,
          // here we have to change it to mkldnn node
          if (e.isInstanceOf[nn.Identity[Float]]) {
            forwardExecution(j).element = toDnnIdentity(e.asInstanceOf[nn.Identity[Float]])
            backwardExecution(i).element = forwardExecution(j).element
          } else {
            require(e.isInstanceOf[MklDnnModule], s"DnnGraph should only contain dnn layers," +
                s"but find ${forwardExecution(j).element.getName()} is not a mkldnn layer")
          }
          backId2ForwardId(i) = j
          find = true
        }
        j += 1
      }
      require(find, "Cannot find backward layer in forward executions")
      i += 1
    }
    this
  }

  /**
   * When doing inference, we may not have to compute forward primitives for some blas layers
   */
  private def skipInitFwdPrimitives() : Unit = {
    val skipNodesMap = new mutable.HashMap[String, Boolean]()
    util.Arrays.fill(skipPrimitiveId, 0, skipPrimitiveId.length, false)
    if (!this.train) {
      var i = forwardExecution.length - 1
      while (i >= 0) {
        val node = forwardExecution(i)
        skipPrimitiveId(i) = skip(node, skipNodesMap)
        skipNodesMap(node.element.getName()) = skipPrimitiveId(i)
        i -= 1
      }
    }
  }

  /**
   * to determine whether to skip computing primitives for current node
   * Now, if current node is blaswrapper node and meets one of following cases,
   * then we will skip computing primitives for this node
   * case 1: it has no next nodes
   * case 2: all next nodes are identity node, and those next nodes has no next nodes
   * case 3: all next nodes are also skip nodes
   * In some special case, if previous nodes are not blas node, we can not skip this node,
   * but don't have to compute its output shape.
   * @param node current node
   * @return
   */
  private def skip(node: ModuleNode[Float],
                   skipNodesMap: mutable.HashMap[String, Boolean]) : Boolean = {
    if (node.element.isInstanceOf[BlasWrapper] || node.element.isInstanceOf[Identity]) {
      if (node.nextNodes.length == 0) return true
      var isSkip : Boolean = true
      node.nextNodes.map(n => {
        if ((skipNodesMap.getOrElse(n.element.getName(), false))
          || (n.element.isInstanceOf[mkldnn.Identity] && n.nextNodes.length == 0)) {
        } else isSkip = false
      })
      node.prevNodes.map(n =>
        if (!n.element.isInstanceOf[BlasWrapper]
          && node.element.isInstanceOf[BlasWrapper] && isSkip) {
          node.element.asInstanceOf[BlasWrapper].needOutputFormats = false
         isSkip = false
        }
      )
      isSkip
    } else false
  }

  // change nn identity to mkldnn identity
  private def toDnnIdentity(model: nn.Identity[Float])
  : AbstractModule[Activity, Activity, Float] = {
    mkldnn.Identity[Float]().setName(model.getName())
      .asInstanceOf[AbstractModule[Activity, Activity, Float]]
  }

  // if node has no previous node, then it will just use graph input as real module input
  private def findDnnInput(node: ModuleNode[Float], input: Activity): Activity = {
    if (node.element.isInstanceOf[WithoutInput]) return null

    val realInputFormats = node.element.asInstanceOf[MklDnnModule].inputFormats()

    val nodeInput = if (node.prevNodes.isEmpty) {
      getInput(node, input)
    } else {
      val prevActivitiesAndFormats = node.prevNodesAndEdges
        .filterNot(n => n._1.element.isInstanceOf[ControlDependency[Float]])
        .map(n => {
          val format = n._1.element.asInstanceOf[MklDnnModule].outputFormats()
          n._2.fromIndex match {
            case Some(i) =>
              if (n._1.element.output == null || (i == 1 && n._1.element.output.isTensor)) {
                (n._1.element.output, format)
              } else {
                (n._1.element.output.toTable.apply[Activity](i), Array(format(i - 1)))
              }
            case None => (n._1.element.output, format)
          }
        })

      val inputAndFormat = if (prevActivitiesAndFormats.length == 1) {
        prevActivitiesAndFormats.head
      } else {
        (T.seq(prevActivitiesAndFormats.map(m => m._1)),
          prevActivitiesAndFormats.map(m => m._2).toArray.flatMap(_.toSeq))
      }
      reorderManager.infer(inputAndFormat._2, realInputFormats, inputAndFormat._1)
    }
    nodeInput
  }

  private def findDnnGradOutput(curNode: ModuleNode[Float], gradOutput: Activity,
    isAcc: Boolean = false): Activity = {
    var curGradOutput : Activity = if (curNode.eq(dummyOutputGrad)) gradOutput else null

    val realGradOutputFormats = if (isAcc) {
      curNode.element.asInstanceOf[MklDnnModule].gradOutputWeightFormats()
    } else {
      curNode.element.asInstanceOf[MklDnnModule].gradOutputFormats()
    }

    curNode.prevNodesAndEdges.filterNot(n => n._1.element.isInstanceOf[ControlDependency[Float]])
      .foreach(n => {
        val (otherActivity, format) =
          if (n._1.element.gradInput.isTensor || n._1.nextEdges.length == 1) {
            (n._1.element.gradInput, n._1.element.asInstanceOf[MklDnnModule].gradInputFormats())
          } else {
            val index = n._1.nextEdges.indexOf(n._2) + 1
            (n._1.element.gradInput.toTable.apply[Activity](index),
              Array(n._1.element.asInstanceOf[MklDnnModule].gradInputFormats().apply(index - 1)))
          }

        n._2.fromIndex match {
          case Some(i) =>
            if (i == 1 && curNode.element.output.isTensor) {
              curGradOutput = addActivity(curGradOutput, realGradOutputFormats,
                otherActivity, format)
            } else {
              if (curNode.element.output.isTable && curGradOutput == null) {
                curGradOutput = T()
              }
              val curActivity = curGradOutput.toTable.getOrElse[Activity](i, null)
              curGradOutput.toTable(i) = addActivity(curActivity, realGradOutputFormats,
                otherActivity, format)
            }
          case None =>
            curGradOutput = addActivity(curGradOutput, realGradOutputFormats,
              otherActivity, format)
        }
      })

    if (curNode.element.output.isTable) {
      addZeroTensorToMissingGradOutput(curNode.element.output.toTable, curGradOutput.toTable)
    }
    curGradOutput
  }

  private def addActivity(activity: Activity, realFormats: Array[MemoryData],
    other: Activity, otherFormats: Array[MemoryData]): Activity = {
    val realOthers = if (otherFormats.length > 0) {
      reorderManager.infer(otherFormats, realFormats, other)
    } else {
      other
    }
    super.accActivity(activity, realOthers)
  }

  final def compile(phase: Phase) : Unit = {
    setRuntime(new MklDnnRuntime())
    initPrimitives(phase, Array[MemoryData]())
  }

  override def setRuntime(runtime: MklDnnRuntime): Unit = {
    this.runtime = runtime
    reorderManager.setRuntime(runtime)
    forwardExecution.foreach(m => m.element.asInstanceOf[MklDnnModule].setRuntime(runtime))
  }

  private def initPrimitives(phase: Phase, inputFormats: Array[MemoryData]): Unit = {
    _outputFormats = initFwdPrimitives(inputFormats, phase)._2
    if (phase == Phase.TrainingPhase) {
      _gradOutputFormats = initBwdPrimitives(_outputFormats, phase)._1
      _gradOutputFormatsForWeight = initGradWPrimitives(_outputFormats, phase)
    }
  }

  private def getInputMemoryData(node: ModuleNode[Float], memoryData: Array[MemoryData])
    : Array[MemoryData] = {
    // the model may contain two inputs and all of them is Input.
    if (inputs.length == 1 || memoryData.isEmpty) {
      require(inputs.contains(node), "input node must be in the input list")
      memoryData
    } else {
      val i = inputs.indexOf(node)
      require(i != -1, "input node is not in the input list")
      Array(memoryData(i))
    }
  }

  private def findInputFormats(node: ModuleNode[Float], inputs: Array[MemoryData])
    : Array[MemoryData] = {
    if (node.prevNodes.isEmpty) {
      getInputMemoryData(node, inputs)
    } else {
      val prevFormats = node.prevNodesAndEdges
        .filterNot(n => n._1.element.isInstanceOf[ControlDependency[Float]])
        .map(n => {
          val outputFormats = n._1.element.asInstanceOf[MklDnnModule].outputFormats()
          // if outputFormats length is 1, output is a tensor
          n._2.fromIndex match {
            case Some(i) =>
              if (n._1.element.output == null || (i == 1 && outputFormats.length == 1)) {
                outputFormats
              } else {
                val index = n._2.fromIndex.get
                Array(outputFormats(index))
              }
            case None => outputFormats
          }
        }).toArray
      prevFormats.flatMap(n => n.toSeq)
    }
  }

  private def findGradOutputFormats(node: ModuleNode[Float], inputs: Array[MemoryData])
    : Array[MemoryData] = {
    if (node.prevNodes.isEmpty) {
      inputs
    } else {
      val prevFormats = node.prevNodesAndEdges
        .filterNot(n => n._1.element.isInstanceOf[ControlDependency[Float]])
        .map(n => {
          // gradInput is tensor or nextEdges number is 1
          if (n._1.element.asInstanceOf[MklDnnModule].gradInputFormats().length == 1 ||
            n._1.nextEdges.length == 1) {
            n._1.element.asInstanceOf[MklDnnModule].gradInputFormats()
          } else {
            val index = n._1.nextEdges.indexOf(n._2)
            val f = n._1.element.asInstanceOf[MklDnnModule].gradInputFormats()
            Array(f(index))
          }
        }).toArray
      // reminder: if node has more than one previous node, use the first one as gradoutput format
      prevFormats(0)
    }
  }

  /**
   * fuse some layers when doing inference
   * first fuse layers in sequence, mainly relu with bn/conv, conv with bn.
   * after that, fuse sum operation.
   */
  private def fusion(): Unit = {
    if (!this.train) {
      for (j <- 0 to 4) {
        var i = forwardExecution.length - 1
        while (i >= 0) {
          if (j == 0) Fusion.fuseScale(forwardExecution(i))
          if (j == 1) Fusion.fuseModule(forwardExecution(i))
          // we should do this before sum fusion, because it will change the structure of graph
          if (j == 2) Fusion.setNegativeInputOfConv(forwardExecution(i))
          if (j == 3) Fusion.fuseCAdd(forwardExecution(i))
          if (j == 4) Fusion.setScalesPrevousJoinTable(forwardExecution(i))
          i -= 1
        }
      }
    }
  }

  private def getHeapFormat(inputs: Array[MemoryData]): Int = {
    var heapFormat: Int = -1
    inputs.foreach(m => {
      if (m.shape.length == 4) {
        return inputs(0).layout
      }
    })

    @inline
    def transferFormat(format: DataFormat): Int = {
      if (format == DataFormat.NHWC) Memory.Format.nhwc else Memory.Format.nchw
    }

    for (i <- 0 until forwardExecution.length) {
      val m = forwardExecution(i).element
      val format = m match {
        case conv: mkldnn.SpatialConvolution => transferFormat(conv.format)
        case maxPool: mkldnn.MaxPooling => transferFormat(maxPool.format)
        case avgPool: mkldnn.AvgPooling => transferFormat(avgPool.format)
        case sbn: mkldnn.SpatialBatchNormalization => transferFormat(sbn.format)
        case lrn: mkldnn.LRN => transferFormat(lrn.format)
        case _ => -1
      }

      if (heapFormat == -1) {
        heapFormat = format
      } else if (format != -1) {
        require(heapFormat == format,
          s"layer ${m} should use format ${heapFormat}, but get ${format}")
      }
    }
    if (heapFormat == -1) Memory.Format.nchw else heapFormat
  }

  // init forward primitives
  override private[bigdl] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase)
    : (Array[MemoryData], Array[MemoryData]) = {
    skipInitFwdPrimitives()
    fusion()
    var lastOutputFormats = inputs
    var firstRealInputFormats: Array[MemoryData] = null
    val heapFormat : Int = getHeapFormat(inputs)
    for (i <- 0 until forwardExecution.length) {
      if (!skipPrimitiveId(i)) {
        val m = forwardExecution(i)
        lastOutputFormats = findInputFormats(m, inputs)
        val (realInputFormats, realOutputFormats) =
          m.element.asInstanceOf[MklDnnModule].initFwdPrimitives(lastOutputFormats, phase)
        if (realOutputFormats != null) realOutputFormats.foreach(_.setHeapFormat(heapFormat))
        lastOutputFormats.zip(realInputFormats).foreach {
          case (o, i) =>
            Utils.copyMaskAndScales(o, i)
            reorderManager.register(o, i)
        }

        // copy the scales from the input formats to output formats, for some layers,
        // it will not copy the mask and scales automatically or generate the scales themselves
        Utils.copyMaskAndScales(realInputFormats, realOutputFormats)

        if (i == 0) firstRealInputFormats = realInputFormats
      }
    }
    _inputFormats = firstRealInputFormats
    _outputFormats = lastOutputFormats
    (firstRealInputFormats, lastOutputFormats)
  }

  // init updateGradInput primitives
  override private[bigdl] def initBwdPrimitives(grads: Array[MemoryData], phase: Phase)
    : (Array[MemoryData], Array[MemoryData]) = {
    var lastGradInputFormats = grads
    var firstRealGradOutputFormats: Array[MemoryData] = null
    for (i <- 0 until backwardExecution.length - 1) {
      val m = backwardExecution(i)
      lastGradInputFormats = findGradOutputFormats(m, grads)
      val realGradOutputAndInputFomrats =
        m.element.asInstanceOf[MklDnnModule].initBwdPrimitives(lastGradInputFormats, phase)
      lastGradInputFormats.zip(realGradOutputAndInputFomrats._1).foreach {
        case (gi, go) => reorderManager.register(gi, go)
      }
      if (i == 0) firstRealGradOutputFormats = realGradOutputAndInputFomrats._1
    }
    _gradOutputFormats = firstRealGradOutputFormats
    _gradInputFormats = lastGradInputFormats
    (firstRealGradOutputFormats, lastGradInputFormats)
  }

  // init acc primitives
  override private[bigdl] def initGradWPrimitives(grads: Array[MemoryData], phase: Phase)
    : Array[MemoryData] = {
    var lastGradInputFormats = grads
    var firstRealGradOutputFormats: Array[MemoryData] = null
    for (i <- 0 until backwardExecution.length - 1) {
      val m = backwardExecution(i)
      lastGradInputFormats = findGradOutputFormats(m, grads)
      val realGradOutput =
        m.element.asInstanceOf[MklDnnModule].initGradWPrimitives(lastGradInputFormats, phase)
      lastGradInputFormats.zip(realGradOutput).foreach {
        case (gi, go2) => reorderManager.register(gi, go2)
      }
      if (i == 0) firstRealGradOutputFormats = realGradOutput
    }
    _gradOutputFormatsForWeight = firstRealGradOutputFormats
    firstRealGradOutputFormats
  }

  override def populateModules(): Unit = {
    modules.appendAll(
      forwardGraph.topologySort
        // todo: convert control dep node to edge
        .filterNot(_.element.isInstanceOf[ControlDependency[Float]])
        .filter(n => !n.eq(dummyOutput)).map(_.element)
        .reverse
    )
    checkDuplicate()
  }

  override def release(): Unit = {
    // do not call super.release, it will call MklDnnLayer.release()
    modules.foreach(_.release())
    // we need to call releaseResources here because super.release will never be called
    this.releaseResources()
  }

  override def calcScales(input: Activity): Unit = {
    if (input == null) return

    var i = 0
    while(i < forwardExecution.length) {
      val node = forwardExecution(i)
      val nodeInput = if (skipPrimitiveId(i)) {
        findInput(node, input)
      } else {
        findDnnInput(node, input)
      }

      node.element match {
        case convertible: MklInt8Convertible =>
          convertible.calcScales(nodeInput)
        case _ =>
      }
      i += 1
    }
  }

  override def setQuantize(value: Boolean): DnnGraph.this.type = {
    this.forwardExecution.foreach { node =>
      if (node.element.isInstanceOf[MklDnnModule]) {
        node.element.asInstanceOf[MklDnnModule].setQuantize(value)
      }
    }
    this
  }
}

object DnnGraph {
  def apply(
    inputs : Seq[ModuleNode[Float]],
    outputs : Seq[ModuleNode[Float]],
    variables: Option[(Array[Tensor[Float]], Array[Tensor[Float]])] = None,
    enableExcludeChecking: Boolean = true): DnnGraph =
    new DnnGraph(inputs, outputs, variables, enableExcludeChecking)
}
