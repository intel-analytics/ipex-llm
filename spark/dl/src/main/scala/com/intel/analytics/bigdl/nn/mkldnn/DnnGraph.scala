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

import breeze.linalg.Axis._1
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.tf.{ControlDependency, WithoutInput}
import com.intel.analytics.bigdl.nn.{StaticGraph, mkldnn}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{LayerException, Node, T}
import com.intel.analytics.bigdl.nn

import scala.reflect.ClassTag


class DnnGraph(
  private val _inputs : Seq[ModuleNode[Float]],
  private val _outputs : Seq[ModuleNode[Float]],
  private val _variables: Option[(Array[Tensor[Float]], Array[Tensor[Float]])] = None,
  private val enableExcludeChecking: Boolean = true)
  extends StaticGraph[Float](_inputs, _outputs, _variables, enableExcludeChecking)
  with MklDnnLayer {
  private var forwardExecution: Array[Node[AbstractModule[Activity, Activity, Float]]] = _
  private var backwardExecution: Array[Node[AbstractModule[Activity, Activity, Float]]] = _
  private var inputCache: Array[Activity] = _
  private var backId2ForwardId: Array[Int] = _

  @transient protected lazy val reorderManager = new ReorderManager()

  if (enableExcludeChecking) {
    excludeInvalidLayers(forwardExecution.map {_.element})
  }

  buildBackwardGraph()

  override def updateOutput(input: Activity): Activity = {
    var i = 0
    while(i < forwardExecution.length) {
      val node = forwardExecution(i)
      val nodeInput = findDnnInput(node, input)
      inputCache(i) = nodeInput
      node.element.forward(nodeInput)
      i += 1
    }
    output = dummyOutput.element.output
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
        curNode.element.updateGradInput(curInput, curGradOutput)
      }
      i += 1
    }
    gradInput = fetchModelGradInput()
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
    forwardExecution = forwardGraph.topologySort.reverse
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
    if (inputs.length == 1) {
      require(inputs(0).eq(node), "input node is not in the input list")
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

  // init forward primitives
  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase)
    : (Array[MemoryData], Array[MemoryData]) = {
    var lastOutputFormats = inputs
    var firstRealInputFormats: Array[MemoryData] = null
    for (i <- 0 until forwardExecution.length) {
      val m = forwardExecution(i)
      lastOutputFormats = findInputFormats(m, inputs)
      val realInputAndOutputFormats =
        m.element.asInstanceOf[MklDnnModule].initFwdPrimitives(lastOutputFormats, phase)
      lastOutputFormats.zip(realInputAndOutputFormats._1).foreach {
        case (o, i) => reorderManager.register(o, i)
      }
      if (i == 0) firstRealInputFormats = realInputAndOutputFormats._1
    }
    _inputFormats = firstRealInputFormats
    _outputFormats = lastOutputFormats
    (firstRealInputFormats, lastOutputFormats)
  }

  // init updateGradInput primitives
  override private[mkldnn] def initBwdPrimitives(grads: Array[MemoryData], phase: Phase)
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
  override private[mkldnn] def initGradWPrimitives(grads: Array[MemoryData], phase: Phase)
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
}

object DnnGraph {
  def apply(
    inputs : Seq[ModuleNode[Float]],
    outputs : Seq[ModuleNode[Float]],
    variables: Option[(Array[Tensor[Float]], Array[Tensor[Float]])] = None,
    enableExcludeChecking: Boolean = true): DnnGraph =
    new DnnGraph(inputs, outputs, variables, enableExcludeChecking)
}
