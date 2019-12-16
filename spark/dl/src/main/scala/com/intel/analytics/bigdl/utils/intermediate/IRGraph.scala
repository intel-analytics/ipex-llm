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

package com.intel.analytics.bigdl.utils.intermediate

import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn.{Graph, SpatialMaxPooling, keras}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat}
import com.intel.analytics.bigdl.nn.mkldnn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils._

import scala.reflect.ClassTag

/**
 * Generate IR graph
 * @param inputs input nodes for graph
 * @param outputs output nodes for graph
 * @param variables
 * @param generateBackward
 * @param inputFormats input memory layout for graph
 * @param outputFormats output memory layout for graph
 * @param ev$1
 * @param ev
 * @tparam T The numeric type in this module parameters.
 */
private[bigdl] class IRGraph[T: ClassTag](
    val inputs : Seq[Node[IRElement[T]]],
    val outputs : Seq[Node[IRElement[T]]],
    val variables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None,
    val generateBackward: Boolean = true,
    val inputFormats: Seq[Int] = Seq(Memory.Format.nchw),
    val outputFormats: Seq[Int] = Seq(Memory.Format.nc))
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Activity, Activity, T] with Serializable {

  @transient private var initPrim: Boolean = false

  require(inputFormats.length == inputs.length, s"IRGraph: inputFormats" +
    s"length ${inputFormats.length} should be same with input nodes length ${inputs.length}")
  require(outputFormats.length == outputs.length, s"IRGraph: outputFormats" +
    s"length ${outputFormats.length} should be same with output nodes length ${outputs.length}")

  private[bigdl] var graph: Graph[T] = null

  private[bigdl] def isBuild(): Boolean = graph != null

  override def updateOutput(input: Activity): Activity = {
    if (graph == null) {
      throw new UnsupportedOperationException("forward not supported, Please build graph first")
    }
    if (graph.isInstanceOf[DnnGraph]) {
      // if using multi MKL-DNN model, we just use current thread directly
      // because it's in sequential mode of MKL and MKL-DNN
      if (Engine.isMultiModels) {
        initPrimitives(input)
        graph.updateOutput(input)
      } else {
        Engine.dnnComputing.invokeAndWait2(Array(0).map(_ => () => {
          initPrimitives(input)
          graph.updateOutput(input)
        }))
      }
    } else graph.updateOutput(input)
    output = graph.output
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    if (graph == null) {
      throw new UnsupportedOperationException("backward not supported, Please build graph first")
    }
    if (graph.isInstanceOf[DnnGraph]) {
      Engine.dnnComputing.invokeAndWait2(Array(0).map(_ => () => {
        graph.updateGradInput(input, gradOutput)
      }))
    } else graph.updateGradInput(input, gradOutput)
    gradInput = graph.gradInput
    gradInput
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    if (graph == null) {
      throw new UnsupportedOperationException("backward not supported, Please build graph first")
    }
    if (graph.isInstanceOf[DnnGraph]) {
      Engine.dnnComputing.invokeAndWait2(Array(0).map(_ => () => {
        graph.accGradParameters(input, gradOutput)
      }))
    } else graph.accGradParameters(input, gradOutput)
  }

  def build(): this.type = {
    graph = new IRConverter[T](this).toGraph()
    this
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    graph.parameters()
  }

  override def getParametersTable(): Table = graph.getParametersTable()

  override def training(): this.type = {
    train = true
    graph.training()
    this
  }

  /**
   * Set the module to evaluate mode
   * @return
   */
  override def evaluate(): this.type = {
    train = false
    graph.evaluate()
    this
  }

  override def getExtraParameter(): Array[Tensor[T]] = {
    graph.getExtraParameter()
  }

  override def getTimes(): Array[(AbstractModule[_ <: Activity, _ <: Activity, T], Long, Long)] = {
    graph.getTimes()
  }

  override def resetTimes(): Unit = {
    graph.resetTimes()
  }

  private def initPrimitives(input: Activity): Unit = {
    if (!initPrim && graph.isInstanceOf[DnnGraph]) {
      val inputMemory = new Array[MemoryData](inputFormats.length)
      if (input.isInstanceOf[Tensor[T]]) {
        // todo: handle for 3 dimensions, expand 3 dims to 4 dims
        val size = input.toTensor[T].size()
        val sizeNew = if (size.length == 3 && inputFormats(0) != Memory.Format.ntc
          && inputFormats(0) != Memory.Format.tnc) {
          Array(size(0), 1, size(1), size(2))
        } else if (inputFormats(0) == Memory.Format.nhwc) {
          // always use NCHW to create heap data
          Array(size(0), size(3), size(1), size(2))
        } else size
        inputMemory(0) = HeapData(sizeNew, inputFormats(0))
      } else {
        val tensors = input.toTable
        require(tensors.length() == inputFormats.length, s"table input length " +
          s"${tensors.length()} should be the same with inputFormats length ${inputFormats.length}")
        tensors.foreach(t => {
          require(t._2.isInstanceOf[Tensor[T]],
            "Only support input with tensor type, table not supported")
          val t1 = t._1.asInstanceOf[Int] // starts from 1
          val t2 = t._2.asInstanceOf[Tensor[T]]
          if (inputFormats(t1 - 1 ) == Memory.Format.nhwc) {
            val sizeNew = Array(t2.size(1), t2.size(4), t2.size(2), t2.size(3))
            inputMemory(t1 - 1) = HeapData(sizeNew, inputFormats(t1 - 1))
          } else {
            inputMemory(t1 - 1) = HeapData(t2.size(), inputFormats(t1 - 1))
          }
        })
      }
      val dnnGraph = graph.asInstanceOf[DnnGraph]
      val phase = if (dnnGraph.isTraining()) Phase.TrainingPhase else Phase.InferencePhase
      dnnGraph.setRuntime(new MklDnnRuntime())
      dnnGraph.initFwdPrimitives(inputMemory, phase)
      if (dnnGraph.isTraining()) {
        dnnGraph.initBwdPrimitives(dnnGraph.outputFormats(), phase)
        dnnGraph.initGradWPrimitives(dnnGraph.outputFormats(), phase)
      }
      initPrim = true
    }
  }

  def setQuantize(value: Boolean): this.type = {
    require(graph != null, s"you should build the graph first")
    if (graph.isInstanceOf[DnnGraph]) {
      graph.asInstanceOf[DnnGraph].setQuantize(value)
    }
    this
  }

  override def release(): Unit = {
    if (graph.isInstanceOf[DnnGraph]) {
      Engine.dnnComputing.invokeAndWait2(Array(0).map(_ => () => {
        graph.release()
      }))
    }
  }
}

object IRGraph {
  def apply[T: ClassTag](
    inputs: Seq[Node[IRElement[T]]],
    outputs: Seq[Node[IRElement[T]]],
    variables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None,
    generateBackward: Boolean = true,
    inputFormats: Int = Memory.Format.nchw,
    outputFormats: Int = Memory.Format.nc
  )( implicit ev: TensorNumeric[T]): IRGraph[T] = {
    new IRGraph[T](inputs, outputs, variables, generateBackward,
      Seq(inputFormats), Seq(outputFormats))
  }

  def apply[T: ClassTag](
    inputs: Seq[Node[IRElement[T]]],
    outputs: Seq[Node[IRElement[T]]],
    variables: Option[(Array[Tensor[T]], Array[Tensor[T]])],
    generateBackward: Boolean,
    inputFormats: Seq[Int],
    outputFormats: Seq[Int]
  )( implicit ev: TensorNumeric[T]): IRGraph[T] = {
    new IRGraph[T](inputs, outputs, variables, generateBackward, inputFormats, outputFormats)
  }
}
