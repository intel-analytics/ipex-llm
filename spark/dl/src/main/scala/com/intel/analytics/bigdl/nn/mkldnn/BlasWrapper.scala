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
import breeze.linalg.dim
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.MiniBatch
import com.intel.analytics.bigdl.mkl.{MKL, Memory}
import com.intel.analytics.bigdl.nn.{DetectionOutputSSD, PriorBox}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat, TensorModule}
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Engine._
import com.intel.analytics.bigdl.utils.{Util => NNUtils, _}
import org.apache.log4j.Logger

/**
 * wrap blas module to dnn module,
 * and the module should have implemented "computeOutputShape" func.
 * @param module
 */
private[bigdl] class BlasWrapper(val module: AbstractModule[Activity, Activity, Float])
  extends MklDnnLayer {

  require(!module.isInstanceOf[MklDnnModule], "Only support wrapper blas layer to dnn layer")

  output = module.output
  gradInput = module.gradInput

  // reminder: for dim 3, there may be ntc or tnc, now we just support ntc
  private def getFormats(dims: Int): Int = {
    dims match {
      case 4 => Memory.Format.nchw
      case 3 => Memory.Format.ntc
      case 2 => Memory.Format.nc
      case 1 => Memory.Format.x
      case _ => throw new UnsupportedOperationException(s"UnSupported dims ${dims}")
    }
  }

  private def getHeapFormats(in: MemoryData): Int = {
    if (in.heapFormat == -1 || in.shape.length != 4) {
      getFormats(in.shape.length)
    } else in.heapFormat
  }

  private[mkldnn] var needOutputFormats: Boolean = true
  @transient private lazy val logger = Logger.getLogger(getClass)

  @transient private var subModels: Array[Module[Float]] = _
  @transient private var subModelNumber : Int = 1
  @transient private var withMultiThread: Boolean = false
  @transient private var inputBuffer : Array[Activity] = _
  @transient private var tensorBuffer : Array[Tensor[Float]] = _
  @transient private var batchSize : Int = _
  @transient private var initEnv: Boolean = false

  private def inferInputFormats(inputs: Array[MemoryData]): Array[MemoryData] = {
    inputs.map(in => {
      val heap = if (in.layout == Memory.Format.tnc) {
        val size = in.shape
        HeapData(Array(size(1), size(0), size(2)), Memory.Format.ntc)
      } else {
        HeapData(in.shape, getHeapFormats(in))
      }
      heap.setHeapFormat(in.heapFormat)
    })
  }

  private def inferOutputFormats(inputs: Array[MemoryData]): Array[MemoryData] = {
    val inputShape = inputs.map(in => Shape(in.getHeapShape()))
    val outputShape = if (inputShape.length == 1) {
      List(module.computeOutputShape(inputShape(0)))
    } else {
      // multi shape
      val out = module.computeOutputShape(MultiShape(inputShape.toList))
      if (out.isInstanceOf[MultiShape]) out.toMulti() else List(out)
    }
    outputShape.map(in => {
      val size = in.toSingle().toArray
      val f = if (size.length == 4 && inputs(0).heapFormat == Memory.Format.nhwc) {
        Memory.Format.nhwc
      } else getFormats(size.length)
      val outSize = if (f == Memory.Format.nhwc) {
        Array(size(0), size(3), size(1), size(2))
      } else size
      HeapData(outSize, f).setHeapFormat(f)
    }).toArray
  }

  /**
   * Blas layers normally do not have competitive performance when running under mkldnn.
   * So we can leverage multi-threading to resolve bottleneck introduced by one model only
   * for mkl-dnn backend. The parallelism is determined by both bath size and core number,
   * with restrictions that both input and output format must be batched.
   */
  private def setMultiThreadEnv(input: Activity): Unit = {
    initEnv = true
    val multiThread = System.getProperty("multiThread", "false").toBoolean
    if (this.train && multiThread) {
      throw new IllegalArgumentException("Please not set multiThread to true for model training")
    }
    if (this.train
      || !multiThread
      || (_outputFormats != null && _outputFormats.length != 1)
      || (_outputFormats != null && _inputFormats != null
      && _inputFormats(0).shape(0) != _outputFormats(0).shape(0))
      || !flattenInput(input)
    ) {
      return
    }
    batchSize = tensorBuffer(0).size(1)
    val residue = batchSize % Engine.coreNumber()
    if (residue != 0 || batchSize < 2 || Engine.coreNumber() < 2) {
      logger.warn("If you want to use multiThread property to speed up, " +
        "please attention core number should be greater than 1, " +
        s"batch size should be greater than 1 and divided by core number, " +
        s"but now get core number ${Engine.coreNumber()} batch size ${batchSize}")
      return
    }
    subModelNumber = Engine.coreNumber()
    initModules()
    withMultiThread = true
  }
  private def flattenInput(input: Activity): Boolean = {
    val inputDepth = if (input.isTensor) 1 else input.toTable.length()
    if (tensorBuffer == null) tensorBuffer = new Array[Tensor[Float]](inputDepth)
    var batch : Int = 0
    if (inputDepth == 1) {
      tensorBuffer(0) = input.toTensor[Float]
    } else {
      val in = input.toTable
      for (i <- 1 to in.length()) {
        if (in.get(i).get.isInstanceOf[Table]) return false
        tensorBuffer(i - 1) = in.get[Tensor[Float]](i).get
        if (i == 1) batch = tensorBuffer(i - 1).size(1)
        // reminder: inputs for DetectionOutputSSD are not all in batch,
        // but the non-batched input can be shared in all batch. So this layer can be paralleled.
        if (batch != tensorBuffer(i - 1).size(1)
          && !module.isInstanceOf[DetectionOutputSSD[Float]]) {
          return false
        }
      }
    }
    true
  }
  private def initModules(): Unit = {
    subModels = if (module.parameters() != null) {
        val wb = NNUtils.getAndClearWeightBias(module.parameters())
        val models = (1 to subModelNumber).map(i => {
          val m = module.cloneModule()
          NNUtils.putWeightBias(wb, m)
          m.asInstanceOf[Module[Float]]
        }).toArray
        NNUtils.putWeightBias(wb, module)
        models
      } else {
        val models = (1 to subModelNumber).map(i => {
          val m = module.cloneModule()
          m.asInstanceOf[Module[Float]]
        }).toArray
        models
      }
  }

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    _inputFormats = inferInputFormats(inputs)
    _outputFormats = if (needOutputFormats) inferOutputFormats(_inputFormats) else null
    if (_outputFormats != null) {
      _outputFormats.map(_.getPrimitive(runtime))
    }
    (_inputFormats, _outputFormats)
  }

  override private[mkldnn] def initBwdPrimitives(grad: Array[MemoryData], phase: Phase) = {
    _gradOutputFormats = _outputFormats
    _gradInputFormats = _inputFormats
    (_gradOutputFormats, _gradInputFormats)
  }

  override private[mkldnn] def initGradWPrimitives(grad: Array[MemoryData], phase: Phase) = {
    _gradOutputFormatsForWeight = _outputFormats
    _gradOutputFormatsForWeight
  }

  private def getInput(dim: Int, index: Int, size: Int): Activity = {
    if (tensorBuffer.length == 1) {
      tensorBuffer(0).narrow(dim, index, size)
    } else {
      // the third tensor of inputs for DetectionOutputSSD is not in batch,
      // but it can be shared with all batch.
      if (module.isInstanceOf[DetectionOutputSSD[Float]]) {
        T(tensorBuffer(0).narrow(dim, index, size),
          tensorBuffer(1).narrow(dim, index, size), tensorBuffer(2))
      } else {
        T.array(tensorBuffer.map(_.narrow(dim, index, size)))
      }
    }
  }
  private def forwardInParallel(input: Activity): Activity = {
    if (inputBuffer == null) inputBuffer = new Array[Activity](subModelNumber)
    val stackSize = batchSize / subModelNumber

    val tasks = Engine.wrapperComputing.invoke((0 until subModelNumber).map(i =>
      () => inputBuffer(i) = getInput(1, i * stackSize + 1, stackSize)))
    Engine.wrapperComputing.sync(tasks)

    val forwardThreads = Engine.wrapperComputing.invoke((0 until subModelNumber).map(i =>
      () => subModels(i).forward(inputBuffer(i)).toTensor[Float]))
    Engine.wrapperComputing.sync(forwardThreads)

    if (subModels(0).output.isTable) {
      withMultiThread = false
      module.forward(input)
    } else {
      val subOutSize = subModels(0).output.toTensor[Float].size()
      if (subOutSize(0) != stackSize) {
        withMultiThread = false
        module.forward(input)
      } else {
        subOutSize(0) = batchSize
        if (output == null || output.toTensor[Float].isEmpty) {
          output = Tensor[Float]().resize(subOutSize)
        }
        val copyThreads = Engine.wrapperComputing.invoke((0 until subModelNumber).map(i =>
          () => {
            output.toTensor[Float].narrow(1, i * stackSize + 1, stackSize)
              .copy(subModels(i).output.toTensor[Float])
          }))
        Engine.wrapperComputing.sync(copyThreads)

        output
      }
    }
  }

  override def updateOutput(input: Activity): Activity = {
    if (!initEnv) setMultiThreadEnv(input)
    output = if (withMultiThread) {
      forwardInParallel(input)
    } else {
      module.forward(input)
    }
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    gradInput = module.updateGradInput(input, gradOutput)
    gradInput
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    module.accGradParameters(input, gradOutput)
  }

  override def clearState() : this.type = {
    super.clearState()
    module.clearState()
    this
  }

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    module.parameters()
  }

  override def equals(obj: Any): Boolean = {
    if (!super.equals(obj) || !obj.isInstanceOf[BlasWrapper]) {
      return false
    }
    val other = obj.asInstanceOf[BlasWrapper]
    if (this.eq(other)) {
      return true
    }
    if (module != other) return false
    true
  }

  override def hashCode() : Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + module.hashCode()
    hash
  }

  override def training(): this.type = {
    train = true
    module.training()
    this
  }

  /**
   * Set the module to evaluate mode
   * @return
   */
  override def evaluate(): this.type = {
    train = false
    module.evaluate()
    this
  }

  override def release(): Unit = module.release()

}


private[bigdl] object BlasWrapper {
  def apply(module: AbstractModule[Activity, Activity, Float]): BlasWrapper =
    new BlasWrapper(module)
}
