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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.MiniBatch
import com.intel.analytics.bigdl.mkl.{MKL, Memory}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.utils.intermediate.IRGraph
import spire.syntax.module

/**
 * wrap blas module to be dnn module,
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

  private[mkldnn] var needOutputFormats: Boolean = true
  @transient private var workingModels: Array[Module[Float]] = _
  @transient private var _subModelNumber : Int = 1
  @transient private var withMultiThread: Boolean = false
  @transient private var inputBuffer : Array[Activity] = _
  @transient private var tensorBuffer : Array[Tensor[Float]] = _
  @transient private var batchSize : Int = _

  private def inferInputFormats(inputs: Array[MemoryData]): Array[MemoryData] = {
    inputs.map(in => HeapData(in.shape, getFormats(in.shape.length)))
  }

  private def inferOutputFormats(inputs: Array[MemoryData]): Array[MemoryData] = {
    val inputShape = inputs.map(in => Shape(in.shape))
    val outputShape = if (inputShape.length == 1) {
      List(module.computeOutputShape(inputShape(0)))
    } else {
      // multi shape
      val out = module.computeOutputShape(MultiShape(inputShape.toList))
      if (out.isInstanceOf[MultiShape]) out.toMulti() else List(out)
    }
    outputShape.map(in => {
      val size = in.toSingle().toArray
      HeapData(size, getFormats(size.length))
    }).toArray
  }

  /**
   * Blas layers may not have good performance with mkldnn backend,
   * so we can use java multi thread to run blas layers and thread number is
   * related with batch size and core number.
   * input and output for this module must be in batch.
   */
  private def setMultiThreadEnv(): Unit = {
    val multiThread = System.getProperty("multiThread", "false").toBoolean
    if (this.train || !multiThread) return
    if (_inputFormats == null || _outputFormats == null || _outputFormats.length != 1) return
    if (_inputFormats(0).shape(0) != _outputFormats(0).shape(0)) return
    batchSize = _inputFormats(0).shape(0)
    val t = batchSize % Engine.coreNumber()
    if (t != 0 || batchSize < 2 || Engine.coreNumber() < 2) return
    _subModelNumber = Engine.coreNumber()
    initModules()
    withMultiThread = true
  }

  private def initModules(): Unit = {
    workingModels = if (module.parameters() != null) {
        val wb = Util.getAndClearWeightBias(module.parameters())
        val models = (1 to _subModelNumber).map(i => {
          val m = module.cloneModule()
          Util.putWeightBias(wb, m)
          m.asInstanceOf[Module[Float]]
        }).toArray
        Util.putWeightBias(wb, module)
        models
      } else {
        val models = (1 to _subModelNumber).map(i => {
          val m = module.cloneModule()
          m.asInstanceOf[Module[Float]]
        }).toArray
        models
      }
  }

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    _inputFormats = inferInputFormats(inputs)
    _outputFormats = if (needOutputFormats) inferOutputFormats(inputs) else null
    setMultiThreadEnv()
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

  private def flattenInput(input: Activity): Unit = {
    if (input.isTensor) {
      tensorBuffer(0) = input.toTensor[Float]
    } else {
      val in = input.toTable
      for (i <- 0 to in.length()) {
        tensorBuffer(i) = in.get[Tensor[Float]](i + 1).get
      }
    }
  }
  private def getInput(dim: Int, index: Int, size: Int): Activity = {
    if (tensorBuffer.length == 1) {
      tensorBuffer(0).narrow(dim, index, size)
    } else {
      T.array(tensorBuffer.map(_.narrow(dim, index, size)))
    }
  }
  private def multiThreadForInference(input: Activity): Activity = {
    if (inputBuffer == null) inputBuffer = new Array[Activity](_subModelNumber)
    if (tensorBuffer == null) tensorBuffer = new Array[Tensor[Float]](_inputFormats.length)
    if (output == null || output.toTensor[Float].isEmpty) {
      output = Tensor[Float]().resize(_outputFormats(0).shape)
    }
    flattenInput(input)

    val stackSize = batchSize / _subModelNumber
    val tasks = Engine.wrapperComputing.invoke(() => {
      var b = 0
      while (b < _subModelNumber) {
        inputBuffer(b) = getInput(1, b * stackSize + 1, stackSize)
        b += 1
      }
    })
    Engine.wrapperComputing.sync(Seq(tasks))

    val trainingThreads = Engine.wrapperComputing.invoke((0 until _subModelNumber).map(i =>
      () => {
          val out = workingModels(i).forward(inputBuffer(i)).toTensor[Float]
          output.toTensor[Float].narrow(1, i * stackSize + 1, stackSize).copy(out)
      }))
    Engine.wrapperComputing.sync(trainingThreads)

    output
  }

  override def updateOutput(input: Activity): Activity = {
    output = if (withMultiThread) {
      multiThreadForInference(input)
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
