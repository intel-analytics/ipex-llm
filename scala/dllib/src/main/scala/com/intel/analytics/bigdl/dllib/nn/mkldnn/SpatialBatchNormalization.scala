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

import com.intel.analytics.bigdl.mkl._
import com.intel.analytics.bigdl.nn.abstractnn.{Activity, DataFormat, Initializable}
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.nn.{MklInt8Convertible, Ones, VariableFormat, Zeros}
import com.intel.analytics.bigdl.tensor._

import scala.collection.mutable.ArrayBuffer

class SpatialBatchNormalization(
  val nOutput: Int,
  val eps: Double = 1e-5,
  val momentum: Double = 0.1,
  private val initWeight: Tensor[Float] = null,
  private val initBias: Tensor[Float] = null,
  private val initGradWeight: Tensor[Float] = null,
  private val initGradBias: Tensor[Float] = null,
  val format: DataFormat = DataFormat.NCHW
) extends MklDnnLayer with Initializable with MklInt8Convertible {

  @transient private var forwardDesc: Long = 0L
  private var _relu: Boolean = false

  def setReLU(value: Boolean): this.type = {
    _relu = value
    this
  }
  def relu: Boolean = _relu

  // reminder: runningMean/runningVariance in blas batch_norm is
  // same to scaled runningMean/runningVariance in dnn.
  private[bigdl] var needScale = false

  class SwitchablePrimitives() {
    private var _forwardDesc: Long = 0
    private var _updateOutputMemoryPrimitives : Array[Long] = _
    private var _updateOutputPrimitives: Array[Long] = _
    private var _fwdPrimDesc: Long = 0
    private var _inputFormat: NativeData = _
    private var _outputFormat: NativeData = _

    def switchInOutFormats(): Unit = {
      if (_inputFormat == null) {
        _inputFormat = MemoryData.operationWant(fwdPrimDesc, Query.SrcPd)
      }
      if (_outputFormat == null) {
        _outputFormat = MemoryData.operationWant(fwdPrimDesc, Query.DstPd)
      }
      _inputFormats(0) = _inputFormat
      _outputFormats(0) = _outputFormat
    }

    def fwdPrimDesc: Long = {
      if (_fwdPrimDesc == 0) {
        _fwdPrimDesc = if (relu) {
          val postOps = MklDnnMemory.CreatePostOps()
          MklDnn.PostOpsAppendEltwise(postOps, 1.0f, AlgKind.EltwiseRelu, 0.0f, 0.0f)
          val attr = MklDnnMemory.CreateAttr()
          MklDnn.AttrSetPostOps(attr, postOps)
          MklDnnMemory.PrimitiveDescCreateV2(_forwardDesc, attr, runtime.engine, 0)
        } else {
          MklDnnMemory.PrimitiveDescCreate(_forwardDesc, runtime.engine, 0)
        }
      }
      _fwdPrimDesc
    }

    def forwardDesc(gen: () => Long): Long = {
      if (_forwardDesc == 0) {
        _forwardDesc = gen()
      }
      _forwardDesc
    }

    def switchUpdateOutputMemoryPrimitives(gen: () => (Array[Long], Array[Long])): Unit = {
      if (_updateOutputMemoryPrimitives == null) {
        val generated = gen()
        _updateOutputMemoryPrimitives = generated._1
        _updateOutputPrimitives = generated._2
      }
      updateOutputMemoryPrimitives = _updateOutputMemoryPrimitives
      updateOutputPrimitives = _updateOutputPrimitives
    }
  }

  @transient private lazy val trainingPrimitives = new SwitchablePrimitives
  @transient private lazy val inferencePrimitives = new SwitchablePrimitives

  @transient private var updateOutputTensors: Array[Tensor[Float]] = _
  @transient private var updateOutputMemoryPrimitives: Array[Long] = _
  @transient private var updateGradInputTensors: Array[Tensor[Float]] = _
  @transient private var updateGradInputMemoryPrimitives: Array[Long] = _
  @transient private var modelPhase: Phase = null

  private val mean: DnnTensor[Float] = DnnTensor[Float](nOutput)
  private val variance: DnnTensor[Float] = DnnTensor[Float](nOutput)

  private[mkldnn] val runningMean = new TensorMMap(Array(nOutput))
  private[mkldnn] val runningVariance = new TensorMMap(Array(nOutput))
  // TODO we should make it private. Currently, ResNet50 will use it out of this scope.
  val weightAndBias = new TensorMMap(Array(nOutput * 2))
  val gradWeightAndBias = new TensorMMap(Array(nOutput * 2))

  // TODO the two should be learnable parameters
  var scaleFactor: Float = 1.0f
  var biasFactor: Float = 1.0f

  private val runningMeanScaled = Tensor[Float].resizeAs(runningMean.dense)
  private val runningVarianceScaled = Tensor[Float].resizeAs(runningVariance.dense)

  // the blank shoud be here, otherwise the runningVarianceScaled will be a method
  {
    val wInit = Ones // RandomUniform(0, 1)
    val bInit = Zeros
    setInitMethod(wInit, bInit)
  }

  override def reset(): Unit = {
    val init = Tensor[Float]().resize(Array(2, nOutput))
    val weight = init.select(1, 1)
    val bias = init.select(1, 2)

    if (initWeight != null) {
      require(initWeight.size(1) == nOutput)
      weight.copy(initWeight)
    } else {
      weightInitMethod.init(weight, VariableFormat.ONE_D)
    }

    if (initBias != null) {
      require(initBias.size(1) == nOutput)
      bias.copy(initBias)
    } else {
      biasInitMethod.init(bias, VariableFormat.ONE_D)
    }

    weightAndBias.dense.copy(init.view(2 * nOutput))

    val zeros = Tensor[Float](Array(nOutput)).fill(0)
    mean.copy(zeros)
    variance.copy(zeros)

    runningMean.copy(zeros)
    runningVariance.copy(zeros)
  }

  private object Index extends Serializable {
    val input = 0
    val weight = 1
    val output = 2
    val mean = 3
    val variance = 4
  }

  private def initPhase(phase: Phase): Unit = {
    if (phase != null) modelPhase = phase
    (isTraining(), modelPhase) match {
      case (true, InferencePhase) =>
        train = false
      case (false, TrainingPhase) =>
        train = true
      case (true, null) =>
        modelPhase = TrainingPhase
      case (false, null) =>
        modelPhase = InferencePhase
      case _ =>
    }
  }

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    val m = inputs(0).shape.product / this.nOutput
    biasFactor = if (m > 1) { m.toFloat / (m - 1) } else { 1 }

    val List(mean, variance, runningMean, runningVariance): List[NativeData] =
      (0 until 4).map { _ =>
        NativeData(Array(nOutput), Memory.Format.x)
      }.toList
    // weight and bias should be combined
    val weightAndBias: NativeData = NativeData(Array(nOutput * 2), Memory.Format.x)

    // the bn only accept F32 as input, like lrn
    val src = NativeData(inputs.head.shape, inputs.head.layout, DataType.F32)

    // init once
    if (_inputFormats == null) {
      _inputFormats = new Array[MemoryData](1)
      require(_outputFormats == null)
      _outputFormats = new Array[MemoryData](1)
    }

    // init phase status
    initPhase(phase)

    modelPhase match {
      case TrainingPhase =>
        forwardDesc = trainingPrimitives.forwardDesc(() => MklDnnMemory.BatchNormForwardDescInit(
          PropKind.Forward,
          src.getMemoryDescription(), eps.toFloat, MklDnn.BatchNormFlag.mkldnn_use_scaleshift))
        val fwdPrimDesc = trainingPrimitives.fwdPrimDesc
        trainingPrimitives.switchInOutFormats()
        trainingPrimitives.switchUpdateOutputMemoryPrimitives(() => {
          val srcs = Array(inputFormats()(0), weightAndBias).map(_.getPrimitive(runtime))
          val dsts = Array(outputFormats()(0), mean, variance).map(_.getPrimitive(runtime))
          val indexes = Array.fill(srcs.length)(0)
          val primitive = MklDnnMemory.PrimitiveCreate2(fwdPrimDesc, srcs, indexes,
            srcs.length, dsts, dsts.length)
          val _updateOutputMemoryPrimitives = srcs ++ dsts
          val _updateOutputPrimitives = Array(primitive)
          (_updateOutputMemoryPrimitives, _updateOutputPrimitives)
        })
      case InferencePhase =>
        // we always use the weight and bias / scale and offset. So the flags should be combined
        // with use_scaleshift and use_global_stats.
        forwardDesc = inferencePrimitives.forwardDesc(() =>
          MklDnnMemory.BatchNormForwardDescInit(PropKind.ForwardInference,
            src.getMemoryDescription(), eps.toFloat, MklDnn.BatchNormFlag.mkldnn_use_global_stats
              | MklDnn.BatchNormFlag.mkldnn_use_scaleshift))
        val fwdPrimDesc = inferencePrimitives.fwdPrimDesc
        inferencePrimitives.switchInOutFormats()
        inferencePrimitives.switchUpdateOutputMemoryPrimitives(() => {
          val srcs = Array(inputFormats()(0), mean, variance, weightAndBias).map(_.getPrimitive
          (runtime))
          val dsts = Array(outputFormats()(0).getPrimitive(runtime))
          val indexes = Array.fill(srcs.length)(0)
          val primitive = MklDnnMemory.PrimitiveCreate2(fwdPrimDesc, srcs, indexes,
            srcs.length, dsts, dsts.length)
          val _updateOutputMemoryPrimitives = srcs ++ dsts
          val _updateOutputPrimitives = Array(primitive)
          (_updateOutputMemoryPrimitives, _updateOutputPrimitives)
        })
      case _ => throw new UnsupportedOperationException
    }

    // init once
    // if the output is not null, it means we have initialized the primitives before.
    // so we do not need create weightAndBias native space again.
    if (output == null || output.isInstanceOf[DnnTensor[_]] &&
      output.toTensor[Float].size().deep != outputFormats()(0).shape.deep) {
      output = initTensor(outputFormats()(0))
    }

    if (updateOutputTensors != null) {
      updateOutputTensors = null
    }

    // init once
    if (this.weightAndBias.native == null) {
      if (modelPhase == InferencePhase) {
        this.runningMean.setMemoryData(
          HeapData(this.runningMean.size(), Memory.Format.x), runningMean, runtime)
        this.runningVariance.setMemoryData(
          HeapData(this.runningVariance.size(), Memory.Format.x), runningVariance, runtime)
        // for inference, we must copy the heap memory to native first.
        this.runningMean.sync()
        this.runningVariance.sync()
      } else {
        this.runningMean.setMemoryData(runningMean,
          HeapData(this.runningMean.size(), Memory.Format.x), runtime)
        this.runningVariance.setMemoryData(runningVariance,
          HeapData(this.runningVariance.size(), Memory.Format.x), runtime)
      }
      // for runningMean and runningVariance, we should copy them to native at first
      this.weightAndBias.setMemoryData(HeapData(this.weightAndBias.size(), Memory.Format.x),
        weightAndBias, runtime)
    }
    this.weightAndBias.sync()

    (inputFormats(), outputFormats())
  }

  override def updateOutput(input: Activity): Activity = {
    if (updateOutputTensors == null) {
      if (this.isTraining()) {
        val buffer = new ArrayBuffer[Tensor[Float]]()
        buffer.append(input.asInstanceOf[Tensor[Float]])
        buffer.append(weightAndBias.native)
        buffer.append(output.asInstanceOf[Tensor[Float]])
        buffer.append(mean)
        buffer.append(variance)
        updateOutputTensors = buffer.toArray
      } else {
        val buffer = new ArrayBuffer[Tensor[Float]]()
        buffer.append(input.asInstanceOf[Tensor[Float]])
        buffer.append(mean)
        buffer.append(variance)
        buffer.append(weightAndBias.native)
        buffer.append(output.asInstanceOf[Tensor[Float]])
        updateOutputTensors = buffer.toArray
      }
    }

    if (this.isTraining()) {
      weightAndBias.sync()
    } else {
      // we should re-computing the running mean and running variance.
      // FIXME should do it at `initFwdPrimitives`
      mean.scale(runningMean.native, 1 / scaleFactor)
      variance.scale(runningVariance.native, 1 / scaleFactor)
    }

    updateWithNewTensor(updateOutputTensors, 0, input)

    MklDnnOps.streamSubmit(runtime.stream, 1, updateOutputPrimitives, updateOutputPrimitives.length,
      updateOutputMemoryPrimitives, updateOutputTensors)

    if (this.isTraining()) {
      // update running(Mean, Var) and scaleFactor
      scaleFactor = scaleFactor * momentum.toFloat + 1

      mean.axpby(1, momentum.toFloat, runningMean.native)
      variance.axpby(biasFactor, momentum.toFloat, runningVariance.native)

      runningMean.sync()
      runningVariance.sync()
    }

    output
  }

  override private[mkldnn] def initBwdPrimitives(grad: Array[MemoryData], phase: Phase) = {
    _gradOutputFormats = Array(NativeData(outputFormats()(0).shape, outputFormats()(0).layout))

    // init phase status
    initPhase(phase)
    // [PERF] the format of gradInput should be the same as input
    val backwardDesc = modelPhase match {
      case TrainingPhase =>
        MklDnnMemory.BatchNormBackwardDescInit(PropKind.Backward,
          inputFormats()(0).getMemoryDescription(),
          inputFormats()(0).getMemoryDescription(), eps.toFloat,
          MklDnn.BatchNormFlag.mkldnn_use_scaleshift)
      case _ => throw new UnsupportedOperationException
    }

    val gradWeightAndBias: NativeData = NativeData(Array(nOutput * 2), Memory.Format.x)
    val gradWeightPrimitive = gradWeightAndBias.getPrimitive(runtime)

    val primDesc = MklDnnMemory.PrimitiveDescCreate(backwardDesc, runtime.engine, 0)

    _gradInputFormats = Array(MemoryData.operationWant(primDesc, Query.DiffSrcPd))

    // maybe will throw null exception
    val srcs = Array(updateOutputMemoryPrimitives(Index.input),
      updateOutputMemoryPrimitives(Index.mean),
      updateOutputMemoryPrimitives(Index.variance),
      grad(0).getPrimitive(runtime),
      updateOutputMemoryPrimitives(Index.weight))
    val indexes = Array.fill(srcs.length)(0)
    val dsts = Array(gradInputFormats()(0), gradWeightAndBias).map(_.getPrimitive(runtime))

    val primitive = MklDnnMemory.PrimitiveCreate2(primDesc, srcs, indexes, srcs.length,
      dsts, dsts.length)

    updateGradInputMemoryPrimitives = srcs ++ dsts
    updateGradInputPrimitives = Array(primitive)
    gradInput = initTensor(gradInputFormats()(0))

    this.gradWeightAndBias.setMemoryData(gradWeightAndBias,
      HeapData(this.gradWeightAndBias.size(), Memory.Format.x), runtime)
    this.gradWeightAndBias.zero()

    (_gradOutputFormats, gradInputFormats())
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    if (updateGradInputTensors == null) {
      val buffer = new ArrayBuffer[Tensor[Float]]()
      buffer.append(input.asInstanceOf[Tensor[Float]])
      buffer.append(mean)
      buffer.append(variance)
      buffer.append(gradOutput.asInstanceOf[Tensor[Float]])
      buffer.append(weightAndBias.native)
      buffer.append(gradInput.asInstanceOf[Tensor[Float]])
      buffer.append(gradWeightAndBias.native)
      updateGradInputTensors = buffer.toArray
    }

    updateWithNewTensor(updateGradInputTensors, 0, input)
    updateWithNewTensor(updateGradInputTensors, 3, gradOutput)

    MklDnnOps.streamSubmit(runtime.stream, 1, updateGradInputPrimitives,
      updateGradInputPrimitives.length, updateGradInputMemoryPrimitives, updateGradInputTensors)

    gradWeightAndBias.sync()

    gradInput
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    // do nothing
  }

  override def zeroGradParameters(): Unit = {
  }

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    (Array(weightAndBias.dense), Array(gradWeightAndBias.dense))
  }

  override def paramsMMap(): (Array[TensorMMap], Array[TensorMMap]) = {
    (Array(weightAndBias), Array(gradWeightAndBias))
  }

  override def getExtraParameter(): Array[Tensor[Float]] = {
    if (needScale) {
      runningMeanScaled.copy(runningMean.dense).div(scaleFactor)
      runningVarianceScaled.copy(runningVariance.dense).div(scaleFactor)
      Array(runningMeanScaled, runningVarianceScaled)
    } else {
      Array(runningMean.dense, runningVariance.dense)
    }
  }

  override def toString(): String = {
    s"nn.mkl.SpatialBatchNormalization($nOutput, $eps, $momentum)"
  }

  override def evaluate(): this.type = {
    if (modelPhase == TrainingPhase) {
      initFwdPrimitives(inputFormats(), InferencePhase)
    }
    this
  }

  override def training(): this.type = {
    if (modelPhase == InferencePhase) {
      initFwdPrimitives(inputFormats(), TrainingPhase)
    }
    this
  }
}

object SpatialBatchNormalization {
  def apply(
    nOutput: Int,
    eps: Double = 1e-5,
    momentum: Double = 0.1,
    affine: Boolean = true,
    initWeight: Tensor[Float] = null,
    initBias: Tensor[Float] = null,
    initGradWeight: Tensor[Float] = null,
    initGradBias: Tensor[Float] = null,
    format: DataFormat = DataFormat.NCHW): SpatialBatchNormalization = {
    new SpatialBatchNormalization(nOutput, eps, momentum, initWeight, initBias, initGradWeight,
      initGradBias, format = format)
  }
}
