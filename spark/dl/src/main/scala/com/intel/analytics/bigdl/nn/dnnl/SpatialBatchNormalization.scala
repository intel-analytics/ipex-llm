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

import com.intel.analytics.bigdl.dnnl._
import com.intel.analytics.bigdl.nn.abstractnn.{Activity, Initializable}
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.nn.{MklInt8Convertible, Ones, VariableFormat, Zeros}
import com.intel.analytics.bigdl.tensor._

import scala.collection.mutable


class SpatialBatchNormalization(
  val nOutput: Int,
  val eps: Double = 1e-5,
  val momentum: Double = 0.1,
  private val initWeight: Tensor[Float] = null,
  private val initBias: Tensor[Float] = null,
  private val initGradWeight: Tensor[Float] = null,
  private val initGradBias: Tensor[Float] = null
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

  // move it into DnnBase
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
        NativeData(Array(nOutput), Memory.FormatTag.x)
      }.toList

    // weight and bias should be combined
    val weightAndBias: NativeData = NativeData(Array(nOutput * 2), Memory.FormatTag.x)

    _inputFormats = singleNativeData(inputs)

//    val src = NativeData(inputs.head.shape, inputs.head.layout, DataType.F32)
//    _inputFormats = Array(src)
    // _inputFormats = Array(NativeData(inputs.head.shape, Memory.FormatTag.nchw))

//    val src = NativeData(inputs.head.shape, inputs.head.layout, DataType.F32)
//    _inputFormats = singleNativeData(inputs)
//    _inputFormats = Array(NativeData(inputs.head.shape, Memory.FormatTag.nchw))
    // init phase status
    initPhase(phase)

    val propKind = modelPhase match {
      case TrainingPhase => PropKind.Forward
      case InferencePhase => PropKind.ForwardInference
      case _ => throw new UnsupportedOperationException("Unknown prop kind")
    }
    val normalizationFlag = modelPhase match {
      case TrainingPhase => DNNL.BatchNormFlag.mkldnn_use_scaleshift
      case InferencePhase => DNNL.BatchNormFlag.mkldnn_use_global_stats |
        DNNL.BatchNormFlag.mkldnn_use_scaleshift
      case _ => throw new UnsupportedOperationException("Unknown prop kind")
    }

    val desc = DnnlMemory.BatchNormForwardDescInit(
      propKind,
      inputFormats().head.getMemoryDescriptor(),
      eps.toFloat,
      normalizationFlag)

    forwardDesc = if (relu) {
      val postOps = DnnlMemory.CreatePostOps()
      DNNL.PostOpsAppendEltwise(postOps, 1.0f, AlgKind.EltwiseRelu, 0.0f, 0.0f)
      val attr = DnnlMemory.CreateAttr()
      DNNL.AttrSetPostOps(attr, postOps)
      DnnlMemory.PrimitiveDescCreateV2(desc, attr, runtime.engine, 0)
    } else {
      DnnlMemory.PrimitiveDescCreate(desc, runtime.engine, 0L)
    }

    val realDst = MemoryData.operationWant(forwardDesc, Query.DstMd)
    _outputFormats = Array(realDst)
    val primitive = DnnlMemory.PrimitiveCreate(forwardDesc)

    fwdExecArgs = mutable.Map(
      ArgType.DNNL_ARG_SRC ->
        inputFormats().head.getMemoryObject(runtime),
      ArgType.DNNL_ARG_SCALE_SHIFT ->
        weightAndBias.getMemoryObject(runtime),
      ArgType.DNNL_ARG_MEAN ->
        mean.getMemoryObject(runtime),
      ArgType.DNNL_ARG_VARIANCE ->
        variance.getMemoryObject(runtime),
      ArgType.DNNL_ARG_DST ->
        outputFormats().head.getMemoryObject(runtime)
    )

    updateOutputPrimitives = Array(primitive)

    // init once
    // if the output is not null, it means we have initialized the primitives before.
    // so we do not need create weightAndBias native space again.
    if (output == null || output.isInstanceOf[DnnTensor[_]] &&
      output.toTensor[Float].size().deep != outputFormats()(0).shape.deep) {
      output = initTensor(outputFormats()(0))
    }

    // init once
    if (this.weightAndBias.native == null) {
      if (modelPhase == InferencePhase) {
        this.runningMean.setMemoryData(
          HeapData(this.runningMean.size(), Memory.FormatTag.x), runningMean, runtime)
        this.runningVariance.setMemoryData(
          HeapData(this.runningVariance.size(), Memory.FormatTag.x), runningVariance, runtime)
        // for inference, we must copy the heap memory to native first.
        this.runningMean.sync()
        this.runningVariance.sync()
      } else {
        this.runningMean.setMemoryData(runningMean,
          HeapData(this.runningMean.size(), Memory.FormatTag.x), runtime)
        this.runningVariance.setMemoryData(runningVariance,
          HeapData(this.runningVariance.size(), Memory.FormatTag.x), runtime)
      }
      // for runningMean and runningVariance, we should copy them to native at first
      this.weightAndBias.setMemoryData(HeapData(this.weightAndBias.size(), Memory.FormatTag.x),
        weightAndBias, runtime)
    }
    this.weightAndBias.sync()
    (inputFormats(), outputFormats())
  }

  override def updateOutput(input: Activity): Activity = {
    if (updateOutputTensors == null) {
      updateOutputTensors = mutable.Map(
        ArgType.DNNL_ARG_SRC -> input.asInstanceOf[Tensor[Float]],
        ArgType.DNNL_ARG_SCALE_SHIFT -> weightAndBias.native,
        ArgType.DNNL_ARG_MEAN -> mean,
        ArgType.DNNL_ARG_VARIANCE -> variance,
        ArgType.DNNL_ARG_DST -> output.asInstanceOf[Tensor[Float]]
      )
    }

    if (this.isTraining()) {
      weightAndBias.sync()
      scaleFactor = scaleFactor * momentum.toFloat + 1
      mean.axpby(1, momentum.toFloat, runningMean.native)
      variance.axpby(biasFactor, momentum.toFloat, runningVariance.native)
      runningMean.sync()
      runningVariance.sync()
    } else {
      // we should re-computing the running mean and running variance.
      // FIXME should do it at `initFwdPrimitives`
      mean.scale(runningMean.native, 1 / scaleFactor)
      variance.scale(runningVariance.native, 1 / scaleFactor)
    }

    updateWithNewTensor(updateOutputTensors, ArgType.DNNL_ARG_SRC, input)
    MklDnnOps.streamSubmit(updateOutputPrimitives, runtime.stream,
      fwdExecArgs, updateOutputTensors)

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
        DnnlMemory.BatchNormBackwardDescInit(PropKind.Backward,
          inputFormats()(0).getMemoryDescriptor(),
          inputFormats()(0).getMemoryDescriptor(), eps.toFloat,
          DNNL.BatchNormFlag.mkldnn_use_scaleshift)
      case _ => throw new UnsupportedOperationException
    }

    val gradWeightAndBias: NativeData = NativeData(Array(nOutput * 2), Memory.FormatTag.x)
    val gradWeightPrimitive = gradWeightAndBias.getMemoryObject(runtime)

    val primDesc = DnnlMemory.PrimitiveDescCreate(backwardDesc, runtime.engine, 0)

    _gradInputFormats = Array(MemoryData.operationWant(primDesc, Query.DiffSrcMd))
    _gradOutputFormats = Array(MemoryData.operationWant(primDesc, Query.DiffDstMd))

    // maybe will throw null exception
    bwdExecArgs = mutable.Map(
      ArgType.DNNL_ARG_SRC -> inputFormats().head.getMemoryObject(runtime),
      ArgType.DNNL_ARG_MEAN -> fwdExecArgs.get(ArgType.DNNL_ARG_MEAN).get,
      ArgType.DNNL_ARG_VARIANCE -> fwdExecArgs.get(ArgType.DNNL_ARG_VARIANCE).get,
      ArgType.DNNL_ARG_DIFF_SRC -> grad(0).getMemoryObject(runtime),
      ArgType.DNNL_ARG_SCALE_SHIFT -> fwdExecArgs.get(ArgType.DNNL_ARG_SCALE_SHIFT).get,
      ArgType.DNNL_ARG_DIFF_SRC -> gradInputFormats()(0).getMemoryObject(runtime),
      ArgType.DNNL_ARG_DIFF_DST -> _gradOutputFormats(0).getMemoryObject(runtime),
      ArgType.DNNL_ARG_DIFF_SCALE_SHIFT -> gradWeightAndBias.getMemoryObject(runtime)
    )

    val primitive = DnnlMemory.PrimitiveCreate(primDesc)

    updateGradInputPrimitives = Array(primitive)
    gradInput = initTensor(gradInputFormats()(0))

    this.gradWeightAndBias.setMemoryData(gradWeightAndBias,
      HeapData(this.gradWeightAndBias.size(), Memory.FormatTag.x), runtime)
    this.gradWeightAndBias.zero()

    (_gradOutputFormats, gradInputFormats())
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    if (updateGradInputTensors == null) {
      updateGradInputTensors = mutable.Map(
        ArgType.DNNL_ARG_SRC -> input.asInstanceOf[Tensor[Float]],
        ArgType.DNNL_ARG_MEAN -> mean,
        ArgType.DNNL_ARG_VARIANCE -> variance,
        ArgType.DNNL_ARG_DIFF_DST -> gradOutput.asInstanceOf[Tensor[Float]],
        ArgType.DNNL_ARG_SCALE_SHIFT -> weightAndBias.native,
        ArgType.DNNL_ARG_DIFF_SRC -> gradInput.asInstanceOf[Tensor[Float]],
        ArgType.DNNL_ARG_DIFF_SCALE_SHIFT -> gradWeightAndBias.native
      )
    }
    updateWithNewTensor(updateGradInputTensors, ArgType.DNNL_ARG_SRC, input)
    updateWithNewTensor(updateGradInputTensors, ArgType.DNNL_ARG_DIFF_DST, gradOutput)
    MklDnnOps.streamSubmit(updateGradInputPrimitives, runtime.stream,
      bwdExecArgs, updateGradInputTensors)
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
    initGradBias: Tensor[Float] = null): SpatialBatchNormalization = {
    new SpatialBatchNormalization(nOutput, eps, momentum, initWeight, initBias, initGradWeight,
      initGradBias)
  }
}
