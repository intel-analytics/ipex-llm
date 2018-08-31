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
import com.intel.analytics.bigdl.nn.abstractnn.{Activity, Initializable}
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.nn.{Ones, VariableFormat, Zeros}
import com.intel.analytics.bigdl.tensor._

import scala.collection.mutable.ArrayBuffer

class SpatialBatchNormalization(
  val nOutput: Int,
  val eps: Double = 1e-5,
  val momentum: Double = 0.1,
  val affine: Boolean = true,
  private val initWeight: Tensor[Float] = null,
  private val initBias: Tensor[Float] = null,
  private val initGradWeight: Tensor[Float] = null,
  private val initGradBias: Tensor[Float] = null
) extends MklDnnLayer with Initializable {

  @transient private var forwardDesc: Long = 0L
  private var _relu: Boolean = false

  def setReLU(value: Boolean): this.type = {
    _relu = value
    this
  }
  def relu: Boolean = _relu

  @transient private var updateOutputTensors: Array[Tensor[Float]] = _
  @transient private var updateOutputMemoryPrimitives: Array[Long] = _
  @transient private var updateGradInputTensors: Array[Tensor[Float]] = _
  @transient private var updateGradInputMemoryPrimitives: Array[Long] = _

  private val mean: DnnTensor[Float] = DnnTensor[Float](nOutput)
  private val variance: DnnTensor[Float] = DnnTensor[Float](nOutput)

  private[mkldnn] val runningMean = new Blob(Array(nOutput))
  private[mkldnn] val runningVariance = new Blob(Array(nOutput))
  // TODO we should make it private. Currently, ResNet50 will use it out of this scope.
  val weightAndBias = new Blob(Array(nOutput * 2))
  val gradWeightAndBias = new Blob(Array(nOutput * 2))

  var scaleFactor: Float = 0.0f
  var biasFactor: Float = 0.0f

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

    weightAndBias.copy(init.view(2 * nOutput))

    val zeros = Tensor[Float](Array(nOutput)).fill(0)
    mean.copy(zeros)
    variance.copy(zeros)

    runningMean.zero()
    runningVariance.zero()
  }

  private object Index extends Serializable {
    val input = 0
    val weight = 1
    val output = 2
    val mean = 3
    val variance = 4
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

    forwardDesc = phase match {
      case TrainingPhase =>
        MklDnn.BatchNormForwardDescInit(PropKind.Forward,
          inputs(0).getMemoryDescription(), eps.toFloat, MklDnn.BatchNormFlag.mkldnn_use_scaleshift)
      case InferencePhase =>
        // we always use the weight and bias / scale and offset. So the flags should be combined
        // with use_scaleshift and use_global_stats.
        MklDnn.BatchNormForwardDescInit(PropKind.ForwardInference,
          inputs(0).getMemoryDescription(), eps.toFloat,
          MklDnn.BatchNormFlag.mkldnn_use_global_stats | MklDnn.BatchNormFlag.mkldnn_use_scaleshift)
      case _ => throw new UnsupportedOperationException
    }

    val primDesc = if (relu) {
      val postOps = MklDnn.CreatePostOps()
      MklDnn.PostOpsAppendEltwise(postOps, 1.0f, AlgKind.EltwiseRelu, 0.0f, 0.0f)
      val attr = MklDnn.CreateAttr()
      MklDnn.AttrSetPostOps(attr, postOps)
      MklDnn.PrimitiveDescCreateV2(forwardDesc, attr, runtime.engine, 0)
      // TODO we should destroy these ops
    } else {
      MklDnn.PrimitiveDescCreate(forwardDesc, runtime.engine, 0)
    }

    if (_inputFormats == null) {
      _inputFormats = new Array[MemoryData](1)
    }

    if (_outputFormats == null) {
      _outputFormats = new Array[MemoryData](1)
    }

    _inputFormats(0) = MemoryData.operationWant(primDesc, Query.SrcPd)
    _outputFormats(0) = MemoryData.operationWant(primDesc, Query.DstPd)

    val (srcs, dsts) = if (phase == TrainingPhase) {
      val srcs = Array(inputFormats()(0), weightAndBias).map(_.getPrimitive(runtime))
      val dsts = Array(outputFormats()(0), mean, variance).map(_.getPrimitive(runtime))
      (srcs, dsts)
    } else {
      val srcs = Array(inputFormats()(0), mean, variance, weightAndBias).map { x =>
        x.getPrimitive(runtime)
      }
      val dsts = Array(outputFormats()(0).getPrimitive(runtime))
      (srcs, dsts)
    }
    val indexes = Array.fill(srcs.length)(0)

    val primitive = MklDnn.PrimitiveCreate2(primDesc, srcs, indexes, srcs.length, dsts, dsts.length)

    updateOutputMemoryPrimitives = srcs ++ dsts
    updateOutputPrimitives = Array(primitive)

    if (output == null || output.isInstanceOf[DnnTensor[_]] &&
      output.toTensor[Float].size().deep != outputFormats()(0).shape.deep) {
      output = initTensor(outputFormats()(0))
    }

    if (updateOutputTensors != null) {
      updateOutputTensors = null
    }

    (isTraining(), phase) match {
      case (true, InferencePhase) => train = false
      case (false, TrainingPhase) => train = true
      case _ =>
    }

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
      weightAndBias.syncToNative()
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

      runningMean.syncToHeap()
      runningVariance.syncToHeap()
    }

    output
  }

  override private[mkldnn] def initBwdPrimitives(grad: Array[MemoryData], phase: Phase) = {
    _gradOutputFormats = Array(NativeData(outputFormats()(0).shape, outputFormats()(0).layout))

    // [PERF] the format of gradInput should be the same as input
    val backwardDesc = phase match {
      case TrainingPhase =>
        MklDnn.BatchNormBackwardDescInit(PropKind.Backward,
          inputFormats()(0).getMemoryDescription(),
          inputFormats()(0).getMemoryDescription(), eps.toFloat,
          MklDnn.BatchNormFlag.mkldnn_use_scaleshift)
      case _ => throw new UnsupportedOperationException
    }

    val gradWeightAndBias: NativeData = NativeData(Array(nOutput * 2), Memory.Format.x)
    val gradWeightPrimitive = gradWeightAndBias.getPrimitive(runtime)

    val primDesc = MklDnn.PrimitiveDescCreate(backwardDesc, runtime.engine, 0)

    _gradInputFormats = Array(MemoryData.operationWant(primDesc, Query.DiffSrcPd))

    // maybe will throw null exception
    val srcs = Array(updateOutputMemoryPrimitives(Index.input),
      updateOutputMemoryPrimitives(Index.mean),
      updateOutputMemoryPrimitives(Index.variance),
      grad(0).getPrimitive(runtime),
      updateOutputMemoryPrimitives(Index.weight))
    val indexes = Array.fill(srcs.length)(0)
    val dsts = Array(gradInputFormats()(0), gradWeightAndBias).map(_.getPrimitive(runtime))

    val primitive = MklDnn.PrimitiveCreate2(primDesc, srcs, indexes, srcs.length,
      dsts, dsts.length)

    updateGradInputMemoryPrimitives = srcs ++ dsts
    updateGradInputPrimitives = Array(primitive)
    gradInput = initTensor(gradInputFormats()(0))

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

    gradWeightAndBias.syncToHeap()

    gradInput
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    // do nothing
  }

  override def zeroGradParameters(): Unit = {
    if (affine) {
      gradWeightAndBias.zero()
    }
  }

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    (Array(weightAndBias.dense), Array(gradWeightAndBias.dense))
  }

  override def getExtraParameter(): Array[Tensor[Float]] = {
    Array(runningMean.dense, runningVariance.dense)
  }

  override def parametersWithShape(): (Array[MemoryData], Array[MemoryData]) = {
    (Array(NativeData(weightAndBias.size(), Memory.Format.x)),
    Array(NativeData(gradWeightAndBias.size(), Memory.Format.x)))
  }

  override def toString(): String = {
    s"nn.mkl.SpatialBatchNormalization($nOutput, $eps, $momentum, $affine)"
  }

  override def evaluate(): this.type = {
    if (isTraining()) {
      initFwdPrimitives(inputFormats(), InferencePhase)
    }
    this
  }

  override def training(): this.type = {
    if (!isTraining()) {
      initFwdPrimitives(inputFormats(), TrainingPhase)
    }
    this
  }

  override def release(): Unit = {
    super.release()
    List(weightAndBias, gradWeightAndBias, runningMean, runningVariance).foreach(_.release())
    List(mean, variance).foreach(_.release())
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
    new SpatialBatchNormalization(nOutput, eps, momentum, affine,
      initWeight, initBias, initGradWeight, initGradBias)
  }
}
