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

import com.intel.analytics.bigdl.mkl.MklDnn
import com.intel.analytics.bigdl.mkl.MklDnn.{EngineType, StreamType}
import com.intel.analytics.bigdl.nn.abstractnn.{Initializable, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class SpatialBatchNormalization[T: ClassTag](
  val nOutput: Int,
  val eps: Double = 1e-5,
  val momentum: Double = 0.1,
  val affine: Boolean = true,
  private val initWeight: Tensor[T] = null,
  private val initBias: Tensor[T] = null,
  private val initGradWeight: Tensor[T] = null,
  private val initGradBias: Tensor[T] = null
)(implicit ev: TensorNumeric[T]) extends TensorModule[T] with Initializable {

  val mean: Tensor[T] = Tensor[T](nOutput)
  val variance: Tensor[T] = Tensor[T](nOutput)

  val (all, weight, bias) = createParams(initWeight, initBias)
  val (gradAll, gradWeight, gradBias) = createParams(initGradWeight, initGradBias)
  val (diffAll, diffWeight, diffBias) = createParams(initGradWeight, initGradBias)

  // TODO should be refactored to on module
  val engine = MklDnn.EngineCreate(EngineType.cpu, 0)

  var forwardStream = 0L
  var backwardStream = 0L

  var forwardPrim = 0L
  var backwardPrim = 0L
  var forwardPrimDesc = 0L

  // forward memory primitive
  @transient var userSrcMemoryPrim = 0L
  @transient var userWeightMemoryPrim = 0L
  @transient var userMeanMemoryPrim = 0L
  @transient var userVarMemoryPrim = 0L
  @transient var userDstMemoryPrim = 0L

  @transient var userDiffSrcMemoryPrim = 0L
  @transient var userDiffWeightMemoryPrim = 0L
  @transient var userDiffDstMemoryPrim = 0L

  private def initDataMemory(dim: Int, dims: Array[Int], format: Int,
    dataType: Int, engine: Long, tensor: Tensor[T]): Long = {
    val primMd = MklDnn.MemoryDescInit(dim, dims, dataType, format)
    val userPd = MklDnn.MemoryPrimitiveDescCreate(primMd, engine)
    val memory = MklDnn.PrimitiveCreate0(userPd)

    MklDnn.PrimitiveDescDestroy(userPd)
    memory
  }

  private def setHandle(tensor: Tensor[T], primitive: Long): Unit = {
    val data = tensor.storage().array().asInstanceOf[Array[Float]]
    val offset = tensor.storageOffset() - 1
    MklDnn.MemorySetDataHandle(primitive, data, offset)
  }

  private def releaseHandles(input: Tensor[T], ptr: Long): Unit = {
    MklDnn.MemoryReleaseDataHandle(
      input.storage().array().asInstanceOf[Array[Float]], ptr)
  }

  // TODO train and inference mode ???

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input)

    if (forwardPrim == 0L) {
      val srcMemDesc = MklDnn.MemoryDescInit(input.dim(), input.size(),
        MklDnn.DataType.f32, MklDnn.MemoryFormat.nchw)

      val allMemDesc = MklDnn.MemoryDescInit(all.dim(), all.size(),
        MklDnn.DataType.f32, MklDnn.MemoryFormat.x)

      val meanMemDesc = MklDnn.MemoryDescInit(mean.dim(), mean.size(),
        MklDnn.DataType.f32, MklDnn.MemoryFormat.x)
        val varMemDesc = MklDnn.MemoryDescInit(variance.dim(), variance.size(),
        MklDnn.DataType.f32, MklDnn.MemoryFormat.x)

      val dstMemDesc = MklDnn.MemoryDescInit(output.dim(), output.size(),
        MklDnn.DataType.f32, MklDnn.MemoryFormat.nchw)

      val opDesc = MklDnn.BatchNormForwardDescInit(MklDnn.PropKind.forward,
        srcMemDesc, eps.toFloat, MklDnn.BatchNormFlag.mkldnn_use_scaleshift)
      val opPrimDesc = MklDnn.PrimitiveDescCreate(opDesc, engine, 0)
      forwardPrimDesc = opPrimDesc

      userSrcMemoryPrim = initDataMemory(input.dim(), input.size(),
        MklDnn.MemoryFormat.nchw, MklDnn.DataType.f32, engine, input)
      userWeightMemoryPrim = initDataMemory(all.dim(), all.size(),
        MklDnn.MemoryFormat.x, MklDnn.DataType.f32, engine, all)
      userDstMemoryPrim = initDataMemory(output.dim(), output.size(),
        MklDnn.MemoryFormat.nchw, MklDnn.DataType.f32, engine, output)
      userMeanMemoryPrim = initDataMemory(mean.dim(), mean.size(),
        MklDnn.MemoryFormat.x, MklDnn.DataType.f32, engine, mean)
      userVarMemoryPrim = initDataMemory(variance.dim(), variance.size(),
        MklDnn.MemoryFormat.x, MklDnn.DataType.f32, engine, variance)

      val srcs = Array(userSrcMemoryPrim, userWeightMemoryPrim)
      val indexes = Array(0, 0)
      val dsts = Array(userDstMemoryPrim, userMeanMemoryPrim, userVarMemoryPrim)

      forwardPrim = MklDnn.PrimitiveCreate2(opPrimDesc, srcs, indexes, srcs.length,
        dsts, dsts.length)
    }

    if (forwardStream == 0L) {
      forwardStream = MklDnn.StreamCreate(StreamType.eager)
    }

    setHandle(input, userSrcMemoryPrim)
    setHandle(all, userWeightMemoryPrim)
    setHandle(output, userDstMemoryPrim)
    setHandle(mean, userMeanMemoryPrim)
    setHandle(variance, userVarMemoryPrim)

    MklDnn.StreamSubmit(forwardStream, 1, Array(forwardPrim))

    releaseHandles(input, userSrcMemoryPrim)
    releaseHandles(all, userWeightMemoryPrim)
    releaseHandles(output, userDstMemoryPrim)
    releaseHandles(mean, userMeanMemoryPrim)
    releaseHandles(variance, userVarMemoryPrim)
    output
  }

  override def backward(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input)

    if (backwardPrim == 0L) {
      val srcMemDesc = MklDnn.MemoryDescInit(input.dim(), input.size(),
        MklDnn.DataType.f32, MklDnn.MemoryFormat.nchw)

      val diffDstMemDesc = MklDnn.MemoryDescInit(gradOutput.dim(), gradOutput.size(),
        MklDnn.DataType.f32, MklDnn.MemoryFormat.nchw)

      val desc = MklDnn.BatchNormBackwardDescInit(MklDnn.PropKind.backward,
        diffDstMemDesc,
        srcMemDesc,
        eps.toFloat,
        MklDnn.BatchNormFlag.mkldnn_use_scaleshift)
      val primDesc = MklDnn.PrimitiveDescCreate(desc, engine, forwardPrimDesc)

      userDiffDstMemoryPrim = initDataMemory(gradOutput.dim(), gradOutput.size(),
        MklDnn.MemoryFormat.nchw, MklDnn.DataType.f32, engine, gradOutput)

      userDiffSrcMemoryPrim = initDataMemory(gradInput.dim(), gradInput.size(),
        MklDnn.MemoryFormat.nchw, MklDnn.DataType.f32, engine, gradInput)

      userDiffWeightMemoryPrim = initDataMemory(all.dim(), all.size(),
        MklDnn.MemoryFormat.x, MklDnn.DataType.f32, engine, all)

      val dataSrcs = Array(userSrcMemoryPrim, userMeanMemoryPrim, userVarMemoryPrim,
        userDiffDstMemoryPrim, userWeightMemoryPrim)
      val dataIndexes = Array.fill(dataSrcs.length)(0)
      val dataDsts = Array(userDiffSrcMemoryPrim, userDiffWeightMemoryPrim)

      backwardPrim = MklDnn.PrimitiveCreate2(primDesc, dataSrcs, dataIndexes, dataSrcs.length,
        dataDsts, dataDsts.length)

      MklDnn.PrimitiveDescDestroy(primDesc)
    }

    if (backwardStream == 0) {
      backwardStream = MklDnn.StreamCreate(StreamType.eager)
    }

    setHandle(input, userSrcMemoryPrim)
    setHandle(all, userWeightMemoryPrim)
    setHandle(mean, userMeanMemoryPrim)
    setHandle(variance, userVarMemoryPrim)
    setHandle(gradOutput, userDiffDstMemoryPrim)
    setHandle(gradInput, userDiffSrcMemoryPrim)
    setHandle(diffAll, userDiffWeightMemoryPrim)

    MklDnn.StreamSubmit(backwardStream, 1, Array(backwardPrim))

    releaseHandles(input, userSrcMemoryPrim)
    releaseHandles(all, userWeightMemoryPrim)
    releaseHandles(mean, userMeanMemoryPrim)
    releaseHandles(variance, userVarMemoryPrim)
    releaseHandles(gradOutput, userDiffDstMemoryPrim)
    releaseHandles(gradInput, userDiffSrcMemoryPrim)
    releaseHandles(diffAll, userDiffWeightMemoryPrim)

    gradAll.add(diffAll)

    gradInput
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput
  }

  // there's no relavant accGrasdParameters in mkl-dnn. we use @backward instead of
  // @updateGradInput and @accGradParameters
  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
  }

  private type Params[R] = (Tensor[R], Tensor[R], Tensor[R])
  // in mkl dnn, the weight and bias should be all in the same array
  private def createParams(initWeight: Tensor[T], initBias: Tensor[T]): Params[T] = {
    val weightAndBias: Tensor[T] = if (affine) {
      Tensor[T](2, nOutput)
    } else {
      null
    }
    weightAndBias.fill(ev.fromType(0)) // by default, we init them with 0

    // we should delete the first dim which is 1 after narrow.
    val weight: Tensor[T] = weightAndBias.narrow(1, 1, 1).squeeze(1)
    val bias: Tensor[T] = weightAndBias.narrow(1, 2, 1).squeeze(1)

    // weightAndBias should be 1-dim, which will be used for creating primitive.
    val all = weightAndBias.view(2 * nOutput)


    if (initWeight != null) {
      require(initWeight.size(1) == nOutput)
      weight.copy(initWeight)
    }

    if (initBias != null) {
      require(initBias.size(1) == nOutput)
      bias.copy(initBias)
    }

    (all, weight, bias)
  }
}

object SpatialBatchNormalization {
  def apply[@specialized(Float, Double) T: ClassTag](
    nOutput: Int,
    eps: Double = 1e-5,
    momentum: Double = 0.1,
    affine: Boolean = true,
    initWeight: Tensor[T] = null,
    initBias: Tensor[T] = null,
    initGradWeight: Tensor[T] = null,
    initGradBias: Tensor[T] = null)
    (implicit ev: TensorNumeric[T]): SpatialBatchNormalization[T] = {

    new SpatialBatchNormalization[T](
      nOutput, eps, momentum, affine, initWeight, initBias, initGradWeight, initGradBias)
  }

  def apply[@specialized(Float, Double) T: ClassTag](
    affine: Option[Int])(implicit ev: TensorNumeric[T]): SpatialBatchNormalization[T] = {
    new SpatialBatchNormalization[T](nOutput = affine.getOrElse(1), affine = affine.isDefined)
  }
}
