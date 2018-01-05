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
import com.intel.analytics.bigdl.nn.{ErrorInfo, InitializationMethod, RandomUniform, VariableFormat}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

class Linear[T: ClassTag](
  val inputSize: Int,
  val outputSize: Int,
  val withBias: Boolean = true,
  var wRegularizer: Regularizer[T] = null,
  var bRegularizer: Regularizer[T] = null,
  private val initWeight: Tensor[T] = null,
  private val initBias: Tensor[T] = null,
  private val initGradWeight: Tensor[T] = null,
  private val initGradBias: Tensor[T] = null
)(implicit ev: TensorNumeric[T]) extends TensorModule[T] with Initializable {
  val weight: Tensor[T] =
    if (initWeight != null) initWeight else Tensor[T](outputSize, inputSize)
  val bias: Tensor[T] =
    if (initBias != null) initBias else if (withBias) Tensor[T](outputSize) else null
  val addBuffer: Tensor[T] = Tensor[T]()

  val gradWeight: Tensor[T] =
    if (initGradWeight != null) initGradWeight else Tensor[T]()
  val gradBias: Tensor[T] =
    if (initGradBias != null) initGradBias else if (withBias) Tensor[T]() else null

  val diffWeight: Tensor[T] = Tensor[T].resizeAs(gradWeight)
  val diffBias: Tensor[T] = Tensor[T].resizeAs(gradBias)

  {
    val stdv = 1.0 / math.sqrt(weight.size(2))
    val wInit: InitializationMethod = RandomUniform(-stdv, stdv)
    val bInit: InitializationMethod = RandomUniform(-stdv, stdv)
    setInitMethod(wInit, bInit)
  }

  override def reset(): Unit = {
    if (initWeight == null) {
      weightInitMethod.init(weight, VariableFormat.OUT_IN)
    }
    if (initBias == null) {
      Option(bias).foreach(biasInitMethod.init(_, VariableFormat.ONE_D))
    }
    zeroGradParameters()
  }

  val engine = MklDnn.EngineCreate(EngineType.cpu, 0)

  var forwardStream = 0L
  var backDataStream = 0L
  var backWeightStream = 0L

  var forwardPrim = 0L
  var backDataPrim = 0L
  var backWeightPrim = 0L

  // forward memory primitive
  @transient var userSrcMemoryPrim = 0L
  @transient var userWeightMemoryPrim = 0L
  @transient var userBiasMemoryPrim = 0L
  @transient var userDstMemoryPrim = 0L

  @transient var userDiffSrcMemoryPrim = 0L
  @transient var userDiffWeightMemoryPrim = 0L
  @transient var userDiffBiasMemoryPrim = 0L
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

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (input.dim() == 1) {
      output.resize(Array(outputSize))
      if (withBias) output.copy(bias) else output.zero()
      output.addmv(ev.fromType[Int](1), weight, input)
    } else if (input.dim() == 2) {
      val nFrame = input.size(1)
      val t = Array(nFrame, weight.size(1))
      output.resize(t)
    }

    if (forwardPrim == 0L) {
      // TODO mkldnn memory
      val srcMemDesc = MklDnn.MemoryDescInit(input.dim(), input.size(),
        MklDnn.DataType.f32, MklDnn.MemoryFormat.nc)

      val weightMemDesc = MklDnn.MemoryDescInit(weight.dim(), weight.size(),
        MklDnn.DataType.f32, MklDnn.MemoryFormat.oi)

      val biasMemDesc = MklDnn.MemoryDescInit(bias.dim(), bias.size(),
        MklDnn.DataType.f32, MklDnn.MemoryFormat.x)

      val dstMemDesc = MklDnn.MemoryDescInit(output.dim(), output.size(),
        MklDnn.DataType.f32, MklDnn.MemoryFormat.nc)

      val opDesc = MklDnn.LinearForwardDescInit(MklDnn.PropKind.forward,
        srcMemDesc, weightMemDesc, biasMemDesc, dstMemDesc)
      val opPrimDesc = MklDnn.PrimitiveDescCreate(opDesc, engine, 0)

      userSrcMemoryPrim = initDataMemory(input.dim(), input.size(),
        MklDnn.MemoryFormat.nc, MklDnn.DataType.f32, engine, input)
      userWeightMemoryPrim = initDataMemory(weight.dim(), weight.size(),
        MklDnn.MemoryFormat.oi, MklDnn.DataType.f32, engine, weight)
      userBiasMemoryPrim = initDataMemory(bias.dim(), bias.size(),
        MklDnn.MemoryFormat.x, MklDnn.DataType.f32, engine, bias)
      userDstMemoryPrim = initDataMemory(output.dim(), output.size(),
        MklDnn.MemoryFormat.nc, MklDnn.DataType.f32, engine, output)

      val srcs = Array(userSrcMemoryPrim, userWeightMemoryPrim, userBiasMemoryPrim)
      val indexes = Array(0, 0, 0)
      val dsts = Array(userDstMemoryPrim)

      forwardPrim = MklDnn.PrimitiveCreate2(opPrimDesc, srcs, indexes, srcs.length,
        dsts, dsts.length)

      MklDnn.PrimitiveDestroy(opPrimDesc)
    }

    if (forwardStream == 0L) {
      forwardStream = MklDnn.StreamCreate(StreamType.eager)
    }

    setHandle(input, userSrcMemoryPrim)
    setHandle(weight, userWeightMemoryPrim)
    setHandle(bias, userBiasMemoryPrim)
    setHandle(output, userDstMemoryPrim)

    MklDnn.StreamSubmit(forwardStream, 1, Array(forwardPrim))

    releaseHandles(input, userSrcMemoryPrim)
    releaseHandles(weight, userWeightMemoryPrim)
    releaseHandles(bias, userBiasMemoryPrim)
    releaseHandles(output, userDstMemoryPrim)

    output
  }

  var backwardPrim = 0L
  var backwardStream = 0L

  def backward2(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input)
    gradWeight.resizeAs(weight)
    if (withBias) {
      gradBias.resizeAs(bias)
    }

    if (backDataPrim == 0L || backWeightPrim == 0) {
      val diffSrcMemDesc = MklDnn.MemoryDescInit(gradInput.dim(), gradInput.size(),
        MklDnn.DataType.f32, MklDnn.MemoryFormat.any)

      val weightMemDesc = MklDnn.MemoryDescInit(weight.dim(), weight.size(),
        MklDnn.DataType.f32, MklDnn.MemoryFormat.any)

      val diffDstMemDesc = MklDnn.MemoryDescInit(gradOutput.dim(), gradOutput.size(),
        MklDnn.DataType.f32, MklDnn.MemoryFormat.any)

      val dataDesc = MklDnn.LinearBackwardDataDescInit(diffSrcMemDesc, weightMemDesc,
        diffDstMemDesc)
      val dataPrimDesc = MklDnn.PrimitiveDescCreate(dataDesc, engine, 0)

      userDiffDstMemoryPrim = initDataMemory(gradOutput.dim(), gradOutput.size(),
        MklDnn.MemoryFormat.nc, MklDnn.DataType.f32, engine, gradOutput)

      userDiffSrcMemoryPrim = initDataMemory(gradInput.dim(), gradInput.size(),
        MklDnn.MemoryFormat.nc, MklDnn.DataType.f32, engine, gradInput)

      val dataSrcs = Array(userDiffDstMemoryPrim, userWeightMemoryPrim)
      val dataIndexes = Array(0, 0)
      val dataDsts = Array(userDiffSrcMemoryPrim)

      backDataPrim = MklDnn.PrimitiveCreate2(dataPrimDesc, dataSrcs, dataIndexes, dataSrcs.length,
        dataDsts, dataDsts.length)

      MklDnn.PrimitiveDescDestroy(dataPrimDesc)

      val srcMemDesc = MklDnn.MemoryDescInit(input.dim(), input.size(),
        MklDnn.DataType.f32, MklDnn.MemoryFormat.any)

      val diffWeightMemDesc = MklDnn.MemoryDescInit(gradWeight.dim(), gradWeight.size(),
        MklDnn.DataType.f32, MklDnn.MemoryFormat.any)

      val diffBiasMemDesc = MklDnn.MemoryDescInit(gradBias.dim(), gradWeight.size(),
        MklDnn.DataType.f32, MklDnn.MemoryFormat.x)

      val opDesc = MklDnn.LinearBackwardWeightsDescInit(
        srcMemDesc, diffWeightMemDesc, diffBiasMemDesc, diffDstMemDesc)
      val opPrimDesc = MklDnn.PrimitiveDescCreate(opDesc, engine, 0)

      userDiffWeightMemoryPrim = initDataMemory(gradWeight.dim(),
        gradWeight.size(), MklDnn.MemoryFormat.oi, MklDnn.DataType.f32, engine,
        gradWeight)

      userDiffBiasMemoryPrim = initDataMemory(gradBias.dim(), gradBias.size(),
        MklDnn.MemoryFormat.x, MklDnn.DataType.f32, engine, gradBias)

      val srcs = Array(userSrcMemoryPrim, userDiffDstMemoryPrim)
      val indexes = Array(0, 0)
      val dsts = Array(userDiffWeightMemoryPrim, userDiffBiasMemoryPrim)

      backWeightPrim = MklDnn.PrimitiveCreate2(opPrimDesc,
        srcs, indexes, srcs.length, dsts, dsts.length)

      MklDnn.PrimitiveDescDestroy(opPrimDesc)
    }

    if (backwardStream == 0) {
      backwardStream = MklDnn.StreamCreate(StreamType.eager)
    }

    setHandle(input, userSrcMemoryPrim)
    setHandle(weight, userWeightMemoryPrim)

    setHandle(gradOutput, userDiffDstMemoryPrim)
    setHandle(gradInput, userDiffSrcMemoryPrim)
    setHandle(gradWeight, userDiffWeightMemoryPrim)
    setHandle(gradBias, userDiffBiasMemoryPrim)

    MklDnn.StreamSubmit(backwardStream, 2, Array(backDataPrim, backWeightPrim))

    releaseHandles(input, userSrcMemoryPrim)
    releaseHandles(weight, userWeightMemoryPrim)
    releaseHandles(gradInput, userDiffSrcMemoryPrim)
    releaseHandles(gradOutput, userDiffDstMemoryPrim)
    releaseHandles(gradWeight, userDiffWeightMemoryPrim)
    releaseHandles(gradBias, userDiffBiasMemoryPrim)

    gradInput
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input)

    if (backDataPrim == 0L) {
      val diffSrcMemDesc = MklDnn.MemoryDescInit(gradInput.dim(), gradInput.size(),
        MklDnn.DataType.f32, MklDnn.MemoryFormat.nc)

      val weightMemDesc = MklDnn.MemoryDescInit(weight.dim(), weight.size(),
        MklDnn.DataType.f32, MklDnn.MemoryFormat.oi)

      val dstMemDesc = MklDnn.MemoryDescInit(gradOutput.dim(), gradOutput.size(),
        MklDnn.DataType.f32, MklDnn.MemoryFormat.nc)

      val opDesc = MklDnn.LinearBackwardDataDescInit(diffSrcMemDesc, weightMemDesc, dstMemDesc)
      val opPrimDesc = MklDnn.PrimitiveDescCreate(opDesc, engine, 0)

      userDiffDstMemoryPrim = initDataMemory(gradOutput.dim(), gradOutput.size(),
        MklDnn.MemoryFormat.nc, MklDnn.DataType.f32, engine, gradOutput)

      userDiffSrcMemoryPrim = initDataMemory(gradInput.dim(), gradInput.size(),
        MklDnn.MemoryFormat.nc, MklDnn.DataType.f32, engine, gradInput)

      val srcs = Array(userDiffDstMemoryPrim, userWeightMemoryPrim)
      val indexes = Array(0, 0)
      val dsts = Array(userDiffSrcMemoryPrim)

      backDataPrim = MklDnn.PrimitiveCreate2(opPrimDesc, srcs, indexes, srcs.length,
        dsts, dsts.length)

      MklDnn.PrimitiveDescDestroy(opPrimDesc)
    }

    if (backDataStream == 0) {
      backDataStream = MklDnn.StreamCreate(StreamType.eager)
    }

    setHandle(gradOutput, userDiffDstMemoryPrim)
    setHandle(weight, userWeightMemoryPrim)
    setHandle(gradInput, userDiffSrcMemoryPrim)

    MklDnn.StreamSubmit(backDataStream, 1, Array(backDataPrim))

    releaseHandles(gradInput, userDiffSrcMemoryPrim)
    releaseHandles(gradOutput, userDiffDstMemoryPrim)
    releaseHandles(weight, userWeightMemoryPrim)

    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    accGradParameters2(input, gradOutput)
  }

  var computing = 0.0
  var aggregating = 0.0

  def accGradParameters2(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    gradWeight.resizeAs(weight)
    if (withBias) {
      gradBias.resizeAs(bias)
    }

    diffWeight.resizeAs(weight)
    if (withBias) {
      diffBias.resizeAs(bias)
    }

    if (backWeightPrim == 0) {
      val srcMemDesc = MklDnn.MemoryDescInit(input.dim(), input.size(),
        MklDnn.DataType.f32, MklDnn.MemoryFormat.nc)

      val diffWeightMemDesc = MklDnn.MemoryDescInit(gradWeight.dim(), gradWeight.size(),
        MklDnn.DataType.f32, MklDnn.MemoryFormat.oi)

      val diffBiasMemDesc = MklDnn.MemoryDescInit(gradBias.dim(), gradWeight.size(),
        MklDnn.DataType.f32, MklDnn.MemoryFormat.x)

      val diffDstMemDesc = MklDnn.MemoryDescInit(gradOutput.dim(), gradOutput.size(),
        MklDnn.DataType.f32, MklDnn.MemoryFormat.nc)

      val opDesc = MklDnn.LinearBackwardWeightsDescInit(
        srcMemDesc, diffWeightMemDesc, diffBiasMemDesc, diffDstMemDesc)
      val opPrimDesc = MklDnn.PrimitiveDescCreate(opDesc, engine, 0)

      userDiffWeightMemoryPrim = initDataMemory(diffWeight.dim(),
        diffWeight.size(), MklDnn.MemoryFormat.oi, MklDnn.DataType.f32, engine,
        diffWeight)

      userDiffBiasMemoryPrim = initDataMemory(diffBias.dim(), diffBias.size(),
        MklDnn.MemoryFormat.x, MklDnn.DataType.f32, engine, diffBias)

      val srcs = Array(userSrcMemoryPrim, userDiffDstMemoryPrim)
      val indexes = Array(0, 0)
      val dsts = Array(userDiffWeightMemoryPrim, userDiffBiasMemoryPrim)

      backWeightPrim = MklDnn.PrimitiveCreate2(opPrimDesc,
        srcs, indexes, srcs.length, dsts, dsts.length)
    }

    if (backWeightStream == 0) {
      backWeightStream = MklDnn.StreamCreate(StreamType.eager)
    }

    setHandle(input, userSrcMemoryPrim)
    setHandle(gradOutput, userDiffDstMemoryPrim)
    setHandle(diffWeight, userDiffWeightMemoryPrim)
    setHandle(diffBias, userDiffBiasMemoryPrim)

    val start1 = System.nanoTime()
    MklDnn.StreamSubmit(backWeightStream, 1, Array(backWeightPrim))
    val end1 = System.nanoTime()
    computing += end1 - start1

    releaseHandles(input, userSrcMemoryPrim)
    releaseHandles(gradOutput, userDiffDstMemoryPrim)
    releaseHandles(diffWeight, userDiffWeightMemoryPrim)
    releaseHandles(diffBias, userDiffBiasMemoryPrim)

    val start2 = System.nanoTime()
//    gradWeight.add(ev.fromType(1), diffWeight)
//    if (withBias) {
//      gradBias.add(ev.fromType(1), diffBias)
//    }
    val end2 = System.nanoTime()
    aggregating += end2 - start2
  }

  def accGradParameters1(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    require(input.dim() == 1 || input.dim() == 2,
      "Linear: " + ErrorInfo.constrainInputAsVectorOrBatch +
        s"input dim ${input.dim()}")

    gradWeight.resize(outputSize, inputSize)
    if (withBias) {
      gradBias.resize(outputSize)
    }

    if (input.dim() == 1) {
      if (scaleW != 0) {
        gradWeight.addr(ev.fromType[Double](scaleW), gradOutput, input)
      }

      if (withBias && scaleB != 0) {
        gradBias.add(ev.fromType[Double](scaleB), gradOutput)
      }
    }
    else if (input.dim() == 2) {
      if (scaleW != 0) {
        gradWeight.addmm(ev.fromType[Double](scaleW), gradOutput.t, input)
      }

      val nFrame = input.size(1)
      if (addBuffer.nElement() != nFrame) {
        addBuffer.resize(Array(nFrame)).fill(ev.one)
      }

      if (withBias && scaleB != 0) {
        gradBias.addmv(ev.fromType[Double](scaleB), gradOutput.t, addBuffer)
      }
    }

    if (null != wRegularizer && scaleW != 0) {
      wRegularizer.accRegularization(weight, gradWeight, scaleW)
    }
    if (null != bRegularizer && scaleB != 0) {
      bRegularizer.accRegularization(bias, gradBias, scaleB)
    }
  }
  override def updateParameters(learningRate: T): Unit = {
    weight.add(ev.negative(learningRate), gradWeight)
    if (withBias) bias.add(ev.negative(learningRate), gradBias)
  }

  override def zeroGradParameters(): Unit = {
    gradWeight.resize(outputSize, inputSize)
    gradWeight.zero()
    if (withBias) {
      gradBias.resize(outputSize)
      gradBias.zero()
    }
  }

  override def clearState() : this.type = {
    super.clearState()
    addBuffer.set()
    this
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    if (null == bias) {
      (Array(this.weight), Array(this.gradWeight))
    } else {
      (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
    }
  }

  override def getParametersTable(): Table = {
    if (null == bias) {
      T(getName() -> T("weight" -> weight, "gradWeight" -> gradWeight))
    } else {
      T(getName() -> T("weight" -> weight, "bias" -> bias,
        "gradWeight" -> gradWeight, "gradBias" -> gradBias))
    }
  }

  override def equals(obj: Any): Boolean = {

    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[Linear[T]]) {
      return false
    }
    val other = obj.asInstanceOf[Linear[T]]
    if (this.eq(other)) {
      return true
    }

    gradWeight == other.gradWeight &&
      gradBias == other.gradBias &&
      weight == other.weight &&
      bias == other.bias
  }

  override def hashCode() : Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + gradWeight.hashCode()
    hash = hash * seed + gradBias.hashCode()
    hash = hash * seed + weight.hashCode()
    hash = hash * seed + bias.hashCode()

    hash
  }

  override def toString(): String = {
    s"${getPrintName}($inputSize -> $outputSize)"
  }
}

object Linear {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputSize: Int,
    outputSize: Int,
    withBias: Boolean = true,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    initWeight: Tensor[T] = null,
    initBias: Tensor[T] = null,
    initGradWeight: Tensor[T] = null,
    initGradBias: Tensor[T] = null
  )(implicit ev: TensorNumeric[T]) : Linear[T] = {
    new Linear[T](inputSize, outputSize,
      withBias, wRegularizer, bRegularizer, initWeight, initBias, initGradWeight, initGradBias)
  }
}
