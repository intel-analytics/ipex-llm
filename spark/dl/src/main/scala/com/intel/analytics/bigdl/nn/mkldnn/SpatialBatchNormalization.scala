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

import java.io.{IOException, ObjectInputStream}

import com.intel.analytics.bigdl.mkl.{Memory, MklDnn}
import com.intel.analytics.bigdl.nn.abstractnn.{Initializable, TensorModule}
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.collection.mutable.ArrayBuffer
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
  val mean: MklDnnTensor[T] = MklDnnTensor[T](Array(nOutput))
  val variance: MklDnnTensor[T] = MklDnnTensor[T](Array(nOutput))

  val all = createParams(initWeight, initBias)
  val gradAll = createParams(initGradWeight, initGradBias)
  val diffAll = MklDnnTensor[T](all.size())
  val prvAll = MklDnnTensor[T](all.size())

  @transient var engine = 0L
  @transient var stream = 0L

  @transient var forwardPrims: ArrayBuffer[Long] = ArrayBuffer.empty
  @transient var forwardReorderPrims: ArrayBuffer[Long] = ArrayBuffer.empty
  @transient var backwardPrims: ArrayBuffer[Long] = ArrayBuffer.empty
  @transient var backwardReorderPrims: ArrayBuffer[Long] = ArrayBuffer.empty
  @transient var forwardPrimDesc = 0L

  @throws(classOf[IOException])
  private def readObject(in: ObjectInputStream): Unit = {
    in.defaultReadObject()
    forwardPrims = ArrayBuffer.empty
    forwardReorderPrims = ArrayBuffer.empty
    backwardPrims = ArrayBuffer.empty
    backwardReorderPrims = ArrayBuffer.empty
  }

  var _shouldConvert: Boolean = false
  def shouldConvert: Boolean = _shouldConvert
  def setShouldConvert(v: Boolean): this.type = {
    _shouldConvert = v
    this
  }

  object OpPrim {
    val input, output, weightAndBias, mean, variance,
        diffInput, diffOutput, diffWeightAndBias = new MemoryPrimitive[T]()
  }

  private def init1(primDesc: Long): Long = {
    MklDnn.PrimitiveCreate0(primDesc)
  }

  private def init4(tensor: Tensor[T], dataType: Int, format: Int, engine: Long): Long = {
    // TODO refactor for linear
    val (dim, size) = if (tensor.dim() == 1 && (format == MklDnn.MemoryFormat.nc ||
      format == MklDnn.MemoryFormat.oi)) {
      (2, Array(1) ++ tensor.size())
    } else if (tensor.dim() == 2 && (format == MklDnn.MemoryFormat.oihw)) {
      (4, tensor.size() ++ Array(1, 1))
    } else {
      (tensor.dim(), tensor.size())
    }

    val desc = MklDnn.MemoryDescInit(dim, size, dataType, format)
    val primDesc = MklDnn.MemoryPrimitiveDescCreate(desc, engine)
    val primitive = MklDnn.PrimitiveCreate0(primDesc)

    MklDnn.PrimitiveDescDestroy(primDesc)
    primitive
  }

  def initUser(tensor: Tensor[T], dataType: Int, format: Int, engine: Long): Long = {
    val primDesc = tensor.getPrimitiveDesc()
    val primitive = if (primDesc != 0L) { // if the tensor comes from mkldnn layer
      init1(primDesc)
    } else {
      init4(tensor, dataType, format, engine)
    }
    primitive
  }

  def initInternal(userPrim: Long, layerPrimDesc: Long, queryType: Int,
    userToPrim: Boolean = true): (Long, Long) = {
    val primDescFromLayer = MklDnnOps.primitiveDescQueryPd(layerPrimDesc, queryType, 0)
    val res = MklDnnOps.prepareReorder(userPrim, primDescFromLayer, userToPrim)
    val memoryPrimitive = res._2
    val reorderPrimitive = res._1
    (memoryPrimitive, reorderPrimitive)
  }

  def initUser(tensor: Tensor[T], layerPrimDesc: Long, queryType: Int, index: Int): Long = {
    val primDesc = MklDnnOps.primitiveDescQueryPd(layerPrimDesc, queryType, 0)
    tensor.setPrimitiveDesc(primDesc)
    val primitive = MklDnn.PrimitiveCreate0(primDesc)
    primitive
  }

  // TODO train and inference mode

  @transient var internalInput, internalOutput: MklDnnTensor[T] = _

  var defaultFormat = MklDnn.MemoryFormat.nchw

  private def toMklDnnTensor(t: Tensor[T]): MklDnnTensor[T] = t.asInstanceOf[MklDnnTensor[T]]

  @transient var inputUserPrim = 0L
  @transient var inputReorderMemoryPrim = 0L
  @transient var inputReorderPrim = 0L
  @transient var outputUserPrim = 0L
  @transient var weightAndBiasUserPrim = 0L
  @transient var meanUserPrim = 0L
  @transient var varianceUserPrim = 0L
  @transient var previousSize: Array[Int] = _
  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val s1 = System.nanoTime()
    if (previousSize == null) {
      previousSize = input.size()
    } else if (previousSize.deep != input.size().deep) {
      previousSize = input.size()
      for (i <- forwardPrims ++ backwardPrims ++ forwardReorderPrims ++ backwardReorderPrims) {
        MklDnn.PrimitiveDestroy(i)
      }
      forwardPrims = ArrayBuffer.empty
      backwardPrims = ArrayBuffer.empty
      forwardReorderPrims = ArrayBuffer.empty
      backwardReorderPrims = ArrayBuffer.empty
    }

    if (forwardPrims.isEmpty) {
      if (output.getTensorType == MklDnnType) {
        toMklDnnTensor(output).release()
      }
      output = MklDnnTensor[T](input.size())

      engine = this.getDnnEngine(0)
      stream = this.getStream()

      val srcMemDesc = if (input.getPrimitiveDesc() == 0L) {
        MklDnn.MemoryDescInit(input.dim(), input.size(), MklDnn.DataType.f32, defaultFormat)
      } else {
        MklDnnOps.primitiveDescQueryMemory(input.getPrimitiveDesc())
      }

      val bnDesc = MklDnn.BatchNormForwardDescInit(MklDnn.PropKind.forward,
        srcMemDesc, eps.toFloat, MklDnn.BatchNormFlag.mkldnn_use_scaleshift)
      val opPrimDesc = MklDnn.PrimitiveDescCreate(bnDesc, engine, 0)

      forwardPrimDesc = opPrimDesc

      val dataFormat = defaultFormat
      val paramsFormat = MklDnn.MemoryFormat.x
      val dataType = MklDnn.DataType.f32

      inputUserPrim = initUser(input, dataType, dataFormat, engine)
      val i1 = initInternal(inputUserPrim, opPrimDesc, MklDnn.Query.src_pd)
      inputReorderMemoryPrim = i1._1
      inputReorderPrim = i1._2
      outputUserPrim = initUser(output, opPrimDesc, MklDnn.Query.dst_pd, 0)

      // because they're 1-d, so we need not to initialize it.
      weightAndBiasUserPrim = initUser(all, dataType, paramsFormat, engine)
      meanUserPrim = initUser(mean, dataType, paramsFormat, engine)
      varianceUserPrim = initUser(variance, dataType, paramsFormat, engine)

      val inputMemoryPrim = if (inputReorderPrim != 0) {
        forwardReorderPrims += inputReorderPrim
        inputReorderMemoryPrim
      } else {
        inputUserPrim
      }

      val srcs = Array(inputMemoryPrim, weightAndBiasUserPrim)
      val indexes = Array.fill(srcs.length)(0)
      val dsts = Array(outputUserPrim, meanUserPrim, varianceUserPrim)

      forwardPrims += MklDnn.PrimitiveCreate2(opPrimDesc, srcs, indexes, srcs.length,
        dsts, dsts.length)

      if (inputReorderPrim == 0 && input.getTensorType == MklDnnType) {
        internalInput = input.asInstanceOf[MklDnnTensor[T]]
      } else {
        if (internalInput != null) {
          internalInput.release()
        }
        internalInput = MklDnnTensor[T](input.size())
      }

      if (internalInput.size().deep != input.size().deep) {
        internalInput.release()
        internalInput = MklDnnTensor[T](input.size())
      }
    }

    if (input.getTensorType == DenseType) {
      internalInput.set(input)
    }

    var inputPtr = 0L
    if (inputReorderPrim != 0) {
      if (input.getTensorType == DenseType) {
        inputPtr = MklDnn.MemorySetDataHandle(inputUserPrim,
          input.storage().array().asInstanceOf[Array[Float]],
          input.storageOffset() - 1)
        Memory.SetDataHandle(inputReorderMemoryPrim, internalInput.ptr, 0)
      } else {
        Memory.SetDataHandle(inputUserPrim,
          input.asInstanceOf[MklDnnTensor[T]].ptr,
          0)
        Memory.SetDataHandle(inputReorderMemoryPrim, internalInput.ptr, 0)
      }
    } else {
      if (input.getTensorType == DenseType) {
        MklDnnTensor.syncFromHeap(internalInput, input.storage().array(), input.storageOffset() - 1)
        Memory.SetDataHandle(inputUserPrim, internalInput.ptr, 0)
      } else if (input.getTensorType == MklDnnType) {
        Memory.SetDataHandle(inputUserPrim, input.asInstanceOf[MklDnnTensor[T]].ptr, 0)
      }
    }

    MklDnnTensor.syncFromHeap(prvAll, all.storage().array(), all.storageOffset() - 1)
    Memory.SetDataHandle(weightAndBiasUserPrim, prvAll.ptr, 0)
    Memory.SetDataHandle(meanUserPrim, mean.ptr, 0)
    Memory.SetDataHandle(varianceUserPrim, variance.ptr, 0)
    Memory.SetDataHandle(outputUserPrim, output.asInstanceOf[MklDnnTensor[T]].ptr, 0)

    if (inputReorderPrim != 0) {
      MklDnn.StreamSubmit(stream, forwardReorderPrims.length, forwardReorderPrims.toArray)
      if (input.getTensorType == DenseType && inputPtr != 0) {
        MklDnn.MemoryReleaseDataHandle(input.storage().array().asInstanceOf[Array[Float]], inputPtr)
      }
    }

    MklDnn.StreamSubmit(stream, forwardPrims.length, forwardPrims.toArray)

    if (shouldConvert) {
      output.asInstanceOf[MklDnnTensor[T]].syncToHeap()
    }

    val end1 = (System.nanoTime() - s1)/1e6
    if (System.getProperty("debug") == "2") {
      DnnTools.debugFwInfo(this.getName(), end1, input.getFormat(), output.getFormat())
    }
    output
  }

  @transient var gradOutputUserPrim = 0L
  @transient var gradOutputReorderPrim = 0L
  @transient var gradOutputReorderMemoryPrim = 0L
  @transient var gradInputUserPrim = 0L
  @transient var gradWeightAndBiasUserPrim = 0L
  @transient var internalGradInput, internalGradOutput: MklDnnTensor[T] = _
  def backward1(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val s1 = System.nanoTime()
    if (backwardPrims.isEmpty) {
      if (gradInput.getTensorType == MklDnnType) {
        toMklDnnTensor(gradInput).release()
      }
      gradInput = MklDnnTensor[T](input.size())

      val srcMemDesc = if (input.getPrimitiveDesc() == 0) {
        MklDnn.MemoryDescInit(input.dim(), input.size(),
          MklDnn.DataType.f32, defaultFormat)
      } else {
        MklDnnOps.primitiveDescQueryMemory(input.getPrimitiveDesc())
      }

      // [PERF] the format of gradInput should be the same as input
      val diffDstMemDesc = MklDnn.MemoryDescInit(input.dim(), input.size(),
        MklDnn.DataType.f32, MklDnn.getFormat(srcMemDesc))

      val desc = MklDnn.BatchNormBackwardDescInit(MklDnn.PropKind.backward,
        diffDstMemDesc, srcMemDesc, eps.toFloat, MklDnn.BatchNormFlag.mkldnn_use_scaleshift)
      val primDesc = MklDnn.PrimitiveDescCreate(desc, engine, forwardPrimDesc)

      val dataFormat = defaultFormat
      val paramsFormat = MklDnn.MemoryFormat.x
      val dataType = MklDnn.DataType.f32

      gradOutputUserPrim = initUser(gradOutput, dataType, dataFormat, engine)
      val g1 = initInternal(gradOutputUserPrim, primDesc, MklDnn.Query.diff_dst_pd)
      gradOutputReorderMemoryPrim = g1._1
      gradOutputReorderPrim = g1._2
      gradWeightAndBiasUserPrim = initUser(diffAll, dataType, paramsFormat, engine)
      gradInputUserPrim = initUser(gradInput, primDesc, MklDnn.Query.diff_src_pd, 0)

      val inputMemoryPrim = if (inputReorderPrim != 0) {
        inputReorderMemoryPrim
      } else {
        inputUserPrim
      }

      val gradOutputMemoryPrim = if (gradOutputReorderPrim != 0) {
        backwardReorderPrims += gradOutputReorderPrim
        gradOutputReorderMemoryPrim
      } else {
        gradOutputUserPrim
      }

      val dataSrcs = Array(inputMemoryPrim, meanUserPrim,
        varianceUserPrim, gradOutputMemoryPrim,
        weightAndBiasUserPrim)
      val dataIndexes = Array.fill(dataSrcs.length)(0)
      val dataDsts = Array(gradInputUserPrim, gradWeightAndBiasUserPrim)

      backwardPrims += MklDnn.PrimitiveCreate2(primDesc, dataSrcs, dataIndexes, dataSrcs.length,
        dataDsts, dataDsts.length)

      if (backwardReorderPrims.isEmpty && gradOutput.getTensorType == MklDnnType) {
        internalGradOutput = toMklDnnTensor(gradOutput)
      } else {
        if (internalGradOutput != null) {
          internalGradOutput.release()
        }
        internalGradOutput = MklDnnTensor[T](input.size())
      }

      if (internalGradOutput.size().deep != input.size().deep) {
        internalGradOutput.release()
        internalGradOutput = MklDnnTensor[T](input.size())
      }
    }

    if (gradOutput.getTensorType == DenseType) {
      internalGradOutput.set(gradOutput)
    }

    var gradOutputPtr = 0L
    if (gradOutputReorderPrim != 0) {
      if (gradOutput.getTensorType == DenseType) {
        gradOutputPtr = MklDnn.MemorySetDataHandle(gradOutputUserPrim,
          gradOutput.storage().array().asInstanceOf[Array[Float]],
          gradOutput.storageOffset() - 1)
        Memory.SetDataHandle(gradOutputReorderMemoryPrim, internalGradOutput.ptr, 0)
      } else {
        Memory.SetDataHandle(gradOutputUserPrim,
          gradOutput.asInstanceOf[MklDnnTensor[T]].ptr,
          0)
        Memory.SetDataHandle(gradOutputReorderMemoryPrim, internalGradOutput.ptr, 0)
      }
    } else {
      if (gradOutput.getTensorType == DenseType) {
        MklDnnTensor.syncFromHeap(internalGradOutput, gradOutput.storage().array(),
          gradOutput.storageOffset() - 1)
        Memory.SetDataHandle(gradOutputUserPrim, internalGradOutput.ptr, 0)
      } else if (gradOutput.getTensorType == MklDnnType) {
        Memory.SetDataHandle(gradOutputUserPrim, gradOutput.asInstanceOf[MklDnnTensor[T]].ptr, 0)
      }
    }

    Memory.SetDataHandle(gradWeightAndBiasUserPrim, diffAll.ptr, 0)
    Memory.SetDataHandle(meanUserPrim, mean.ptr, 0)
    Memory.SetDataHandle(varianceUserPrim, variance.ptr, 0)
    Memory.SetDataHandle(gradInputUserPrim, gradInput.asInstanceOf[MklDnnTensor[T]].ptr, 0)

    if (gradOutputReorderPrim != 0) {
      MklDnn.StreamSubmit(stream, backwardReorderPrims.length, backwardReorderPrims.toArray)
      if (gradOutput.getTensorType == DenseType && gradOutputPtr != 0) {
        MklDnn.MemoryReleaseDataHandle(gradOutput.storage().array().asInstanceOf[Array[Float]],
          gradOutputPtr)
      }
    }
    MklDnn.StreamSubmit(stream, backwardPrims.length, backwardPrims.toArray)

    diffAll.syncToHeap()
    gradAll.add(diffAll)

    if (shouldConvert) {
      gradInput.asInstanceOf[MklDnnTensor[T]].syncToHeap()
    }

    val end1 = (System.nanoTime() - s1)/1e6
    if (System.getProperty("debug") == "2") {
      DnnTools.debugBwInfo(this.getName(), end1, gradOutput.getFormat(), gradInput.getFormat())
    }
    gradInput
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    backward1(input, gradOutput)
  }

  // there's no relavant accGrasdParameters in mkl-dnn. we use @backward instead of
  // @updateGradInput and @accGradParameters
  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
  }

  private type Params[R] = (Tensor[R], Tensor[R], Tensor[R])
  // in mkl dnn, the weight and bias should be all in the same array
  private def createParams(initWeight: Tensor[T], initBias: Tensor[T]): Tensor[T] = {
    val weightAndBias: Tensor[T] = if (affine) {
      Tensor[T](Array(2 * nOutput))
    } else {
      null
    }

    val concat = Tensor[T]().resize(Array(2, nOutput)).fill(ev.fromType(0))

    if (initWeight != null) {
      require(initWeight.size(1) == nOutput)
      concat.select(1, 1).copy(initWeight)
    } else {
      concat.select(1, 1).fill(ev.fromType(1))
    }

    if (initBias != null) {
      require(initBias.size(1) == nOutput)
      concat.select(1, 2).copy(initBias)
    } else {
      concat.select(1, 2).fill(ev.fromType(0))
    }

    weightAndBias.copy(concat.view(Array(2 * nOutput)))
    weightAndBias
  }

  override def zeroGradParameters(): Unit = {
    if (affine) {
      gradAll.zero()
    }
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    if (affine) {
      (Array(all), Array(gradAll))
    } else {
      null
    }
  }

  override def getParametersTable(): Table = {
    if (affine) {
      T(getName() -> T("weight" -> all,
        "gradWeight" -> gradAll,
        "runningMean" -> mean, "runningVar" -> variance))
    } else {
      T(getName() -> T("runningMean" -> mean, "runningVar" -> variance))
    }
  }

  override def toString(): String = {
    s"mkldnn.BatchNormalization($nOutput, $eps, $momentum, $affine)"
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
