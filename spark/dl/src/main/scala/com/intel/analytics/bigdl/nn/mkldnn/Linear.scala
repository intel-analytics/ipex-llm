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

import com.intel.analytics.bigdl.mkl.{DataType, Memory, MklDnn, PropKind, Query, Stream => DnnStream}
import com.intel.analytics.bigdl.nn.abstractnn.{Initializable, TensorModule}
import com.intel.analytics.bigdl.nn.{InitializationMethod, RandomUniform, VariableFormat}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{DenseType, MklDnnTensor, MklDnnType, Tensor}
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.collection.mutable.ArrayBuffer
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
  val weight: Tensor[T] = Tensor[T](Array(outputSize, inputSize))
  val bias: Tensor[T] = Tensor[T](Array(outputSize))
  val gradWeight: Tensor[T] = Tensor[T](Array(outputSize, inputSize))
  val gradBias: Tensor[T] = Tensor[T](Array(outputSize))

  if (initWeight != null) weight.copy(initWeight)
  if (initBias != null) bias.copy(initBias)
  if (initGradWeight != null) gradWeight.copy(initGradWeight)
  if (initGradBias != null) gradBias.copy(initGradBias)

  val prvWeight: MklDnnTensor[T] = MklDnnTensor[T](Array(outputSize, inputSize))
  val prvBias: MklDnnTensor[T] = MklDnnTensor[T](Array(outputSize))
  val diffWeight: MklDnnTensor[T] = MklDnnTensor[T](Array(outputSize, inputSize))
  val diffBias: MklDnnTensor[T] = MklDnnTensor[T](Array(outputSize))

  {
    val stdv = 1.0 / math.sqrt(weight.size(2))
    val wInit: InitializationMethod = RandomUniform(-stdv, stdv)
    val bInit: InitializationMethod = RandomUniform(-stdv, stdv)
    setInitMethod(wInit, bInit)
  }

  override def reset(): Unit = {
    if (initWeight == null) {
      val t = Tensor[T](Array(outputSize, inputSize))
      weightInitMethod.init(t, VariableFormat.OUT_IN)
      weight.copy(t)
    }
    if (initBias == null) {
      val t = Tensor[T](Array(outputSize))
      biasInitMethod.init(t, VariableFormat.ONE_D)
      bias.copy(t)
    }
    zeroGradParameters()
  }

  @transient var engine = 0L
  @transient var stream = 0L

  @transient var forwardPrim = 0L
  @transient var backDataPrim = 0L
  @transient var backWeightPrim = 0L

  @transient var forwardPrimDesc = 0L

  @transient private var forwardPrimBuffer: ArrayBuffer[Long] = _
  @transient private var forwardReorderPrimBuffer: ArrayBuffer[Long] = _
  @transient private var backwardDataPrimBuffer: ArrayBuffer[Long] = _
  @transient private var backwardDataReorderPrimBuffer: ArrayBuffer[Long] = _
  @transient private var backwardWeightPrimBuffer: ArrayBuffer[Long] = _
  @transient private var backwardWeightReorderPrimBuffer: ArrayBuffer[Long] = _

  @transient var internalInput: MklDnnTensor[T] = _
  @transient var internalOutput: MklDnnTensor[T] = _

  private def init1(primDesc: Long): Long = {
    MklDnn.PrimitiveCreate0(primDesc)
  }

  private def init4(tensor: Tensor[T], dataType: Int, format: Int, engine: Long): Long = {
    // TODO refactor for linear
    val (dim, size) = if (tensor.dim() == 1 && (format == Memory.Format.nc ||
      format == Memory.Format.oi)) {
      (2, Array(1) ++ tensor.size())
//    } else if (tensor.dim() == 2 && (format == Memory.Format.oihw)) {
//      (4, tensor.size() ++ Array(1, 1))
    } else {
      (tensor.dim(), tensor.size())
    }

    val desc = MklDnn.MemoryDescInit(dim, size, dataType, format)
    val primDesc = MklDnn.MemoryPrimitiveDescCreate(desc, engine)
    val primitive = MklDnn.PrimitiveCreate0(primDesc)

    MklDnn.PrimitiveDescDestroy(primDesc)
    primitive
  }

  private def init4(dim: Int, size: Array[Int], dataType: Int, format: Int, engine: Long): Long = {
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

  var _shouldConvert: Boolean = false
  def shouldConvert: Boolean = _shouldConvert
  def setShouldConvert(v: Boolean): this.type = {
    _shouldConvert = v
    this
  }

  @transient var inputUserPrim = 0L
  @transient var inputReorderMemoryPrim = 0L
  @transient var inputReorderPrim = 0L
  @transient var weightUserPrim = 0L
  @transient var weightReorderMemoryPrim = 0L
  @transient var weightReorderPrim = 0L
  @transient var biasUserPrim = 0L
  @transient var outputUserPrim = 0L
  @transient var previousSize: Array[Int] = _

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (previousSize == null) {
      previousSize = input.size()
    } else if (previousSize.deep != input.size().deep) {
      previousSize = input.size()
      if (forwardPrim != 0) {
        MklDnn.PrimitiveDestroy(forwardPrim)
        MklDnn.PrimitiveDestroy(backDataPrim)
        MklDnn.PrimitiveDestroy(backWeightPrim)
      }

      forwardPrim = 0L
      backDataPrim = 0L
      backWeightPrim = 0L
    }

    if (forwardPrim == 0L) {
      if (input.dim() == 4) {
        weight.resize(weight.size(1), input.size(2), input.size(3), input.size(4))
      }

      if (output.getTensorType == MklDnnType) {
        output.asInstanceOf[MklDnnTensor[T]].release()
      }

      if (input.dim() == 1) {
        output = MklDnnTensor[T](Array(outputSize))
      } else {
        output = MklDnnTensor[T](Array(input.size(1), weight.size(1)))
      }

      if (engine == 0L) engine = this.getDnnEngine(0)
      if (stream == 0L) stream = this.getStream()

      forwardPrimBuffer = ArrayBuffer.empty[Long]
      forwardReorderPrimBuffer = ArrayBuffer.empty[Long]

      val srcMemDesc = if (input.dim() == 1) {
        MklDnn.MemoryDescInit(input.dim() + 1, Array(1) ++ input.size(),
          DataType.F32, Memory.Format.any)
      } else {
        MklDnn.MemoryDescInit(input.dim(), input.size(),
          DataType.F32, Memory.Format.any)
      }
      val weightMemDesc = if (input.dim() == 4) {
        MklDnn.MemoryDescInit(4, weight.size() ++ Array(1, 1),
          DataType.F32, Memory.Format.any)
      } else {
        MklDnn.MemoryDescInit(weight.dim(), weight.size(),
          DataType.F32, Memory.Format.any)
      }
      val biasMemDesc = MklDnn.MemoryDescInit(bias.dim(), bias.size(),
        DataType.F32, Memory.Format.x)

      val dstMemDesc = if (input.dim() == 1) {
        MklDnn.MemoryDescInit(output.dim() + 1, Array(1) ++ output.size(),
          DataType.F32, Memory.Format.any)
      } else {
        MklDnn.MemoryDescInit(output.dim(), output.size(),
          DataType.F32, Memory.Format.any)
      }

      val format = input.dim() match {
        case 1 => Memory.Format.nc
        case 2 => Memory.Format.nc
        case 4 => Memory.Format.nchw
      }

      val weightFormat = input.dim() match {
        case 1 => Memory.Format.oi
        case 2 => Memory.Format.oi
        case 4 => Memory.Format.oihw
      }

      val opDesc = MklDnn.LinearForwardDescInit(PropKind.Forward,
        srcMemDesc, weightMemDesc, biasMemDesc, dstMemDesc)
      val opPrimDesc = MklDnn.PrimitiveDescCreate(opDesc, engine, 0)
      forwardPrimDesc = opPrimDesc

      inputUserPrim = initUser(input, DataType.F32, format, engine)
      val i1 = initInternal(inputUserPrim, opPrimDesc,
        Query.SrcPd)
      inputReorderMemoryPrim = i1._1
      inputReorderPrim = i1._2
      weightUserPrim = initUser(weight, DataType.F32, weightFormat, engine)
      val w1 = initInternal(weightUserPrim, opPrimDesc,
        Query.WeightsPd)
      weightReorderMemoryPrim = w1._1
      weightReorderPrim = w1._2
      biasUserPrim = initUser(bias, DataType.F32, Memory.Format.x, engine)

      // we create output primitive with any format
      outputUserPrim = initUser(output, opPrimDesc, Query.DstPd, 0)

      // ------------------------------------------------------------------------------------------
      val inputMemoryPrim = if (inputReorderPrim != 0) {
        forwardReorderPrimBuffer += inputReorderPrim
        inputReorderMemoryPrim
      } else {
        inputUserPrim
      }

      val weightMemoryPrim = if (weightReorderPrim != 0) {
        forwardReorderPrimBuffer += weightReorderPrim
        weightReorderMemoryPrim
      } else {
        weightUserPrim
      }
      val srcs = Array(inputMemoryPrim, weightMemoryPrim, biasUserPrim)
      val indexes = Array.fill(srcs.length)(0)
      val dsts = Array(outputUserPrim)

      forwardPrim = MklDnn.PrimitiveCreate2(opPrimDesc, srcs, indexes, srcs.length,
        dsts, dsts.length)
      forwardPrimBuffer += forwardPrim

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
        Memory.SetDataHandle(inputUserPrim,
          input.asInstanceOf[MklDnnTensor[T]].ptr, 0)
      }
    }

    var weightPtr = 0L
    if (weightReorderPrim != 0) {
      weightPtr = MklDnn.MemorySetDataHandle(weightUserPrim,
        weight.storage().array().asInstanceOf[Array[Float]],
        weight.storageOffset() - 1)
      Memory.SetDataHandle(weightReorderMemoryPrim, prvWeight.ptr, 0)
    } else {
//      MklDnnTensor.syncFromHeap(prvWeight, weight.storage().array(), weight.storageOffset() - 1)
//      Memory.SetDataHandle(weightUserPrim, prvWeight.ptr, 0)
      weightPtr = MklDnn.MemorySetDataHandle(weightUserPrim,
        weight.storage().array().asInstanceOf[Array[Float]],
        weight.storageOffset() - 1)
    }

    MklDnnTensor.syncFromHeap(prvBias, bias.storage().array(), bias.storageOffset() - 1)

    Memory.SetDataHandle(biasUserPrim, prvBias.ptr, 0)
    Memory.SetDataHandle(outputUserPrim, output.asInstanceOf[MklDnnTensor[T]].ptr, 0)
    if (forwardReorderPrimBuffer.nonEmpty) {
      DnnStream.Submit(stream, forwardReorderPrimBuffer.length, forwardReorderPrimBuffer.toArray)
    }

    if (inputReorderPrim != 0L) {
      if (input.getTensorType == DenseType && inputPtr != 0) {
        MklDnn.MemoryReleaseDataHandle(input.storage().array().asInstanceOf[Array[Float]], inputPtr)
      }
    }

    if (weightPtr != 0L) {
      MklDnn.MemoryReleaseDataHandle(weight.storage().array().asInstanceOf[Array[Float]],
        weightPtr)
    }

    DnnStream.Submit(stream, forwardPrimBuffer.length, forwardPrimBuffer.toArray)

    if (shouldConvert) {
      output.asInstanceOf[MklDnnTensor[T]].syncToHeap()
    }

    output
  }

  @transient var internalGradInput: MklDnnTensor[T] = _
  @transient var internalGradOutput: MklDnnTensor[T] = _
  @transient var gradOutputUserPrim = 0L
  @transient var gradOutputReorderPrim = 0L
  @transient var gradOutputReorderMemoryPrim = 0L
  @transient var gradInputUserPrim = 0L
  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (backDataPrim == 0L) {
      if (gradInput.getTensorType == MklDnnType) {
        gradInput.asInstanceOf[MklDnnTensor[T]].release()
      }
      gradInput = MklDnnTensor[T](input.size())

      backwardDataPrimBuffer = ArrayBuffer.empty[Long]
      backwardDataReorderPrimBuffer = ArrayBuffer.empty[Long]
      val diffSrcMemDesc = if (gradInput.dim() == 1) {
        MklDnn.MemoryDescInit(gradInput.dim() + 1, Array(1) ++ gradInput.size(),
          DataType.F32, Memory.Format.any)
      } else {
        MklDnn.MemoryDescInit(gradInput.dim(), gradInput.size(),
          DataType.F32, Memory.Format.any)
      }

      val weightMemDesc = if (input.dim() == 4) {
        MklDnn.MemoryDescInit(4,
//          Array(outputSize) ++ input.size().slice(1, input.dim()),
          weight.size() ++ Array(1, 1),
          DataType.F32, Memory.Format.any)
      } else {
        MklDnn.MemoryDescInit(weight.dim(), weight.size(),
          DataType.F32, Memory.Format.any)
      }

      val diffDstMemDesc = if (input.dim() == 1) {
        MklDnn.MemoryDescInit(gradOutput.dim() + 1, Array(1) ++ gradOutput.size(),
          DataType.F32, Memory.Format.any)
      } else {
        MklDnn.MemoryDescInit(gradOutput.dim(), gradOutput.size(),
          DataType.F32, Memory.Format.any)
      }

      val format = input.dim() match {
        case 1 => Memory.Format.nc
        case 2 => Memory.Format.nc
        case 4 => Memory.Format.nchw
      }

      val weightFormat = input.dim() match {
        case 1 => Memory.Format.oi
        case 2 => Memory.Format.oi
        case 4 => Memory.Format.oihw
      }

      val opDesc = MklDnn.LinearBackwardDataDescInit(diffSrcMemDesc, weightMemDesc,
        diffDstMemDesc)
      val opPrimDesc = MklDnn.PrimitiveDescCreate(opDesc, engine, forwardPrimDesc)

      gradOutputUserPrim = initUser(gradOutput, DataType.F32, Memory.Format.nc, engine)
      val g1 = initInternal(gradOutputUserPrim, opPrimDesc, Query.DiffDstPd)
      gradOutputReorderMemoryPrim = g1._1
      gradOutputReorderPrim = g1._2
      gradInputUserPrim = initUser(gradInput, opPrimDesc, Query.DiffSrcPd, 0)

      val gradOutputMemoryPrim = if (gradOutputReorderPrim != 0) {
        backwardDataReorderPrimBuffer += gradOutputReorderPrim
        gradOutputReorderMemoryPrim
      } else {
        gradOutputUserPrim
      }

      val weightMemoryPrim = if (weightReorderPrim != 0) {
        weightReorderMemoryPrim
      } else {
        weightUserPrim
      }

      val srcs = Array(gradOutputMemoryPrim, weightMemoryPrim)
      val indexes = Array.fill(srcs.length)(0)
      val dsts = Array(gradInputUserPrim)

      backDataPrim = MklDnn.PrimitiveCreate2(opPrimDesc, srcs, indexes, srcs.length,
        dsts, dsts.length)
      backwardDataPrimBuffer += backDataPrim

      if (gradOutputReorderPrim == 0 && gradOutput.getTensorType == MklDnnType) {
        internalGradOutput = gradOutput.asInstanceOf[MklDnnTensor[T]]
      } else {
        if (internalGradOutput != null) {
          internalGradOutput.release()
        }
        internalGradOutput = MklDnnTensor[T](output.size())
      }
    }

    if (gradOutput.getTensorType == DenseType) {
      internalGradOutput.set(gradOutput)
    }

    var gradOutputPtr: Long = 0
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
        Memory.SetDataHandle(gradOutputUserPrim,
          gradOutput.asInstanceOf[MklDnnTensor[T]].ptr, 0)
      }
    }

    var weightPtr = 0L
    if (weightReorderPrim != 0) {
      weightPtr = MklDnn.MemorySetDataHandle(weightUserPrim,
        weight.storage().array().asInstanceOf[Array[Float]],
        weight.storageOffset() - 1)
      Memory.SetDataHandle(weightReorderPrim, prvWeight.ptr, 0)
    } else {
      weightPtr = MklDnn.MemorySetDataHandle(weightUserPrim,
        weight.storage().array().asInstanceOf[Array[Float]],
        weight.storageOffset() - 1)
    }

    Memory.SetDataHandle(biasUserPrim, prvBias.ptr, 0)
    Memory.SetDataHandle(gradInputUserPrim, gradInput.asInstanceOf[MklDnnTensor[T]].ptr, 0)
    if (backwardDataReorderPrimBuffer.nonEmpty) {
      DnnStream.Submit(stream, backwardDataReorderPrimBuffer.length,
        backwardDataReorderPrimBuffer.toArray)
    }

    DnnStream.Submit(stream, backwardDataPrimBuffer.length, backwardDataPrimBuffer.toArray)

    if (gradOutputReorderPrim != 0) {
      if (gradOutput.getTensorType == DenseType && gradOutputPtr != 0) {
        MklDnn.MemoryReleaseDataHandle(weight.storage().array().asInstanceOf[Array[Float]],
          gradOutputPtr)
      }
    }

    if (weightPtr != 0L) {
      MklDnn.MemoryReleaseDataHandle(weight.storage().array().asInstanceOf[Array[Float]],
        weightPtr)
    }

    if (shouldConvert) {
      gradInput.asInstanceOf[MklDnnTensor[T]].syncToHeap()
    }

    gradInput
  }

  @transient var diffWeightUserPrim = 0L
  @transient var diffWeightReorderMemoryPrim = 0L
  @transient var diffWeightReorderPrim = 0L
  @transient var diffBiasUserPrim = 0L
  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    if (backWeightPrim == 0) {
      backwardWeightPrimBuffer = ArrayBuffer.empty[Long]
      backwardWeightReorderPrimBuffer = ArrayBuffer.empty[Long]
      val diffWeightMemDesc = if (input.dim() == 4) {
        MklDnn.MemoryDescInit(4,
          weight.size() ++ Array(1, 1),
          DataType.F32, Memory.Format.any)
      } else {
        MklDnn.MemoryDescInit(gradWeight.dim(), gradWeight.size(),
          DataType.F32, Memory.Format.any)
      }

      val diffBiasMemDesc = MklDnn.MemoryDescInit(gradBias.dim(), gradBias.size(),
        DataType.F32, Memory.Format.x)

      val srcMemDesc = if (input.dim() == 1) {
        MklDnn.MemoryDescInit(input.dim() + 1, Array(1) ++ input.size(),
          DataType.F32, Memory.Format.any)
      } else {
        MklDnn.MemoryDescInit(input.dim(), input.size(),
          DataType.F32, Memory.Format.any)
      }

      val diffDstMemDesc = if (input.dim() == 1) {
        MklDnn.MemoryDescInit(gradOutput.dim() + 1, Array(1) ++ gradOutput.size(),
          DataType.F32, Memory.Format.any)
      } else {
        MklDnn.MemoryDescInit(gradOutput.dim(), gradOutput.size(),
          DataType.F32, Memory.Format.any)
      }

      val opDesc = MklDnn.LinearBackwardWeightsDescInit(
        srcMemDesc, diffWeightMemDesc, diffBiasMemDesc, diffDstMemDesc)
      val opPrimDesc = MklDnn.PrimitiveDescCreate(opDesc, engine, forwardPrimDesc)

      val weightFormat = input.dim() match {
        case 1 => Memory.Format.oi
        case 2 => Memory.Format.oi
        case 4 => Memory.Format.oihw
      }

      diffWeightUserPrim = if (input.dim() == 4) {
        init4(4, Array(outputSize) ++ input.size().slice(1, input.dim())
          , DataType.F32, weightFormat, engine)
      } else {
        initUser(diffWeight, DataType.F32, weightFormat, engine)
      }
      val d1 = initInternal(diffWeightUserPrim, opPrimDesc, Query.DiffWeightsPd,
        userToPrim = false)
      diffWeightReorderMemoryPrim = d1._1
      diffWeightReorderPrim = d1._2
      diffBiasUserPrim = initUser(diffBias, DataType.F32, Memory.Format.x, engine)

      val diffWeightMemoryPrim = if (diffWeightReorderPrim != 0) {
        diffWeightReorderMemoryPrim
      } else {
        diffWeightUserPrim
      }

      val inputMemoryPrim = if (inputReorderPrim != 0) {
        inputReorderMemoryPrim
      } else {
        inputUserPrim
      }

      val gradOutputMemoryPrim = if (gradOutputReorderPrim != 0) {
        gradOutputReorderMemoryPrim
      } else {
        gradOutputUserPrim
      }

      val srcs = Array(inputMemoryPrim, gradOutputMemoryPrim)
      val indexes = Array.fill(srcs.length)(0)
      val dsts = Array(diffWeightMemoryPrim, diffBiasUserPrim)

      backWeightPrim = MklDnn.PrimitiveCreate2(opPrimDesc,
        srcs, indexes, srcs.length, dsts, dsts.length)

      backwardWeightPrimBuffer += backWeightPrim

      if (diffWeightReorderPrim != 0) {
        backwardWeightReorderPrimBuffer += diffWeightReorderPrim
      }
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
        Memory.SetDataHandle(inputUserPrim, internalInput.ptr, 0)
      } else if (input.getTensorType == MklDnnType) {
        Memory.SetDataHandle(inputUserPrim,
          input.asInstanceOf[MklDnnTensor[T]].ptr, 0)
      }
    }

    var gradOutputPtr: Long = 0
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
        Memory.SetDataHandle(gradOutputUserPrim, internalGradOutput.ptr, 0)
      } else if (gradOutput.getTensorType == MklDnnType) {
        Memory.SetDataHandle(gradOutputUserPrim,
          gradOutput.asInstanceOf[MklDnnTensor[T]].ptr, 0)
      }
    }

    val biasPtr = MklDnn.MemorySetDataHandle(diffBiasUserPrim,
      gradBias.storage().array().asInstanceOf[Array[Float]],
      gradBias.storageOffset() - 1)
    // Memory.SetDataHandle(diffBiasUserPrim, diffBias.ptr, 0)

    var weightPtr = 0L
    weightPtr = MklDnn.MemorySetDataHandle(diffWeightUserPrim,
      gradWeight.storage().array().asInstanceOf[Array[Float]],
      gradWeight.storageOffset() - 1)

    DnnStream.Submit(stream, backwardWeightPrimBuffer.length, backwardWeightPrimBuffer.toArray)

    if (backwardWeightReorderPrimBuffer.nonEmpty) {
      DnnStream.Submit(stream, backwardWeightReorderPrimBuffer.length,
        backwardWeightReorderPrimBuffer.toArray)
    }

    if (gradOutputReorderPrim != 0) {
      if (gradOutput.getTensorType == DenseType && gradOutputPtr != 0) {
        MklDnn.MemoryReleaseDataHandle(gradOutput.storage().array().asInstanceOf[Array[Float]],
          gradOutputPtr)
      }
    }
    if (inputReorderPrim != 0L) {
      if (input.getTensorType == DenseType && inputPtr != 0) {
        MklDnn.MemoryReleaseDataHandle(input.storage().array().asInstanceOf[Array[Float]], inputPtr)
      }
    }

    if (weightPtr != 0) {
      MklDnn.MemoryReleaseDataHandle(gradWeight.storage().array().asInstanceOf[Array[Float]],
        weightPtr)
    }

    MklDnn.MemoryReleaseDataHandle(gradBias.storage().array().asInstanceOf[Array[Float]],
      biasPtr)
  }

  override def zeroGradParameters(): Unit = {
    gradWeight.zero()
    if (withBias) {
      gradBias.zero()
    }

    Memory.Zero(diffWeight.ptr, gradWeight.nElement(), 4)
    Memory.Zero(diffBias.ptr, gradBias.nElement(), 4)
  }

  override def clearState() : this.type = {
    super.clearState()
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
