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

package com.intel.analytics.bigdl.nn.bigquant

import com.intel.analytics.bigdl.bigquant.BigQuant
import com.intel.analytics.bigdl.nn.ErrorInfo
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{FloatType, QuantTensor, Tensor}
import com.intel.analytics.bigdl.utils.serializer.{DataConverter, ModuleData}
import com.intel.analytics.bigdl.utils.{T, Table}
import java.io.{IOException, ObjectInputStream, ObjectOutputStream}
import scala.reflect.ClassTag
import scala.reflect.runtime.universe
import serialization.Bigdl.{AttrValue, BigDLModule}

class Linear[T: ClassTag](
  val inputSize: Int,
  val outputSize: Int,
  val withBias: Boolean = true
)(implicit ev: TensorNumeric[T]) extends QuantModule[T](outputSize) {

  val data: QuantTensor[T] = QuantTensor[T]()
  @transient var weight: QuantTensor[T] = _
  val bias: Tensor[T] = Tensor[T](outputSize)

  val FAULT_TOLERANCE = 0.5f
  val WEIGHT_THRESHOLD = 64.0f
  val THRESHOLD = 127.0f

  @transient var _init = false

  private def setWeightSum(weight: Tensor[T], weightSum: Array[T]): Unit = {
    for (i <- 1 to outputSize) {
      val singleRow = weight.select(1, i)
      weightSum(i - 1) = singleRow.sum()
    }
  }

  def initWeightAndBias(weightFP32: Tensor[T], biasFP32: Tensor[T]): this.type = {
    if (biasFP32 != null) {
      bias.copy(biasFP32)
    } else {
      bias.fill(ev.fromType(0)) // TODO bias may be null, at that time, we should not initialize it
    }

    val weightFP32Tmp = weightFP32.view(Array(outputSize, inputSize))
    setWeightSum(weightFP32Tmp, weightSum)

    for (i <- 1 to outputSize) {
      val singleRow = weightFP32Tmp.select(1, i)
      min(i - 1) = singleRow.min()
      max(i - 1) = singleRow.max()
    }

    val bufferOffset = 0
    val buffer = new Array[Byte](weightFP32.nElement())
    val weightFP32Tensor = weightFP32.asInstanceOf[Tensor[Float]]
    Quant.quantize(weightFP32Tensor, buffer, bufferOffset)

    weight = QuantTensor[T](outputSize, inputSize)
    weight.setStorage(buffer)

    init()

    this
  }

  def init(): this.type = {
    val byteArrayOfWeight = weight.getStorage

    weight.setStorageInJni(
      BigQuant.FCKernelDescInit(outputSize, inputSize))

    ev.getType() match {
      case FloatType =>
        val minArray = min.asInstanceOf[Array[Float]]
        val maxArray = max.asInstanceOf[Array[Float]]

        BigQuant.FCKernelLoadFromModel(weight.getStorageInJni, byteArrayOfWeight,
          minArray, maxArray, outputSize, inputSize, WEIGHT_THRESHOLD, BigQuant.NCHW)
      case _ => throw new UnsupportedOperationException(s"Only support Float for quantized model")
    }

    _init = true

    this
  }

  @throws(classOf[IOException])
  private def writeObject(out: ObjectOutputStream): Unit = {
    out.defaultWriteObject()

    out.writeObject(weight)
  }

  @throws(classOf[IOException])
  private def readObject(in: ObjectInputStream): Unit = {
    in.defaultReadObject()

    weight = in.readObject().asInstanceOf[QuantTensor[T]]

    if (weight.getStorage != null && weight.getStorageInJni == 0L) {
      init()
    }
  }

  def checkAndInit(): Unit = {
    if (!_init && weight.getStorageInJni == 0L) {
      init()
    }
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 1 || input.dim() == 2,
      "bigquant.Linear: " + ErrorInfo.constrainInputAsVectorOrBatch)

    checkAndInit()

    val batchSize = if (input.dim() == 1) {
      output.resize(Array(outputSize)) // TODO
      1
    } else {
      output.resize(Array(input.size(1), outputSize))
      require(inputSize == input.size(2), s"dimension error")
      input.size(1)
    }

    if (!data.isInitialized) {
      data.setStorageInJni(BigQuant.FCDataDescInit(batchSize, inputSize))
    }

    ev.getType() match {
      case FloatType => // TODO
        val src = input.storage().array().asInstanceOf[Array[Float]]
        val offset = input.storageOffset() - 1

        BigQuant.FCDataInit(data.getStorageInJni, src, offset, batchSize, inputSize,
          THRESHOLD, BigQuant.NCHW)

        val outputArray = output.storage().array().asInstanceOf[Array[Float]]
        val outputOffset = output.storageOffset() - 1
        val weightSumArray = weightSum.asInstanceOf[Array[Float]]
        val weightSumOffset = 0
        val biasArray = bias.storage().array().asInstanceOf[Array[Float]]
        val biasOffset = bias.storageOffset() - 1

        BigQuant.InternalMixPrecisionConvolutionGEMM(
          BigQuant.NCHW, weight.getStorageInJni, data.getStorageInJni, outputArray,
          outputOffset, weightSumArray, weightSumOffset, biasArray, biasOffset,
          batchSize, outputSize, 1, 1,
          FAULT_TOLERANCE)

      case _ => throw new UnsupportedOperationException(s"Only support Float for quantized model")
    }

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Doesn't updateGradInput for quantized model")
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(weight, bias), Array(empty, empty))
  }

  override def getParametersTable(): Table = {
    T(getName() -> T("weight" -> weight, "bias" -> bias))
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

    weight == other.weight &&
      bias == other.bias
  }

  override def hashCode(): Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + weight.hashCode()
    hash = hash * seed + bias.hashCode()

    hash
  }

  override def clearState() : this.type = {
    super.clearState()
    this
  }

  override def toString(): String = {
    s"bigquant.${getPrintName()}($inputSize -> $outputSize)"
  }

  def release(): Unit = {
    weight.release()
    data.release()
  }
}


object Linear extends QuantSerializer {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputSize: Int,
    outputSize: Int,
    withBias: Boolean = true
  )(implicit ev: TensorNumeric[T]) : Linear[T] = {
    new Linear[T](inputSize, outputSize, withBias)
  }

  override def serializeWeight[T: ClassTag](module: ModuleData[T],
    modelBuilder: BigDLModule.Builder)(implicit ev: TensorNumeric[T]): Unit = {
    val linear = module.module.asInstanceOf[Linear[T]]
    val weight = new Array[Byte](linear.outputSize * linear.inputSize)
    System.arraycopy(linear.weight.getStorage, 0, weight, 0, weight.length)

    val weightBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(weightBuilder, weight, universe.typeOf[Array[Byte]])
    modelBuilder.putAttr("weight", weightBuilder.build)
  }

  override def loadWeight[T: ClassTag](model: BigDLModule,
    module: ModuleData[T])(implicit ev: TensorNumeric[T]): Unit = {
    val linear = module.module.asInstanceOf[Linear[T]]
    val attrMap = model.getAttrMap
    val byteArray = DataConverter.getAttributeValue(attrMap.get("weight"))
            .asInstanceOf[Array[Byte]]

    linear.weight = new QuantTensor[T](linear.outputSize, linear.inputSize)
    val storage = new Array[Byte](linear.weight.size().product)
    System.arraycopy(byteArray, 0, storage, 0, storage.length)
    linear.weight.setStorage(storage)
  }
}
