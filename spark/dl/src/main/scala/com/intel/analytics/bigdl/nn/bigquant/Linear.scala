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
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.serializer.{DataConverter, ModuleData, ModuleSerializer}
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

  private def initWeightAndBias(weightFP32: Tensor[T], biasFP32: Tensor[T]): this.type = {
    if (biasFP32 != null) {
      bias.copy(biasFP32)
    } else {
      bias.fill(ev.fromType(0)) // TODO bias may be null, at that time, we should not initialize it
    }

    val weightFP32Tmp = weightFP32.view(Array(outputSize, inputSize))

    val bufferOffset = 0
    val buffer = new Array[Byte](weightFP32.nElement())
    val weightFP32Tensor = weightFP32.asInstanceOf[Tensor[Float]]

    val params = LinearWeightParams(outputSize, inputSize)
    weight = QuantTensor[T](weightFP32Tmp, params, LinearWeight)

    this
  }

  override def init(): this.type = {
    weight.init(LinearWeightParams(outputSize, inputSize), LinearWeight)

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

    if (weight.getStorage != null && weight.getNativeStorage == 0L) {
      init()
    }
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 1 || input.dim() == 2,
      "bigquant.Linear: " + ErrorInfo.constrainInputAsVectorOrBatch)

    val batchSize = if (input.dim() == 1) {
      output.resize(Array(outputSize)) // TODO
      1
    } else {
      output.resize(Array(input.size(1), outputSize))
      require(inputSize == input.size(2), s"dimension error")
      input.size(1)
    }

    data.init(LinearDataParams(batchSize, inputSize), LinearData)

    ev.getType() match {
      case FloatType => // TODO
        val src = input.storage().array().asInstanceOf[Array[Float]]
        val offset = input.storageOffset() - 1

        BigQuant.FCDataInit(data.getNativeStorage, src, offset, batchSize, inputSize,
          QuantParams.THRESHOLD, BigQuant.NCHW)

        val outputArray = output.storage().array().asInstanceOf[Array[Float]]
        val outputOffset = output.storageOffset() - 1
        val weightSumArray = weight.sumOfRow.asInstanceOf[Array[Float]]
        val weightSumOffset = 0
        val biasArray = bias.storage().array().asInstanceOf[Array[Float]]
        val biasOffset = bias.storageOffset() - 1

        BigQuant.InternalMixPrecisionConvolutionGEMM(
          BigQuant.NCHW, weight.getNativeStorage, data.getNativeStorage, outputArray,
          outputOffset, weightSumArray, weightSumOffset, biasArray, biasOffset,
          batchSize, outputSize, 1, 1,
          QuantParams.FAULT_TOLERANCE)

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
    withBias: Boolean = true,
    initWeight: Tensor[T] = null,
    initBias: Tensor[T] = null
  )(implicit ev: TensorNumeric[T]) : Linear[T] = {
    val linear = new Linear[T](inputSize, outputSize, withBias)
    linear.initWeightAndBias(initWeight, initBias)
  }

  override def serializeWeight[T: ClassTag](module: ModuleData[T],
    modelBuilder: BigDLModule.Builder)(implicit ev: TensorNumeric[T]): Unit = {
    val linear = module.module.asInstanceOf[Linear[T]]
    val weightBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(weightBuilder, linear.weight, ModuleSerializer.tensorType)
    modelBuilder.putAttr("weight", weightBuilder.build)
  }

  override def loadWeight[T: ClassTag](model: BigDLModule,
    module: ModuleData[T])(implicit ev: TensorNumeric[T]): Unit = {
    val linear = module.module.asInstanceOf[Linear[T]]
    val attrMap = model.getAttrMap

    linear.weight = DataConverter.getAttributeValue(attrMap.get("weight"))
            .asInstanceOf[QuantTensor[T]]
    linear.init()
  }
}
