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
import com.intel.analytics.bigdl.nn.abstractnn.{Initializable, TensorModule}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.serializer.{DataConverter, ModuleData}
import com.intel.analytics.bigdl.utils.{T, Table}
import java.io.{IOException, ObjectInputStream, ObjectOutputStream}
import scala.reflect.ClassTag
import scala.reflect.runtime.universe
import serialization.Bigdl.{AttrValue, BigDLModule}

@SerialVersionUID(- 8008252944905538960L)
class SpatialConvolution[T: ClassTag](
  val nInputPlane: Int, // The number of expected input planes in the image given into forward()
  val nOutputPlane: Int, // The number of output planes the convolution layer will produce.
  val kernelW: Int, // The kernel width of the convolution
  val kernelH: Int, // The kernel height of the convolution
  val strideW: Int = 1, // The step of the convolution in the width dimension.
  val strideH: Int = 1, // The step of the convolution in the height dimension
  val padW: Int = 0, // The additional zeros added per width to the input planes.
  val padH: Int = 0, // The additional zeros added per height to the input planes.
  val nGroup: Int = 1 // Kernel group number
)(implicit ev: TensorNumeric[T]) extends QuantModule[T](nOutputPlane) with Initializable {

  require(nInputPlane % nGroup == 0, "Number of input channels should be multiples of group.")
  require(nOutputPlane % nGroup == 0, "Number of output channels should be multiples of group.")

  @transient var weight: Array[Tensor[T]] = null
  var data: QuantTensor[T] = QuantTensor[T]()

  val bias: Tensor[T] = Tensor[T](nOutputPlane)

  val DILATION_HEIGHT = 1
  val DILATION_WIDTH = 1

  @transient var _init = false

  private def outputSize(input: Int, pad: Int, kernel: Int, stride: Int): Int = {
    (input + 2 * pad - kernel) / stride + 1
  }

  def initWeightAndBias(weightFP32: Tensor[T], biasFP32: Tensor[T]): this.type = {
    if (biasFP32 != null) {
      bias.copy(biasFP32)
    } else {
      bias.fill(ev.fromType(0)) // TODO bias may be null, at that time, we should not initialize it
    }

    weight = new Array[Tensor[T]](nGroup)
    val params = ConvWeightParams(nOutputPlane / nGroup, nInputPlane / nGroup, kernelH, kernelW)
    for (i <- 1 to nGroup) {
      val groupWeight = weightFP32.select(1, i)
      ev.getType() match {
        case FloatType =>
          weight(i - 1) = QuantTensor[T](groupWeight, params, ConvWeight)
        case _ => throw new UnsupportedOperationException(s"Only support Float for quantized model")
      }
    }

    init()

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

    weight = in.readObject().asInstanceOf[Array[Tensor[T]]]
    if (weight(0).asInstanceOf[QuantTensor[T]].getStorage != null &&
      weight(0).asInstanceOf[QuantTensor[T]].getNativeStorage == 0L) {
      init()
    }
  }

  private def init(): this.type = {
    val params = ConvWeightParams(nOutputPlane / nGroup, nInputPlane / nGroup, kernelH, kernelW)
    for (i <- 1 to nGroup) {
      weight(i - 1).asInstanceOf[QuantTensor[T]].init(params, ConvWeight)
    }

    _init = true

    this
  }

  private def checkAndInit(): Unit = {
    if (!_init) {
      init()
    }
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 3 || input.dim() == 4,
      "bigquant.SpatialConvolution: " + ErrorInfo.constrainInputAs3DOrBatch)
    require(input.isContiguous())

    checkAndInit()

    // compute attributes of input and output
    val (batchSize, inputHeight, inputWidth) = if (input.dim() == 3) {
      (1, input.size(2), input.size(3))
    } else {
      (input.size(1), input.size(3), input.size(4))
    }

    val outputHeight = outputSize(inputHeight, padH, kernelH, strideH)
    val outputWidth = outputSize(inputWidth, padW, kernelW, strideW)

    output.resize(Array(batchSize, nOutputPlane, outputHeight, outputWidth))

    data.init(ConvDataParams(nInputPlane / nGroup, kernelH, kernelW,
        strideH, strideW, padH, padW, DILATION_HEIGHT, DILATION_WIDTH, 1,
        inputHeight, inputWidth), ConvData)

    ev.getType() match {
      case FloatType =>
        var batch = 0
        while (batch < batchSize) {
            im2ColAndGemmFloat(batch)
          batch += 1
        }
      case _ => throw new UnsupportedOperationException(s"Only support Float for quantized model")
    }

    @inline def im2ColAndGemmFloat(batch: Int): Unit = {
      val batchOutput = output.select(1, batch + 1)
      val batchInput = input.select(1, batch + 1)

      var group = 0
      while (group < nGroup) {
        val groupBatchOutput = batchOutput.narrow(1, group * nOutputPlane / nGroup + 1,
          nOutputPlane / nGroup)
        val groupBatchInput = batchInput.narrow(1, group * nInputPlane / nGroup + 1,
          nInputPlane / nGroup)
        val groupWeight = weight(group).asInstanceOf[QuantTensor[T]]
        val offset = 0

        groupIm2ColGemm(groupBatchInput, groupBatchOutput, groupWeight, offset)

        group += 1
      }
    }

    @inline def groupIm2ColGemm(input: Tensor[T], output: Tensor[T],
      weight: QuantTensor[T], offset: Int): Unit = {
      val inputArray = input.storage().array().asInstanceOf[Array[Float]]
      val inputOffset = input.storageOffset() - 1

      val outputArray = output.storage().array().asInstanceOf[Array[Float]]
      val outputOffset = output.storageOffset() - 1

      val biasArray = bias.storage().array().asInstanceOf[Array[Float]]
      val biasOffset = bias.storageOffset() - 1 + offset

      val weightSumArray = weight.sumOfRow.asInstanceOf[Array[Float]]
      val weightSumOffset = offset

      BigQuant.ConvDataInit(
        data.getNativeStorage, inputArray, inputOffset,
        nInputPlane / nGroup, kernelH, kernelW, strideH, strideW, padH, padW,
        DILATION_HEIGHT, DILATION_WIDTH, 1, inputHeight, inputWidth, QuantParams.THRESHOLD,
        BigQuant.NCHW)

      BigQuant.InternalMixPrecisionConvolutionGEMM(
        BigQuant.NCHW, weight.getNativeStorage, data.getNativeStorage,
        outputArray, outputOffset, weightSumArray, weightSumOffset,
        biasArray, biasOffset, 1, nOutputPlane / nGroup, outputHeight, outputWidth,
        QuantParams.FAULT_TOLERANCE)
    }

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Doesn't updateGradInput for quantized model")
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (weight :+ bias, Array(empty, empty))
  }

  override def getParametersTable(): Table = {
    T(getName() -> T("weight" -> weight, "bias" -> bias))
  }

  override def equals(obj: Any): Boolean = {
    if (!super.equals(obj)) {
      return false
    }
    if (!obj.isInstanceOf[SpatialConvolution[T]]) {
      return false
    }
    val other = obj.asInstanceOf[SpatialConvolution[T]]
    if (this.eq(other)) {
      return true
    }

    nInputPlane == other.nInputPlane &&
      nOutputPlane == other.nOutputPlane &&
      kernelW == other.kernelW &&
      kernelH == other.kernelH &&
      strideW == other.strideW &&
      strideH == other.strideH &&
      padW == other.padW &&
      padH == other.padH &&
      nGroup == other.nGroup &&
      weight == other.weight &&
      bias == other.bias
  }

  override def hashCode(): Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + nInputPlane.hashCode()
    hash = hash * seed + nOutputPlane.hashCode()
    hash = hash * seed + kernelW.hashCode()
    hash = hash * seed + kernelH.hashCode()
    hash = hash * seed + strideW.hashCode()
    hash = hash * seed + strideH.hashCode()
    hash = hash * seed + padW.hashCode()
    hash = hash * seed + padH.hashCode()
    hash = hash * seed + bias.hashCode()
    hash = hash * seed + weight.hashCode()

    hash
  }

  override def clearState() : this.type = {
    super.clearState()
    this
  }

  override def toString(): String = {
    s"bigquant.SpatialConvolution($nInputPlane -> $nOutputPlane, $kernelW x" +
      s" $kernelH, $strideW, $strideH, $padW, $padH, $nGroup)"
  }

  def release(): Unit = {
    weight.foreach(_.asInstanceOf[QuantTensor[T]].release())
    data.release()
  }
}

object SpatialConvolution extends QuantSerializer {
  def apply[@specialized(Float) T: ClassTag](
    nInputPlane: Int,
    nOutputPlane: Int,
    kernelW: Int,
    kernelH: Int,
    strideW: Int = 1,
    strideH: Int = 1,
    padW: Int = 0,
    padH: Int = 0,
    nGroup: Int = 1
  )(implicit ev: TensorNumeric[T]): SpatialConvolution[T] = {
    new SpatialConvolution[T](nInputPlane, nOutputPlane, kernelW, kernelH,
      strideW, strideH, padW, padH, nGroup)
  }

  override def serializeWeight[T: ClassTag](module: ModuleData[T],
    modelBuilder: BigDLModule.Builder)(implicit ev: TensorNumeric[T]): Unit = {
    val conv = module.module.asInstanceOf[SpatialConvolution[T]]
    val weightBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(weightBuilder, conv.weight)
    modelBuilder.putAttr("weights", weightBuilder.build)
  }

  override def loadWeight[T: ClassTag](model: BigDLModule,
    module: ModuleData[T])(implicit ev: TensorNumeric[T]): Unit = {
    val conv = module.module.asInstanceOf[SpatialConvolution[T]]
    val attrMap = model.getAttrMap
    conv.weight = DataConverter.getAttributeValue(attrMap.get("weights"))
      .asInstanceOf[Array[Tensor[T]]]
  }
}

