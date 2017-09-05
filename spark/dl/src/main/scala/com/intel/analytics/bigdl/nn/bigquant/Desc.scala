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
import com.intel.analytics.bigdl.tensor.{FloatType, QuantTensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import scala.reflect.ClassTag

sealed trait DescType
object ConvData extends DescType with Serializable
object ConvWeight extends DescType with Serializable
object LinearData extends DescType with Serializable
object LinearWeight extends DescType with Serializable

trait DescParams
case class ConvDataParams(nInputPlane: Int, kernelH: Int, kernelW: Int,
  strideH: Int, strideW: Int, padH: Int, padW: Int, dilationHeight: Int, dilationWidth: Int,
  batchSize: Int, inputHeight: Int, inputWidth: Int) extends DescParams with Serializable
case class ConvWeightParams(nOutputPlane: Int, nInputPlane: Int, kernelH: Int,
  kernelW: Int) extends DescParams with Serializable
case class LinearDataParams(batchSize: Int, inputSize: Int) extends DescParams with Serializable
case class LinearWeightParams(outputSize: Int, inputSize: Int) extends DescParams with Serializable

object Desc {
  def get[T: ClassTag](params: DescParams, descType: DescType, bytes: Array[Byte], offset: Int,
    max: Array[T], min: Array[T])(implicit ev: TensorNumeric[T]): Long = {
    val desc = descType match {
      case ConvData =>
        val p = params.asInstanceOf[ConvDataParams]
        BigQuant.ConvDataDescInit(p.nInputPlane,
          p.kernelH, p.kernelW, p.strideH, p.strideW, p.padH, p.padW,
          p.dilationHeight, p.dilationWidth, p.batchSize, p.inputHeight, p.inputWidth)

      case ConvWeight =>
        require(bytes != null, s"unknwon storage, you should init first")
        val p = params.asInstanceOf[ConvWeightParams]
        val desc = BigQuant.ConvKernelDescInit(p.nOutputPlane, p.nInputPlane, p.kernelH, p.kernelW)
        convWeigth(p, desc, bytes, offset, max, min)
        desc

      case LinearData =>
        val p = params.asInstanceOf[LinearDataParams]
        BigQuant.FCDataDescInit(p.batchSize, p.inputSize)

      case LinearWeight =>
        require(bytes != null, s"unknwon storage, you should init first")
        val p = params.asInstanceOf[LinearWeightParams]
        val desc = BigQuant.FCKernelDescInit(p.outputSize, p.inputSize)
        linearWeight(p, desc, bytes, offset, max, min)
        desc
    }

    desc
  }

  def get[T: ClassTag](params: DescParams, descType: DescType, tensor: QuantTensor[T])(implicit
    ev: TensorNumeric[T]): Long = {
    get(params, descType, tensor.getStorage, 0, tensor.maxOfRow, tensor.minOfRow)
  }

  private def linearWeight[T: ClassTag](p: LinearWeightParams, desc: Long, bytes: Array[Byte],
    offset: Int, max: Array[T], min: Array[T])(implicit ev: TensorNumeric[T]): Long = {
    ev.getType() match {
      case FloatType =>
        val minArray = min.asInstanceOf[Array[Float]]
        val maxArray = max.asInstanceOf[Array[Float]]

        BigQuant.FCKernelLoadFromModel(desc, bytes, minArray, maxArray,
          p.outputSize, p.inputSize, QuantParams.WEIGHT_THRESHOLD, BigQuant.NCHW)
      case _ =>
        throw new UnsupportedOperationException(s"Only support Float for quantized model")
    }
    desc
  }

  private def convWeigth[T: ClassTag](p: ConvWeightParams, desc: Long, bytes: Array[Byte],
    offset: Int, max: Array[T], min: Array[T])(implicit ev: TensorNumeric[T]): Long = {
    ev.getType() match {
      case FloatType =>
        val minArray = min.asInstanceOf[Array[Float]]
        val maxArray = max.asInstanceOf[Array[Float]]
        BigQuant.ConvKernelLoadFromModel(desc,
          bytes, offset,
          minArray, maxArray, p.nOutputPlane, p.nInputPlane,
          p.kernelH, p.kernelW, QuantParams.WEIGHT_THRESHOLD, BigQuant.NCHW)
      case _ =>
        throw new UnsupportedOperationException(s"Only support Float for quantized model")
    }
    desc
  }
}

object QuantParams {
  val FAULT_TOLERANCE = 0.5f
  val WEIGHT_THRESHOLD = 64.0f
  val THRESHOLD = 127.0f
}

