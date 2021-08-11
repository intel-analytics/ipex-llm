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

package com.intel.analytics.bigdl.nn.quantized

import com.intel.analytics.bigdl.bigquant.BigQuant
import com.intel.analytics.bigdl.tensor.{FloatType, QuantizedTensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import scala.reflect.ClassTag

sealed trait DescType extends Serializable
object ConvData extends DescType
object ConvWeight extends DescType
object LinearData extends DescType
object LinearWeight extends DescType

trait DescParams extends Serializable with Product {
  def copy(): DescParams
  def array: Array[Any] = this.productIterator.toArray
  def getType: DescType
}

case class ConvDataParams(nInputPlane: Int, kernelH: Int, kernelW: Int,
  strideH: Int, strideW: Int, padH: Int, padW: Int, dilationHeight: Int, dilationWidth: Int,
  batchSize: Int, inputHeight: Int, inputWidth: Int) extends DescParams {
  override def copy(): DescParams = {
    val p = this
    new ConvDataParams(p.nInputPlane, p.kernelH, p.kernelW, p.strideH, p.strideW,
      p.padH, p.padW, p.dilationHeight, p.dilationWidth, p.batchSize, p.inputHeight,
      p.inputWidth)
  }

  override def getType: DescType = ConvData
}

object ConvDataParams {
  def apply(params: Array[Int]): ConvDataParams = new ConvDataParams(
    params(0), // nInputPlane
    params(1), // kernelH
    params(2), // kernelW
    params(3), // strideH
    params(4), // strideW
    params(5), // padH
    params(6), // padW
    params(7), // dilationHeight
    params(8), // dilationWidth
    params(9), // batchSize
    params(10), // inputHeigh
    params(11)) // inputWidth
}

case class ConvWeightParams(nOutputPlane: Int, nInputPlane: Int, kernelH: Int,
  kernelW: Int, dataFormat: Int) extends DescParams {

  override def copy(): DescParams = {
    val p = this
    new ConvWeightParams(p.nOutputPlane, p.nInputPlane, p.kernelH, p.kernelW, dataFormat)
  }

  override def getType: DescType = ConvWeight
}

object ConvWeightParams {
  def apply(params: Array[Int]): ConvWeightParams = ConvWeightParams(
    params(0), // nOutputPlane
    params(1), // nInputPlane
    params(2), // kernelH
    params(3), // kernelW
    params(4)) // data format
}

case class LinearDataParams(batchSize: Int, inputSize: Int) extends DescParams {

  override def copy(): DescParams = {
    val p = this
    new LinearDataParams(p.batchSize, p.inputSize)
  }

  override def getType: DescType = LinearData
}

object LinearDataParams {
  def apply(params: Array[Int]): LinearDataParams = LinearDataParams(
    params(0), // batchSize
    params(1)) // inputSize

}

case class LinearWeightParams(outputSize: Int, inputSize: Int) extends DescParams {

  override def copy(): DescParams = {
    val p = this
    new LinearWeightParams(p.outputSize, p.inputSize)
  }

  override def getType: DescType = LinearWeight
}

object LinearWeightParams {
  def apply(params: Array[Int]): LinearWeightParams = LinearWeightParams(
    params(0), // outputSize
    params(1)) // inputSize

}

object Desc {
  def get[T: ClassTag](params: DescParams, bytes: Array[Byte], offset: Int,
    max: Array[T], min: Array[T]): Long = {
    val desc = params.getType match {
      case ConvData =>
        val p = params.asInstanceOf[ConvDataParams]
        BigQuant.ConvDataDescInit(p.nInputPlane,
          p.kernelH, p.kernelW, p.strideH, p.strideW, p.padH, p.padW,
          p.dilationHeight, p.dilationWidth, p.batchSize, p.inputHeight, p.inputWidth)

      case ConvWeight =>
        val p = params.asInstanceOf[ConvWeightParams]
        val desc = BigQuant.ConvKernelDescInit(p.nOutputPlane, p.nInputPlane, p.kernelH, p.kernelW)
        if (bytes != null) {
          convWeigth(p, desc, bytes, offset, max, min)
        }
        desc

      case LinearData =>
        val p = params.asInstanceOf[LinearDataParams]
        BigQuant.FCDataDescInit(p.batchSize, p.inputSize)

      case LinearWeight =>
        val p = params.asInstanceOf[LinearWeightParams]
        val desc = BigQuant.FCKernelDescInit(p.outputSize, p.inputSize)
        if (bytes != null) {
          linearWeight(p, desc, bytes, offset, max, min)
        }
        desc
    }

    // add every native memory allocation.
    StorageManager.add(desc, params.getType)

    desc
  }

  private def linearWeight[T: ClassTag](p: LinearWeightParams, desc: Long, bytes: Array[Byte],
    offset: Int, max: Array[T], min: Array[T]): Long = {
    val minArray = min.asInstanceOf[Array[Float]]
    val maxArray = max.asInstanceOf[Array[Float]]

    BigQuant.FCKernelLoadFromModel(desc, bytes, minArray, maxArray,
      p.outputSize, p.inputSize, QuantParams.WEIGHT_THRESHOLD, BigQuant.NCHW)
    desc
  }

  private def convWeigth[T: ClassTag](p: ConvWeightParams, desc: Long, bytes: Array[Byte],
    offset: Int, max: Array[T], min: Array[T]): Long = {
    val minArray = min.asInstanceOf[Array[Float]]
    val maxArray = max.asInstanceOf[Array[Float]]
    BigQuant.ConvKernelLoadFromModel(desc,
      bytes, offset,
      minArray, maxArray, p.nOutputPlane, p.nInputPlane,
      p.kernelH, p.kernelW, QuantParams.WEIGHT_THRESHOLD, p.dataFormat)
    desc
  }
}

object QuantParams {
  val FAULT_TOLERANCE = 0.5f
  val WEIGHT_THRESHOLD = 64.0f
  val THRESHOLD = 127.0f
}

private[bigdl] case class StorageInfo(descType: DescType, isFreed: Boolean)

private[bigdl] object StorageManager {
  import java.util.concurrent.ConcurrentHashMap
  private val nativeStorages: ConcurrentHashMap[Long, StorageInfo] = new ConcurrentHashMap()

  def isFreed(nativeStorage: Long): Boolean = {
    nativeStorages.get(nativeStorage).isFreed
  }

  // atomically set the value
  def checkAndSet(nativeStorage: Long): Boolean = {
    val descType = nativeStorages.get(nativeStorage).descType
    nativeStorages.replace(nativeStorage, StorageInfo(descType, false), StorageInfo(descType, true))
  }

  def get(): Map[Long, StorageInfo] = {
    import scala.collection.JavaConverters._
    nativeStorages.asScala.toMap
  }

  def add(key: Long, descType: DescType): Unit = {
    nativeStorages.put(key, StorageInfo(descType, false))
  }
}
