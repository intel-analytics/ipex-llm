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

import com.intel.analytics.bigdl.tensor.QuantTensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer.{DataConverter, ModuleData}
import scala.reflect.ClassTag
import scala.reflect.runtime.universe
import serialization.Bigdl.{AttrValue, BigDLModule}

@SerialVersionUID(- 8572055756810843156L)
class SpatialDilatedConvolution[T: ClassTag](
  nInputPlane: Int, // The number of expected input planes in the image given into forward()
  nOutputPlane: Int, // The number of output planes the convolution layer will produce.
  kernelW: Int, // The kernel width of the convolution
  kernelH: Int, // The kernel height of the convolution
  strideW: Int = 1, // The step of the convolution in the width dimension.
  strideH: Int = 1, // The step of the convolution in the height dimension
  padW: Int = 0, // The additional zeros added per width to the input planes.
  padH: Int = 0, // The additional zeros added per height to the input planes.
  val dilationW: Int = 1,
  val dilationH: Int = 1,
  nGroup: Int = 1 // Kernel group number
)(implicit ev: TensorNumeric[T]) extends SpatialConvolution[T](
  nInputPlane,
  nOutputPlane,
  kernelW,
  kernelH,
  strideW,
  strideH,
  padW,
  padH,
  nGroup
) {
  override val DILATION_WIDTH: Int = dilationW
  override val DILATION_HEIGHT: Int = dilationH

  override def toString(): String = {
    s"fixpoint.SpatialDilatedConvolution($nInputPlane -> $nOutputPlane, $kernelW x" +
      s" $kernelH, $strideW, $strideH, $padW, $padH, $dilationW, $dilationH)"
  }
}

object SpatialDilatedConvolution extends QuantSerializer {
  def apply[@specialized(Float) T: ClassTag](
    nInputPlane: Int,
    nOutputPlane: Int,
    kW: Int,
    kH: Int,
    dW: Int = 1,
    dH: Int = 1,
    padW: Int = 0,
    padH: Int = 0,
    dilationW: Int = 1,
    dilationH: Int = 1
  )(implicit ev: TensorNumeric[T]) : SpatialDilatedConvolution[T] = {
    new SpatialDilatedConvolution[T](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH, dilationW, dilationH)
  }

  override def serializeWeight[T: ClassTag](module: ModuleData[T],
    modelBuilder: BigDLModule.Builder)(implicit ev: TensorNumeric[T]): Unit = {
    val conv = module.module.asInstanceOf[SpatialConvolution[T]]
    val offsets = new Array[Int](conv.weight.length)
    val allWeights = new Array[Byte](conv.nOutputPlane * conv.nInputPlane *
            conv.kernelH * conv.kernelW / conv.nGroup)

    var currentOffset = 0
    for (i <- conv.weight.indices) {
      offsets(i) = conv.weight(i).size().product
      System.arraycopy(conv.weight(i).getStorage, 0, allWeights, currentOffset, offsets(i))
      currentOffset += offsets(i)
    }

    val offsetBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(offsetBuilder, offsets, universe.typeOf[Array[Int]])
    modelBuilder.putAttr("offsets", offsetBuilder.build)

    val weightBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(weightBuilder, allWeights, universe.typeOf[Array[Byte]])
    modelBuilder.putAttr("weights", weightBuilder.build)
  }

  override def loadWeight[T: ClassTag](model: BigDLModule,
    module: ModuleData[T])(implicit ev: TensorNumeric[T]): Unit = {
    val conv = module.module.asInstanceOf[SpatialConvolution[T]]
    val attrMap = model.getAttrMap
    val offsets = DataConverter.getAttributeValue(attrMap.get("offsets")).asInstanceOf[Array[Int]]
    val byteArray = DataConverter.getAttributeValue(attrMap.get("weights"))
            .asInstanceOf[Array[Byte]]

    var currentOffset = 0
    conv.weight = new Array[QuantTensor[T]](offsets.length)
    for (i <- conv.weight.indices) {
      conv.weight(i) = new QuantTensor[T](conv.nOutputPlane / conv.nGroup,
        conv.nInputPlane / conv.nGroup, conv.kernelH, conv.kernelW)
      val storage = new Array[Byte](conv.weight(i).size().product)
      System.arraycopy(byteArray, currentOffset, storage, 0, offsets(i))
      currentOffset += offsets(i)
      conv.weight(i).setStorage(storage)
    }
  }
}

