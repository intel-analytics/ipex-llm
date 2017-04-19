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

package com.intel.analytics.bigdl.utils

import caffe.Caffe.LayerParameter
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.SpatialConvolution
import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag


object CaffeLayerConverter {
  def fromCaffe[T: ClassTag](layer : LayerParameter)(
    implicit ev: TensorNumeric[T]) : Module[T] = {
    val module = null
    layer.getType match {
      case "Convolution" => fromCaffeConvolution(layer)
      case "Input" => fromCaffeData(layer)
      case _ => fromCaffeConvolution(layer)
    }
    return null
  }
  def fromCaffeConvolution[T: ClassTag](layer : LayerParameter)(
    implicit ev: TensorNumeric[T]) : Module[T] = {
    val param = layer.getConvolutionParam
    println(param)
    println(layer.getName)
    println(layer.getBlobsList.get(0))
    val group = if (param.getGroup == 0)  1 else param.getGroup
    val nInputPlane = layer.getBlobs(0).getChannels * group
    val nOutPlane = layer.getBlobs(0).getNum
    var kw = param.getKernelW
    var kh = param.getKernelH
    var dw = param.getStrideW
    var dh = param.getStrideH
    if (kw ==0 || kh == 0) {
      kw = param.getKernelSize(0)
      kh = kw
    }
    if (dw == 0 || dh == 0) {
      dw = param.getStride(0)
      dh = dw
    }
    var pw = param.getPadW
    var ph = param.getPadH
    if (pw == 0 || ph == 0) {
      pw = param.getPad(0)
      ph = pw
    }
    val name = layer.getName
    SpatialConvolution[T](nInputPlane, nOutPlane, kw, kh, dw, dh, pw, ph, group)
  }

  def fromCaffeData[T: ClassTag](layer : LayerParameter) : Module[T] = {
    println("channel size : " + layer.getInputParam.getShape(0))

    return  null
  }
}

