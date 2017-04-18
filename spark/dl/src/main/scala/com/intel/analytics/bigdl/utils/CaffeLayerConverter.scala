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
import com.intel.analytics.bigdl.nn.abstractnn.TensorModule

import scala.reflect.ClassTag


object CaffeLayerConverter {
  def fromCaffe[T: ClassTag](layer : LayerParameter) : Module[T] = {
    val module = null
    layer.getType match {
      case "Convolution" => fromCaffeConvolution(layer)
      case _ => fromCaffeConvolution(layer)
    }
    return null
  }
  def fromCaffeConvolution[T: ClassTag](layer : LayerParameter) : Module[T] = {
    println("axis " + layer.getConvolutionParam.getAxis)
    println("batch size " + layer.getDataParam.getBatchSize)
    //layer.getPoolingParam.get
    return null
  }
}

