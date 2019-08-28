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

package com.intel.analytics.bigdl.python.api

import scala.collection.JavaConverters._
import java.util.{ArrayList => JArrayList, HashMap => JHashMap, List => JList, Map => JMap}

import com.intel.analytics.bigdl.nn.onnx._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag


object PythonBigDLOnnx {

  def ofFloat(): PythonBigDLOnnx[Float] = new PythonBigDLOnnx[Float]()

  def ofDouble(): PythonBigDLOnnx[Double] = new PythonBigDLOnnx[Double]()

}


class PythonBigDLOnnx[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonBigDL[T] {

  def createAveragePool(
    kernel_shape: JList[Int],
    auto_pad: String = "NOTSET",
    ceil_mode: Int = 0,
    count_include_pad: Int = 0,
    pads: JList[Int] = null,
    strides: JList[Int] = null): AveragePool[T] = {

    AveragePool(
      kernel_shape.asScala.toList,
      auto_pad,
      ceil_mode,
      count_include_pad,
      pads.asScala.toList,
      strides.asScala.toList)
  }

  def createBatchNormalization(
    n_output: Int, // number of output channels, BigDL requires.
    epsilon: Float,
    momentum: Float): BatchNormalization[T] = {
    BatchNormalization(nOutput = n_output, epsilon = epsilon, momentum = momentum)
  }

  def createConcat(
    n_input_dims: Int, // BigDL requires
    axis: Int = 0): Concat[T] = {
    Concat(nInputDims = n_input_dims, axis = axis)
  }

  def createConv(
    n_input_plane: Int, // BigDL requires
    n_output_plane: Int, // BigDL requires
    kernel_shape: JList[Int],
    weight: JTensor, // BigDL requires
    bias: JTensor, // BigDL requires
    auto_pad: String = "NOTSET", // missing in BigDL
    dilations: JList[Int] = null,
    group: Int = 1,
    pads: JList[Int] = null,
    strides: JList[Int] = null
  ): Conv[T] = {
    Conv(
      nInputPlane = n_input_plane,
      nOutputPlane = n_output_plane,
      kernelShape = kernel_shape.asScala.toList,
      weight = toTensor(weight),
      bias = toTensor(bias),
      autoPad = auto_pad,
      dilations = dilations.asScala.toList,
      group = group,
      pads = pads.asScala.toList,
      strides = strides.asScala.toList
    )
  }

  def createGather(axis: Int = 0): Gather[T, T] = {
    Gather[T, T](axis = axis)
  }

  def createGemm(alpha: Float = Int.int2float(1), beta: Float = Int.int2float(1),
                 trans_a: Int = 0, trans_b: Int = 0): Gemm[T] = {
    Gemm(alpha, beta,
      if (trans_a == 0) false else true,
      if (trans_b == 0) false else true)
  }

  def createMaxPool(auto_pad: String = "NOTSET", ceil_mode: Int = 0,
    dilations: JList[Int], kernel_shape: JList[Int], pads: JList[Int],
    storage_order: Int = 0, strides: JList[Int]): MaxPool[T] = {
    MaxPool(autoPad = auto_pad,
      ceilMode = ceil_mode,
      dilations = dilations.asScala.toList,
      kernelShape = kernel_shape.asScala.toList,
      pads = pads.asScala.toList,
      storageOrder = storage_order,
      strides = strides.asScala.toList
    )
  }

  def createRelu(): Relu[T] = {
    Relu()
  }

  def createReshape(size: Array[Int]): Reshape[T] = {
    Reshape(size)
  }

  def createShape(): Shape[T] = {
    Shape()
  }

  def createSoftmax(axis: Int = 1): Softmax[T] = {
    Softmax(axis = axis)
  }

  def createOnnxSum(inplace: Boolean = false): OnnxSum[T] = {
    OnnxSum[T](inplace = inplace)
  }

  def createUnsqueeze(axes: JList[Int], nInputDims: Int): Unsqueeze[T] = {
    Unsqueeze(axes = axes.asScala.toList, numInputDims = nInputDims)
  }
}
