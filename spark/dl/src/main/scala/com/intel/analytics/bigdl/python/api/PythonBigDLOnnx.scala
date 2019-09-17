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

import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.nn.onnx._
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric


import scala.reflect.ClassTag


object PythonBigDLOnnx {

  def ofFloat(): PythonBigDLOnnx[Float] = new PythonBigDLOnnx[Float]()

  def ofDouble(): PythonBigDLOnnx[Double] = new PythonBigDLOnnx[Double]()

}


class PythonBigDLOnnx[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonBigDL[T] {

  def createAveragePool(
    kernel_shape: JList[Int],
    auto_pad: String,
    ceil_mode: Int,
    count_include_pad: Int,
    pads: JList[Int],
    strides: JList[Int]): nn.SpatialAveragePooling[T] = {

    val (kW: Int, kH: Int) = kernel_shape.asScala.toList match {
      case List(width, height) => (width, height)
      case _ => throw new IllegalArgumentException(
        "Kernel shape is expected in the form of List(width, height)," +
          "the input kernel shape: " + kernel_shape)
    }
    val (dW: Int, dH: Int) = strides.asScala.toList match {
      case List(width, height) => (width, height)
      case _ => throw new IllegalArgumentException(
        "Strides is expected in the form of List(width, height)," +
          "the input strides: " + strides)
    }
    val (padW: Int, padH: Int) = pads.asScala.toList match {
      case List(width, height) => (width, height)
      case _ => throw new IllegalArgumentException(
        "Pads is expected in the form of List(width, height)," +
          "the input pads: " + pads)
    }

    val globalPooling = false
    val ceilMode = if (ceil_mode == 0) false else true
    val countIncludePad = if (count_include_pad == 0) false else true
    val divide = false
    val format = DataFormat.NCHW

    OnnxOpsMapper.averagePool.apply(
      kW, kH, dW, dH, padW, padH,
      globalPooling, ceilMode, countIncludePad, divide, format
    )

  }


  def createBatchNormalization(
    num_features: Int, // number of input features, BigDL requires.
    epsilon: Float,
    momentum: Float): nn.SpatialBatchNormalization[T] = {

    val nOutput = num_features
    val eps = epsilon.toDouble
    val affine = true
    val initWeight = null
    val initBias = null
    val initGradWeight = null
    val initGradBias = null
    val dataFormat = DataFormat.NCHW

    OnnxOpsMapper.batchNormalization.apply(
      nOutput, eps.toDouble, momentum.toDouble,
      affine, initWeight, initBias, initGradWeight, initGradBias, dataFormat
    )

  }


  def createConcat(n_input_dims: Int, axis: Int): nn.JoinTable[T] = {
    val dimension = axis
    val nInputDims = n_input_dims
    OnnxOpsMapper.concat.apply(dimension, nInputDims)
  }


  def createConstant(value: JTensor): nn.tf.Const[T, T] = {
    OnnxOpsMapper.constant.apply(toTensor(value))
  }

  def createConv(
    n_input_plane: Int, // BigDL requires
    n_output_plane: Int, // BigDL requires
    kernel_shape: JList[Int],
    weight: JTensor, // BigDL requires
    bias: JTensor, // BigDL requires
    auto_pad: String, // missing in BigDL
    dilations: JList[Int],
    group: Int,
    pads: JList[Int],
    strides: JList[Int]
  ): nn.SpatialConvolution[T] = {
    val (dilationW: Int, dilationH: Int) = dilations.asScala.toList match {
      case List(width: Int, height: Int) => (width.toInt, height.toInt)
      case _ => throw new IllegalArgumentException(
        "Dilations is expected in the form of List(width, height)," +
          "the input dilations: " + dilations)
    }

    val (kernelW: Int, kernelH: Int) = kernel_shape.asScala.toList match {
      case List(width: Int, height: Int) => (width, height)
      case _ => throw new IllegalArgumentException(
        "Kernel shape is expected in the form of List(width, height)," +
          "the input kernel shape: " + kernel_shape)
    }

    val (strideW: Int, strideH: Int) = strides.asScala.toList match {
      case List(width: Int, height: Int) => (width, height)
      case _ => throw new IllegalArgumentException(
        "Strides is expected in the form of List(width, height)," +
          "the input strides: " + strides)
    }

    val (padW: Int, padH: Int) = pads.asScala.toList match {
      case List(width: Int, height: Int) => (width, height)
      case _ => throw new IllegalArgumentException(
        "Pads is expected in the form of List(width, height)," +
          "the input pads: " + strides)
    }


    if (dilationH != 1 && dilationW != 1) {
      throw new UnsupportedOperationException(
        "Dilations is expected to be (1, 1)" +
          "the input dilations: " + (dilationW, dilationH))
    }

    val nInputPlane = n_input_plane
    val nOutputPlane = n_output_plane
    val autoPad = auto_pad
    val nGroup = group
    val propagateBack: Boolean = true
    val wRegularizer: Regularizer[T] = null
    val bRegularizer: Regularizer[T] = null
    val initWeight: Tensor[T] = null
    val initBias: Tensor[T] = null
    val initGradWeight: Tensor[T] = null
    val initGradBias: Tensor[T] = null
    val withBias = if (bias != null) true else false
    val format: DataFormat = DataFormat.NCHW

    OnnxOpsMapper.conv.apply(
      nInputPlane, nOutputPlane, kernelW, kernelH, strideW, strideH,
      padW, padH, nGroup, propagateBack, wRegularizer, bRegularizer,
      initWeight, initBias, initGradWeight, initGradBias, withBias, format
    )

  }

  def createGather(axis: Int): nn.ops.Gather[T, T] = {
    OnnxOpsMapper.gather.apply()
  }


  def createGemm(alpha: Float, beta: Float, trans_a: Int, trans_b: Int): Gemm[T] = {
    val transA = (if (trans_a == 0) false else true)
    val transB = (if (trans_b == 0) false else true)
    OnnxOpsMapper.gemm.apply(alpha, beta, transA, transB)
  }


  def createMaxPool(kernel_shape: JList[Int], auto_pad: String,
    ceil_mode: Int, dilations: JList[Int], pads: JList[Int],
    storage_order: Int, strides: JList[Int]): nn.SpatialMaxPooling[T] = {

    val (kW: Int, kH: Int) = kernel_shape.asScala.toList match {
      case List(width, height) => (width, height)
      case _ => throw new IllegalArgumentException(
        "Kernel shape is expected in the form of List(width, height)," +
          "the input kernel shape: " + kernel_shape)
    }
    val (dW: Int, dH: Int) = strides.asScala.toList match {
      case null => (1, 1)
      case List(width, height) => (width, height)
      case _ => throw new IllegalArgumentException(
        "Strides is expected in the form of List(width, height)," +
          "the input strides: " + strides)
    }
    val (padW: Int, padH: Int) = pads.asScala.toList match {
      case null => (0, 0)
      case List(width, height) => (width, height)
      case _ => throw new IllegalArgumentException(
        "Pads is expected in the form of List(width, height)," +
          "the input pads: " + pads)
    }

    if (ceil_mode != 0) {
      throw new IllegalArgumentException("MaxPool doesnt support ceil mode yet.")
    }

    if (storage_order != 0) {
      throw new IllegalArgumentException("MaxPool doesnt support storage order yet.")
    }

    val format = DataFormat.NCHW

    OnnxOpsMapper.maxPool.apply(kW, kH, dW, dH, padW, padH, format)

  }

  def createRelu(inplace: Boolean = false): nn.ReLU[T] = {
    OnnxOpsMapper.relu.apply(inplace)
  }

  def createReshape(): Reshape[T] = {
    OnnxOpsMapper.reshape.apply()
  }

  def createShape(): Shape[T] = {
    OnnxOpsMapper.shape.apply()
  }

  def createSoftmax(axis: Int = 1): nn.SoftMax[T] = {
    OnnxOpsMapper.softmax.apply()
  }

  def createOnnxSum(inplace: Boolean = false): nn.CAddTable[T, T] = {
    OnnxOpsMapper.sum.apply(inplace)
  }

  def createUnsqueeze(axes: JList[Int], num_input_dims: Int): nn.Unsqueeze[T] = {
    val pos = axes.asScala.toList match {
      case List(elem) => elem + 1 // Todo
      case _ => throw new IllegalArgumentException("Bad axes value: " + axes)
    }
    val numInputDims: Int = num_input_dims
    OnnxOpsMapper.unsqueeze.apply(pos, numInputDims)
  }
}


object OnnxOpsMapper {

  def averagePool[T: ClassTag](implicit ev: TensorNumeric[T]):
  (Int, Int, Int, Int, Int, Int, Boolean, Boolean, Boolean, Boolean, DataFormat)
    => nn.SpatialAveragePooling[T] = nn.SpatialAveragePooling[T]

  def batchNormalization[T: ClassTag](implicit ev: TensorNumeric[T]):
  (Int, Double, Double, Boolean, Tensor[T], Tensor[T], Tensor[T],
    Tensor[T], DataFormat) => nn.SpatialBatchNormalization[T] = nn.SpatialBatchNormalization[T]

  def concat[T: ClassTag](implicit ev: TensorNumeric[T]):
  (Int, Int) => nn.JoinTable[T] = nn.JoinTable[T]

  def constant[T: ClassTag](implicit ev: TensorNumeric[T]):
  Tensor[T] => nn.tf.Const[T, T] = nn.tf.Const[T, T]

  def conv[T: ClassTag](implicit ev: TensorNumeric[T]):
  (Int, Int, Int, Int, Int, Int, Int, Int, Int,
    Boolean, Regularizer[T], Regularizer[T], Tensor[T], Tensor[T],
    Tensor[T], Tensor[T], Boolean, DataFormat) => nn.SpatialConvolution[T]
  = nn.SpatialConvolution[T]

  def gather[T: ClassTag, D: ClassTag](implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]):
  () => nn.ops.Gather[T, D] = nn.ops.Gather[T, D]

  def gemm[T: ClassTag](implicit ev: TensorNumeric[T]):
  (Float, Float, Boolean, Boolean) => nn.onnx.Gemm[T] = nn.onnx.Gemm[T]

  def maxPool[T: ClassTag](implicit ev: TensorNumeric[T]):
  (Int, Int, Int, Int, Int, Int, DataFormat) => nn.SpatialMaxPooling[T] = nn.SpatialMaxPooling[T]

  def relu[T: ClassTag](implicit ev: TensorNumeric[T]): Boolean => nn.ReLU[T] = nn.ReLU[T]

  def reshape[T: ClassTag](implicit ev: TensorNumeric[T]):
  () => nn.onnx.Reshape[T] = nn.onnx.Reshape[T]

  def shape[T: ClassTag](implicit ev: TensorNumeric[T]): () => nn.onnx.Shape[T] = nn.onnx.Shape[T]

  def softmax[T: ClassTag](implicit ev: TensorNumeric[T]): () => nn.SoftMax[T] = nn.SoftMax[T]

  def sum[T: ClassTag](implicit ev: TensorNumeric[T]):
  (Boolean) => nn.CAddTable[T, T] = nn.CAddTable[T]

  def unsqueeze[T: ClassTag](implicit ev: TensorNumeric[T]):
  (Int, Int) => nn.Unsqueeze[T] = nn.Unsqueeze[T]

}