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
    kernelShape: JList[Int],
    autoPad: String,
    ceilMode: Int,
    countIncludePad: Int,
    pads: JList[Int],
    strides: JList[Int]): nn.SpatialAveragePooling[T] = {

    val (kW: Int, kH: Int) = kernelShape.asScala.toList match {
      case List(width, height) => (width, height)
      case _ => throw new IllegalArgumentException(
        "Kernel shape is expected in the form of List(width, height)," +
          "the input kernel shape: " + kernelShape)
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

    nn.SpatialAveragePooling[T](
      kW = kW, kH = kH, dW = dW, dH = dH, padW = padW, padH = padH,
      ceilMode = if (ceilMode == 0) false else true,
      countIncludePad = if (countIncludePad == 0) false else true
    )
  }


  def createBatchNormalization(
    numFeatures: Int, // number of input features, BigDL requires.
    epsilon: Float,
    momentum: Float): nn.SpatialBatchNormalization[T] = {

    nn.SpatialBatchNormalization[T](
      nOutput = numFeatures, eps = epsilon.toDouble, momentum = momentum.toDouble
    )
  }


  def createConcat(nInputDims: Int, axis: Int): nn.JoinTable[T] = {
    nn.JoinTable[T](dimension = axis, nInputDims = nInputDims)
  }


  def createConstant(value: JTensor): nn.tf.Const[T, T] = {
    nn.tf.Const[T, T](toTensor(value))
  }


  def createConv(
    nInputPlane: Int, // BigDL requires
    nOutputPlane: Int, // BigDL requires
    kernelShape: JList[Int],
    weight: JTensor, // BigDL requires
    bias: JTensor, // BigDL requires
    autoPad: String, // missing in BigDL
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

    val (kernelW: Int, kernelH: Int) = kernelShape.asScala.toList match {
      case List(width: Int, height: Int) => (width, height)
      case _ => throw new IllegalArgumentException(
        "Kernel shape is expected in the form of List(width, height)," +
          "the input kernel shape: " + kernelShape)
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

    nn.SpatialConvolution[T](
      nInputPlane = nInputPlane, nOutputPlane = nOutputPlane,
      kernelW = kernelW, kernelH = kernelH, strideW = strideW, strideH = strideH,
      padW = padW, padH = padH, nGroup = group,
      initWeight = toTensor(weight), initBias = toTensor(bias),
      withBias = if (bias != null) true else false
    )
  }


  def createGather(axis: Int): nn.ops.Gather[T, T] = {
    nn.ops.Gather[T, T]()
  }


  def createGemm(alpha: Float, beta: Float, transA: Int, transB: Int,
                 matrixB: JTensor, matrixC: JTensor): Gemm[T] = {
    Gemm(alpha, beta,
      (if (transA == 0) false else true),
      (if (transB == 0) false else true),
      toTensor(matrixB), toTensor(matrixC))
  }


  def createMaxPool(kernelShape: JList[Int], autoPad: String,
    ceilMode: Int, dilations: JList[Int], pads: JList[Int],
    storageOrder: Int, strides: JList[Int]): nn.SpatialMaxPooling[T] = {

    val (kW: Int, kH: Int) = kernelShape.asScala.toList match {
      case List(width, height) => (width, height)
      case _ => throw new IllegalArgumentException(
        "Kernel shape is expected in the form of List(width, height)," +
          "the input kernel shape: " + kernelShape)
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

    if (ceilMode != 0) {
      throw new IllegalArgumentException("MaxPool doesnt support ceil mode yet.")
    }

    if (storageOrder != 0) {
      throw new IllegalArgumentException("MaxPool doesnt support storage order yet.")
    }

    nn.SpatialMaxPooling[T](kW = kW, kH = kH, dW = dW, dH = dH,
      padW = padW, padH = padH)
  }


  def createRelu(): nn.ReLU[T] = {
    nn.ReLU[T](ip = true)
  }


  def createReshape(shape: JArrayList[Int]): nn.Reshape[T] = {
    nn.Reshape[T](shape.asScala.toArray, Some(false))
  }


  def createShape(): Shape[T] = {
    nn.onnx.Shape[T]()
  }


  def createSoftmax(axis: Int = 1): nn.SoftMax[T] = {
    nn.SoftMax[T]()
  }


  def createOnnxSum(inplace: Boolean = false): nn.CAddTable[T, T] = {
    nn.CAddTable[T](inplace)
  }

  
  def createUnsqueeze(axes: JList[Int], numInputDims: Int): nn.Unsqueeze[T] = {
    val pos = axes.asScala.toList match {
      case List(elem) => elem + 1 // Todo
      case _ => throw new IllegalArgumentException("Bad axes value: " + axes)
    }
    nn.Unsqueeze[T](pos = pos, numInputDims = numInputDims)
  }
}
