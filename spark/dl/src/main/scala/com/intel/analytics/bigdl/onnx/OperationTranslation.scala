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

package com.intel.analytics.bigdl.onnx

import com.intel.analytics.bigdl.nn.{BatchNormalization, ReLU, SpatialConvolution, SpatialMaxPooling}
import com.intel.analytics.bigdl.tensor.Tensor
import org.bytedeco.onnx.{AttributeProto, NodeProto}
import com.intel.analytics.bigdl.utils.{ReflectionUtils, Shape}

import scala.collection.mutable


object OperationTranslation {

//  def constant(): Object = {
//    null
//  }
//
//  def randomUniform(): Object = {
//    null
//  }
//
//  def randomNormal(): Object = {
//    null
//  }
//
//  def randomUniformLike(): Object = {
//    null
//  }
//
//  def randomNormalLike(): Object = {
//    null
//  }
//
//  def multinomial(): Object = {
//    null
//  }
//
//  def add(): Object = {
//    null
//  }
//
//  def sub(): Object = {
//    null
//  }
//
//  def mul(): Object = {
//    null
//  }
//
//  def div(): Object = {
//    null
//  }
//
//  def abs(): Object = {
//    null
//  }
//
//  def neg(): Object = {
//    null
//  }
//
//  def sum(): Object = {
//    null
//  }
//
//  def tanh(): Object = {
//    null
//  }
//
//  def ceil(): Object = {
//    null
//  }
//
//  def floor(): Object = {
//    null
//  }
//
//  def concat(): Object = {
//    null
//  }
//
//  def sigmoid(): Object = {
//    null
//  }

  def relu(node: NodeProto,
           tensorLookup: mutable.Map[String, Array[Int]]): ReLU[Float] = {

    val currDataTensorName = node.input(0).getString
    val currOutTensorName = node.output(0).getString
    val currDataTensorDims = tensorLookup.get(currDataTensorName).get

    val bNode = ReLU[Float]()

    val inputShape = Shape(currDataTensorDims)
    val outputShape = bNode.computeOutputShape(inputShape)

    tensorLookup.put(currOutTensorName, outputShape.toSingle().toArray[Int])

    bNode
  }

//  def pad(): Object = {
//    null
//  }
//
//  def matMul(): Object = {
//    null
//  }

//  val nInputPlane: Int, // The number of expected input planes in the image given into forward()
//  val nOutputPlane: Int, // The number of output planes the convolution layer will produce.
//  val kernelW: Int, // The kernel width of the convolution
//  val kernelH: Int, // The kernel height of the convolution
//  val strideW: Int = 1, // The step of the convolution in the width dimension.
//  val strideH: Int = 1, // The step of the convolution in the height dimension
//  val padW: Int = 0, // The additional zeros added per width to the input planes.
//  val padH: Int = 0, // The additional zeros added per height to the input planes.
//  val nGroup: Int = 1, // Kernel group number
//  val propagateBack: Boolean = true, // propagate gradient back
//  var wRegularizer: Regularizer[T] = null,
//  var bRegularizer: Regularizer[T] = null,
//  val initWeight: Tensor[T] = null,
//  val initBias: Tensor[T] = null,
//  val initGradWeight: Tensor[T] = null,
//  val initGradBias: Tensor[T] = null,
//  val withBias: Boolean = true,
//  val format: DataFormat = DataFormat.NCHW
  def conv(node: NodeProto,
           tensorLookup: mutable.Map[String, Array[Int]]): SpatialConvolution[Float] = {

    val currNodeAttrSize = node.attribute_size()

    val attrMap: Map[String, AttributeProto] = (0 until currNodeAttrSize).map(i => {
      val currAttr = node.attribute(i)
      val attrName = currAttr.name().getString
      (attrName, currAttr)
    }).toMap[String, AttributeProto]

    val auto_pad = if (attrMap.contains("auto_pad")) {
      attrMap.get("auto_pad").get.s().getString
    } else {
      "NOTSET"
    } // unused in BigDL

    val dilations = if (attrMap.contains("dilations")) {
      (0 until attrMap.get("dilations").get.ints_size()).map(i =>
        attrMap.get("dilations").get.ints(i).toInt
      ).toArray[Int]
    } else {
      Array[Int](1, 1)
    }
    val group = if (attrMap.contains("group")) {
      attrMap.get("group").get.i().toInt
    } else {
      1
    }

    val pads = if (attrMap.contains("pads")) {
      (0 until attrMap.get("pads").get.ints_size()).map(i =>
        attrMap.get("pads").get.ints(i).toInt
      ).toArray[Int]
    } else {
      Array[Int](0, 0)
    }

    val kernel_shape = (0 until attrMap.get("kernel_shape").get.ints_size()).map(i =>
      attrMap.get("kernel_shape").get.ints(i).toInt
    ).toArray[Int]

    val strides = (0 until attrMap.get("strides").get.ints_size()).map(i =>
      attrMap.get("strides").get.ints(i).toInt
    ).toArray[Int]


    val currDataTensor = node.input(0).getString
    val currWeightTensor = node.input(1).getString
    val currOutputTensor = node.output(0).getString

    val currDataTensorDims = tensorLookup.get(currDataTensor).get
    val currWeightTensorDims = tensorLookup.get(currWeightTensor).get

    val nInputPlane: Int = currDataTensorDims(1)
    val nOutputPlane: Int = currWeightTensorDims(0)
    val (kernelW, kernelH) = (kernel_shape(0), kernel_shape(1))
    val (strideW, strideH) = (strides(0), strides(1))
    val (padW, padH) = (pads(0), pads(1))
    val nGroup: Int = 1
    val withBias = if (node.input_size() == 3) true else false

    val bNode = SpatialConvolution[Float](nInputPlane, nOutputPlane, kernelW, kernelH,
      strideW = strideW, strideH = strideH, padW = padW, padH = padH,
      nGroup = nGroup, withBias = withBias)

    val inputShape = Shape(currDataTensorDims)

    val outputShape = bNode.computeOutputShape(inputShape)
    tensorLookup.put(currOutputTensor, outputShape.toSingle().toArray[Int])
    println("output shape", currOutputTensor)

    bNode
  }

  private def hasAttribute(attrMap: Map[String, AttributeProto], attName: String): Boolean = {
    attrMap.contains(attName)
  }

//  def convTranspose(): Object = {
//    null
//  }
//



  def batchNormalization(node: NodeProto,
       tensorLookup: mutable.Map[String, Array[Int]]): BatchNormalization[Float] = {

    val attrMap = (0 until node.attribute_size()).map(i => {
      val currAttr = node.attribute(i)
      val currAttrName = node.attribute(i).name().getString
      (currAttrName, currAttr)
    }).toMap

    val currDataTensorName = node.input(0).getString
    val currOutTensorName = node.output(0).getString

    val epsilon = if (attrMap.contains("epsilon")) attrMap.get("epsilon").get.f() else 1e-05
    val momentum = if (attrMap.contains("momentun")) attrMap.get("momentum").get.f() else 0.9

    val currDataTensorDims = tensorLookup.get(currDataTensorName).get

    val nOutput = (tensorLookup.get(currDataTensorName).get)(1)

    val bNode = BatchNormalization[Float](nOutput = nOutput, eps = epsilon)
    val currOutTensorDims = currDataTensorDims

    tensorLookup.put(currOutTensorName, currOutTensorDims)
    bNode

  }
//
//  def spatialBN(): Object = {
//    null
//  }
//
//  def leakyRelu(): Object = {
//    null
//  }
//
//  def elu(): Object = {
//    null
//  }
//
//  def pRelu(): Object = {
//    null
//  }
//
//  def selu(): Object = {
//    null
//  }
//
//  def softmax(): Object = {
//    null
//  }
//
//  def fc(): Object = {
//    null
//  }
//
//  def globalAveragePool(): Object = {
//    null
//  }
//
//  def globalMaxPool(): Object = {
//    null
//  }
//
//  def globalLpPool(): Object = {
//    null
//  }
//
//  def gemm(): Object = {
//    null
//  }
//
//  def lrn(): Object = {
//    null
//  }
//
//  def dropout(): Object = {
//    null
//  }
//
//  def reshape(): Object = {
//    null
//  }
//
//  def cast(): Object = {
//    null
//  }
//
//  def split(): Object = {
//    null
//  }
//
//  def slice(): Object = {
//    null
//  }
//
//  def transpose(): Object = {
//    null
//  }
//
//  def squeeze(): Object = {
//    null
//  }
//
//  def unsqueeze(): Object = {
//    null
//  }
//
//  def flatten(): Object = {
//    null
//  }
//
//  def identity(): Object = {
//    null
//  }
//
//  def reciprocal(): Object = {
//    null
//  }
//
//  def squareroot(): Object = {
//    null
//  }
//
//  def power(): Object = {
//    null
//  }
//
//  def exponent(): Object = {
//    null
//  }
//
//  def log(): Object = {
//    null
//  }
//
//  def reduceMax(): Object = {
//    null
//  }
//
//  def reduceMean(): Object = {
//    null
//  }
//
//  def reduceMin(): Object = {
//    null
//  }
//
//  def reduceSum(): Object = {
//    null
//  }
//
//  def reduceProd(): Object = {
//    null
//  }
//
//  def averagePool(): Object = {
//    null
//  }

//  * @param kW              kernel width
//    * @param kH              kernel height
//    * @param dW              step size in width
//    * @param dH              step size in height
//    * @param padW            padding in width
//  * @param padH            padding in height
  def maxPool(node: NodeProto,
              tensorLookup: mutable.Map[String, Array[Int]]): SpatialMaxPooling[Float] = {

    val attrMap = (0 until node.attribute_size()).map(i => {
      val currAttr = node.attribute(i)
      val currAttrName = currAttr.name().getString
      (currAttrName, currAttr)
    }).toMap

    val kernelAttr = attrMap.get("kernel_shape").get
    val (kernelW, kernelH) = (kernelAttr.ints(0).toInt, kernelAttr.ints(1).toInt)


    val (strideW, strideH) = if (attrMap.contains("strides")) {
      val strideAttr = attrMap.get("strides").get
      (strideAttr.ints(0).toInt, strideAttr.ints(1).toInt)
    } else {
      (1, 1)
    }

    val (padW, padH) = if (attrMap.contains("pads")) {
      val padAttr = attrMap.get("pads").get
      (padAttr.ints(0).toInt, padAttr.ints(1).toInt)
    } else {
      (0, 0)
    }

    val (dilationW, dilationH) = (0, 0)

    val ceilMode = if (attrMap.contains("ceil_mode")) {
      if (attrMap.get("ceil_mode").get.i() == 1) true else false
    } else {
      false
    }
    val bNode = SpatialMaxPooling[Float](kW = kernelW, kH = kernelH,
      dW = strideW, dH = strideH, padW = padW, padH = padH)

    if (ceilMode) bNode.ceil()

    def helper(input: Int, stride: Int, kernel: Int, pad: Int,
               dilation: Int, ceilMode: Boolean): Int = {
      def roundingFunc: Double => Double = if (ceilMode) Math.ceil else Math.floor

      roundingFunc((input + 2 * pad - (dilation - 1) * (kernel - 1)) / stride + 1).toInt
    }

    val currDataTensorName = node.input(0).getString
    val currOutTensorName = node.output(0).getString
    val currDataTensorDims = tensorLookup.get(currDataTensorName).get
    val currOutTensorDims = currDataTensorDims.clone()

    val (inputW, inputH) = (currDataTensorDims(2), currDataTensorDims(3))
    currOutTensorDims(2) = helper(inputW, strideW, kernelW, padW, dilationW, ceilMode)
    currOutTensorDims(3) = helper(inputH, strideH, kernelH, padH, dilationH, ceilMode)

    tensorLookup.put(currOutTensorName, currOutTensorDims)

    bNode
  }

//  def argMax(): Object = {
//    null
//  }
//
//  def argMin(): Object = {
//    null
//  }
//
//  def max(): Object = {
//    null
//  }
//  def min(): Object = {
//    null
//  }
//
//  def clip(): Object = {
//    null
//  }
//
//  def reduceLogSum(): Object = {
//    null
//  }
//
//  def reduceLogSumExp(): Object = {
//    null
//  }
//
//  def reduceSumSquare(): Object = {
//    null
//  }
//
//  def reduceL1(): Object = {
//    null
//  }
//
//  def reduceL2(): Object = {
//    null
//  }
//
//  def maxRoiPool(): Object = {
//    null
//  }
//
//  def instanceNormalization(): Object = {
//    null
//  }
//
//  def logSoftmax(): Object = {
//    null
//  }
//
//  def softsign(): Object = {
//    null
//  }
//
//  def less(): Object = {
//    null
//  }
//
//  def greater(): Object = {
//    null
//  }
//
//  def equal(): Object = {
//    null
//  }
//
//  def and(): Object = {
//    null
//  }
//
//  def xor(): Object = {
//    null
//  }
//
//  def not(): Object = {
//    null
//  }
//
//  def or(): Object = {
//    null
//  }
//
//  def mean(): Object = {
//    null
//  }
//
//  def acos(): Object = {
//    null
//  }
//
//  def asin(): Object = {
//    null
//  }
//
//  def atan(): Object = {
//    null
//  }
//
//  def cos(): Object = {
//    null
//  }
//
//  def sin(): Object = {
//    null
//  }
//
//  def softplus(): Object = {
//    null
//  }
//
//  def tan(): Object = {
//    null
//  }
//
//  def shape(): Object = {
//    null
//  }
//
//  def size(): Object = {
//    null
//  }
//
//  def gather(): Object = {
//    null
//  }
//
//  def hardSigmoid(): Object = {
//    null
//  }
//
//  def lpPool(): Object = {
//    null
//  }
//
//  def depthToSpace(): Object = {
//    null
//  }
//
//  def spaceToDepth(): Object = {
//    null
//  }
//
//  def hardmax(): Object = {
//    null
//  }
//
//  def lpNormalization(): Object = {
//    null
//  }

}
