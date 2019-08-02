/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http->//www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


package com.intel.analytics.bigdl.onnx

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.quantized.Utils.ModuleNode
import com.intel.analytics.bigdl.onnx.OperationTranslation._
import com.intel.analytics.bigdl.tensor.Tensor
import org.bytedeco.onnx.NodeProto

import scala.collection.mutable

object OperationConverter {

  val operatorMap = Map[String, (NodeProto,
    mutable.Map[String, Array[Int]]) => ModuleNode[Float]] (
  // Generator Functions
//  "Constant"   -> constant,
//  "RandomUniform"     -> randomUniform,
//  "RandomNormal"      -> randomNormal,
//  "RandomUniformLike" -> randomUniformLike,
//  "RandomNormalLike"  -> randomNormalLike,
//  "Multinomial"       -> multinomial,
//
//  // Arithmetic Operators
//  "Add"               -> add,
//  "Sub"               -> sub,
//  "Mul"               -> mul,
//  "Div"               -> div,
//  "Abs"               -> abs,
//  "Neg"               -> neg,
//  "Sum"               -> sum,
//
//  // Hyperbolic functions
//  "Tanh"              -> tanh,
//
//  // Rounding
//  "Ceil"              -> ceil,
//  "Floor"             -> floor,
//
//  // Joining and spliting
//  "Concat"            -> concat,
//
//  // Basic neural network functions
//  "Sigmoid"           -> sigmoid,
  "Relu"              -> relu,
//  "Pad"               -> pad,
//  "MatMul"            -> matMul, // linalg_gemm2
  "Conv"              -> conv,
//  "ConvTranspose"     -> convTranspose,
  "BatchNormalization"-> batchNormalization,
//  "SpatialBN"         -> spatialBN,
//  "LeakyRelu"         -> leakyRelu,
//  "Elu"               -> elu,
//  "PRelu"             -> pRelu,
//  "Selu"              -> selu,
//  "Softmax"           -> softmax,
//  "FC"                -> fc,
//  "GlobalAveragePool" -> globalAveragePool,
//  "GlobalMaxPool"     -> globalMaxPool,
//  "GlobalLpPool"      -> globalLpPool,
//  "Gemm"              -> gemm,
//  "LRN"               -> lrn,
//  "Dropout"           -> dropout,
//
//  // Changing shape and type.
//  "Reshape"           -> reshape,
//  "Cast"              -> cast,
//  "Split"             -> split,
//  "Slice"             -> slice,
//  "Transpose"         -> transpose,
//  "Squeeze"           -> squeeze,
//  "Unsqueeze"         -> unsqueeze,
//  "Flatten"           -> flatten,
//  "Identity"          -> identity,
//
//  // Powers
//  "Reciprocal"        -> reciprocal,
//  "Sqrt"              -> squareroot,
//  "Pow"               -> power,
//  "Exp"               -> exponent,
//  "Log"               -> log,
//
//  // Reduce Functions
//  "ReduceMax"         -> reduceMax,
//  "ReduceMean"        -> reduceMean,
//  "ReduceMin"         -> reduceMin,
//  "ReduceSum"         -> reduceSum,
//  "ReduceProd"        -> reduceProd,
//  "AveragePool"       -> averagePool,
  "MaxPool"           -> maxPool,
//
//  // Sorting and Searching
//  "ArgMax"            -> argMax,
//  "ArgMin"            -> argMin,
//  "Max"               -> max,
//  "Min"               -> min,
//  "Clip"              -> clip,
//  "ReduceLogSum"      -> reduceLogSum,
//  "ReduceLogSumExp"   -> reduceLogSumExp,
//  "ReduceSumSquare"   -> reduceSumSquare,
//  "ReduceL1"          -> reduceL1,
//  "ReduceL2"          -> reduceL2,
//  "MaxRoiPool"        -> maxRoiPool,
//  "InstanceNormalization" -> instanceNormalization,
//  "LogSoftmax"        -> logSoftmax,
//  "Softsign"          -> softsign,
//  "Less"              -> less,
//  "Greater"           -> greater,
//  "Equal"             -> equal,
//  "And"               -> and,
//  "Xor"               -> xor,
//  "Not"               -> not,
//  "Or"                -> or,
//  "Mean"              -> mean,
//  "Acos"              -> acos,
//  "Asin"              -> asin,
//  "Atan"              -> atan,
//  "Cos"               -> cos,
//  "Sin"               -> sin,
//  "Softplus"          -> softplus,
//  "Tan"               -> tan,
//  "Shape"             -> shape,
//  "Size"              -> size,
//  "Gather"            -> gather,
//  "HardSigmoid"       -> hardSigmoid,
//  "LpPool"            -> lpPool,
//  "DepthToSpace"      -> depthToSpace,
//  "SpaceToDepth"      -> spaceToDepth,
//  "Hardmax"           -> hardmax,
//  "LpNormalization"   -> lpNormalization
    "dummy" -> null
  )

  def convertOp(opNode: NodeProto,
            tensorLookup: mutable.Map[String, Array[Int]]): ModuleNode[Float] = {
    val opName = opNode.op_type().getString
    operatorMap.getOrElse(opName,
      throw new UnsupportedOperationException)(opNode, tensorLookup)
    null
  }

}
