/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.nn.mkl

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.MklTensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.language.implicitConversions
import scala.reflect.ClassTag

abstract class MklModule[@specialized(Float, Double) T: ClassTag](implicit ev: TensorNumeric[T])
    extends TensorModule[T] {
  var forwardPrim = 0L
  var backwardPrim = 0L

  private[this] var _firstPassForward: Boolean = true

  def firstPassForward: Boolean = _firstPassForward

  def firstPassForward_=(value: Boolean): Unit = {
    _firstPassForward = value
  }

  private[this] var _firstPassBackward: Boolean = true

  def firstPassBackward: Boolean = _firstPassBackward

  def firstPassBackward_=(value: Boolean): Unit = {
    _firstPassBackward = value
  }

  val inputSize = new Array[Long](4)
  val outputSize = new Array[Long](4)
  val inputStrides = new Array[Long](4)
  val outputStrides = new Array[Long](4)

  var inputMkl = new MklTensor[T]()
  var outputMkl = new MklTensor[T]()
  var gradOutputMkl = new MklTensor[T]()
  var gradInputMkl = new MklTensor[T]()

  trait Ref {
    var input = new MklTensor[T]()
    var output = new MklTensor[T]()
    var gradOutput = new MklTensor[T]()
    var gradInput = new MklTensor[T]()
  }

  implicit def bool2int(b: Boolean) = if (b) 1 else 0
}

object ResourceType {
  val dnnResourceSrc            = 0
  val dnnResourceFrom           = 0
  val dnnResourceDst            = 1
  val dnnResourceTo             = 1
  val dnnResourceFilter         = 2
  val dnnResourceScaleShift     = 2
  val dnnResourceBias           = 3
  val dnnResourceDiffSrc        = 4
  val dnnResourceDiffFilter     = 5
  val dnnResourceDiffScaleShift = 5
  val dnnResourceDiffBias       = 6
  val dnnResourceDiffDst        = 7
  val dnnResourceWorkspace      = 8
  val dnnResourceMultipleSrc    = 16
  val dnnResourceMultipleDst    = 24
  val dnnResourceNumber         = 32
}

object Border {
  val dnnBorderZeros          = 0x0
  val dnnBorderExtrapolation  = 0x3
}

object Error {
  val E_SUCCESS                   =  0
  val E_INCORRECT_INPUT_PARAMETER = -1
  val E_UNEXPECTED_NULL_POINTER   = -2
  val E_MEMORY_ERROR              = -3
  val E_UNSUPPORTED_DIMENSION     = -4
  val E_UNIMPLEMENTED             = -127
}

object Algorithm {
  val dnnAlgorithmConvolutionGemm  = 0 // GEMM based convolution
  val dnnAlgorithmConvolutionDirect= 1 // Direct convolution
  val dnnAlgorithmConvolutionFFT   = 2 // FFT based convolution
  val dnnAlgorithmPoolingMax       = 3 // Maximum pooling
  val dnnAlgorithmPoolingMin       = 4 // Minimum pooling
  val dnnAlgorithmPoolingAvg       = 5 // Average pooling
}

object DataSize {
  val FLOAT  = 4
  val INT    = 4
  val DOUBLE = 8
}

object MklRWType {
  val READ = 0
  val WRITE = 1
}

object ConvertType {
  val INTERNALTOUSR = 0
  val INTERNALTOMKL = 1
  val MKLTOUSR = 2
  val USRTOMKL = 3
  val MKL = 4
  val USR = 5
}
