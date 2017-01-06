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

package com.intel.analytics.bigdl.nn.dnn

import com.intel.analytics.bigdl.mkl.MklDnnFloat
import com.intel.analytics.bigdl.tensor.{FloatType, MklTensor}
import com.intel.analytics.bigdl.nn.abstractnn.ModuleType._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}

import scala.language.implicitConversions
import scala.reflect.ClassTag

abstract class MklModule[@specialized(Float, Double) T: ClassTag](implicit ev: TensorNumeric[T])
    extends TensorModule[T] {

  trait Ref {
    var input = new MklTensor[T]()
    var output = new MklTensor[T]()
    var gradOutput = new MklTensor[T]()
    var gradInput = new MklTensor[T]()
  }

  trait Primitive {
    var forward = 0L
    var backward = 0L
  }

  var profiling = false

  def execute(resources: Array[Long], primitive: Long): Unit = {
    ev.getType() match {
      case FloatType =>
        val start = System.nanoTime()

        MklDnnFloat.execute(resources, primitive)

        val end = System.nanoTime()
        if (profiling) {
          println("Elapsed time of " + getName() + ": " + (end - start) / 1e6)
        }
      case _ => throw new UnsupportedOperationException(s"Only Float supported")
    }
  }

  private[this] var _isInited: Boolean = false
  private[this] var _nextModuleType: ModuleType = BLAS
  private[this] var _prevModuleType: ModuleType = BLAS

  override def convertToMklDnn(prevModule: Option[AbstractModule[Activity, Activity, T]] = None)
    : (ModuleType, AbstractModule[Activity, Activity, T]) = {
    prevModule match {
      case Some(x) => setPrevModuleType(x.moduleType)
      case _ =>
    }

    (DNN, this.asInstanceOf[AbstractModule[Activity, Activity, T]])
  }

  def isInited: Boolean = _isInited
  def setInit(value: Boolean): Unit = {
    _isInited = value
  }

  override def moduleType(): ModuleType = DNN

  override def nextModuleType: ModuleType = _nextModuleType
  override def setNextModuleType(value: ModuleType): Unit = {
    value match {
      case DNN => _nextModuleType = DNN
      case _ =>
    }
  }

  override def prevModuleType: ModuleType = _prevModuleType
  override def setPrevModuleType(value: ModuleType): Unit = {
    value match {
      case DNN => _prevModuleType = DNN
      case _ =>
    }
  }

  implicit def bool2int(b: Boolean): Int = if (b) 1 else 0
}

/**
 * create a mkl layout description
 * @param dim the dimension of layout, sometimes the dimension is not the same as eles length
 * @param eles the size of tensor. Order: width, height, channels, number.
 */
class MklLayout(dim: Int, eles: Array[Long]) {

  def computeStrides(size: Array[Long]): Array[Long] = {
    val stride = new Array[Long](size.length)
    stride(0) = 1
    for (i <- 1 until size.length) {
      stride(i) = size(i - 1) * stride(i - 1)
    }

    stride
  }

  val size = eles.clone()
  val strides = computeStrides(eles)
  val dimension = dim
}

class MklPrimitive {
  private[this] var _primitive: Long = 0L

  def primitive: Long = _primitive

  def setPrimitive(value: Long): Unit = {
    _primitive = value
  }
}

object ResourceType {
  val dnnResourceSrc = 0
  val dnnResourceFrom = 0
  val dnnResourceDst = 1
  val dnnResourceTo = 1
  val dnnResourceFilter = 2
  val dnnResourceScaleShift = 2
  val dnnResourceBias = 3
  val dnnResourceDiffSrc = 4
  val dnnResourceDiffFilter = 5
  val dnnResourceDiffScaleShift = 5
  val dnnResourceDiffBias = 6
  val dnnResourceDiffDst = 7
  val dnnResourceWorkspace = 8
  val dnnResourceMultipleSrc = 16
  val dnnResourceMultipleDst = 24
  val dnnResourceNumber = 32
}

object Border {
  val dnnBorderZeros = 0x0
  val dnnBorderExtrapolation = 0x3
}

object Error {
  val E_SUCCESS = 0
  val E_INCORRECT_INPUT_PARAMETER = -1
  val E_UNEXPECTED_NULL_POINTER = -2
  val E_MEMORY_ERROR = -3
  val E_UNSUPPORTED_DIMENSION = -4
  val E_UNIMPLEMENTED = -127
}

object Algorithm {
  val dnnAlgorithmConvolutionGemm = 0 // GEMM based convolution
  val dnnAlgorithmConvolutionDirect = 1 // Direct convolution
  val dnnAlgorithmConvolutionFFT = 2 // FFT based convolution
  val dnnAlgorithmPoolingMax = 3 // Maximum pooling
  val dnnAlgorithmPoolingMin = 4 // Minimum pooling
  val dnnAlgorithmPoolingAvg = 5 // Average pooling
}

object MklDataSize {
  val FLOAT = 4
  val INT = 4
  val DOUBLE = 8
}
