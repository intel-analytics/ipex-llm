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

package com.intel.analytics.bigdl.utils.intermediate

import com.intel.analytics.bigdl.nn.MklInt8Convertible
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat, TensorModule}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.{Tensor, TensorNumericMath}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

sealed class IROperator[T: ClassTag] extends Serializable {
  val numerics: TensorNumeric[T] = getNumerics(scala.reflect.classTag[T])
  final def getNumerics[T](tag: ClassTag[T]) : TensorNumeric[T] = {
    tag match {
      case ClassTag.Float => TensorNumeric.NumericFloat.asInstanceOf[TensorNumeric[T]]
      case ClassTag.Double => TensorNumeric.NumericDouble.asInstanceOf[TensorNumeric[T]]
      case _ => throw new IllegalArgumentException(s"not supported class tag: ${tag}")
    }
  }
  def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array(scala.reflect.classTag[T]), Array(numerics))
  }
  def name: String = this.getClass.getSimpleName
}

case class IRSpatialMaxPooling[T: ClassTag](
            kW: Int, kH: Int,
            dW: Int = 1, dH: Int = 1,
            padW: Int = 0, padH: Int = 0,
            format: DataFormat = DataFormat.NCHW, ceilMode: Boolean = false) extends IROperator[T]

case class IRSpatialAveragePooling[T: ClassTag](
            kW: Int, kH: Int,
            dW: Int = 1, dH: Int = 1,
            padW: Int = 0, padH: Int = 0,
            globalPooling: Boolean = false,
            ceilMode: Boolean = false, countIncludePad: Boolean = true,
            divide: Boolean = true, format: DataFormat = DataFormat.NCHW) extends IROperator[T]

case class IRSpatialConvolution[T: ClassTag](
            nInputPlane: Int, nOutputPlane: Int,
            kernelW: Int, kernelH: Int,
            strideW: Int = 1, strideH: Int = 1,
            padW: Int = 0, padH: Int = 0,
            nGroup: Int = 1, propagateBack: Boolean = true,
            wRegularizer: Regularizer[T] = null, bRegularizer: Regularizer[T] = null,
            initWeight: Tensor[T] = null, initBias: Tensor[T] = null,
            initGradWeight: Tensor[T] = null, initGradBias: Tensor[T] = null,
            withBias: Boolean = true, format: DataFormat = DataFormat.NCHW) extends IROperator[T]

case class IRSpatialShareConvolution[T: ClassTag](
            nInputPlane: Int, nOutputPlane: Int,
            kernelW: Int, kernelH: Int,
            strideW: Int = 1, strideH: Int = 1,
            padW: Int = 0, padH: Int = 0,
            nGroup: Int = 1, propagateBack: Boolean = true,
            wRegularizer: Regularizer[T] = null, bRegularizer: Regularizer[T] = null,
            initWeight: Tensor[T] = null, initBias: Tensor[T] = null,
            initGradWeight: Tensor[T] = null, initGradBias: Tensor[T] = null,
            withBias: Boolean = true, format: DataFormat = DataFormat.NCHW) extends IROperator[T]

case class IRSpatialBatchNormalization[T: ClassTag](
            nOutput: Int, eps: Double = 1e-5, momentum: Double = 0.1,
            affine: Boolean = true,
            initWeight: Tensor[T] = null, initBias: Tensor[T] = null,
            initGradWeight: Tensor[T] = null, initGradBias: Tensor[T] = null,
            dataFormat: DataFormat = DataFormat.NCHW,
            runningMean: Tensor[T] = null, runningVar: Tensor[T] = null) extends IROperator[T]

case class IRIdentity[T: ClassTag]() extends IROperator[T]

case class IRReLU[T: ClassTag](ip: Boolean = false) extends IROperator[T]

case class IRLinear[T: ClassTag](
            inputSize: Int,
            outputSize: Int,
            withBias: Boolean = true,
            wRegularizer: Regularizer[T] = null,
            bRegularizer: Regularizer[T] = null,
            initWeight: Tensor[T] = null,
            initBias: Tensor[T] = null,
            initGradWeight: Tensor[T] = null,
            initGradBias: Tensor[T] = null) extends IROperator[T]

case class IRSpatialCrossMapLRN[T: ClassTag](
            size: Int = 5,
            alpha: Double = 1.0,
            beta: Double = 0.75,
            k: Double = 1.0,
            format: DataFormat = DataFormat.NCHW) extends IROperator[T]

case class IRSoftMax[T: ClassTag]() extends IROperator[T]

case class IRSelectTable[T: ClassTag](dimension: Int) extends IROperator[T]

case class IRCAddTable[T: ClassTag, D: ClassTag](inplace: Boolean = false) extends IROperator[T] {
  private val ev = getNumerics(scala.reflect.classTag[T])
  private val ev2 = getNumerics(scala.reflect.classTag[D])

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}

case class IRJoinTable[T: ClassTag](dimension: Int,
                                    nInputDims: Int = 0) extends IROperator[T]

case class IRConcatTable[T: ClassTag]() extends IROperator[T]

case class IRInput[T: ClassTag]() extends IROperator[T]

/**
 * if blas module has no corresponding IROperator,
 * then we can use IRGeneralModule to wrap this layer to IROperator
 * @param model
 */
case class IRGeneralModule[T: ClassTag](
             model: AbstractModule[Activity, Activity, T]) extends IROperator[T]

private[bigdl] class IRElement[T: ClassTag](
  val name: String,
  val op: IROperator[T],
  private var weights: Tensor[T] = null,
  private var gradWeights: Tensor[T] = null) extends Serializable with MklInt8Convertible {

  /**
   * set weight and bias
   */
  def setWeights(weightsAndBias: Tensor[T]) : Unit = {
    weights = weightsAndBias
  }

  /**
   * set gradWeight and gradbias
   */
  def setGradWeights(gradWeightsAndGradBias: Tensor[T]) : Unit = {
    gradWeights = gradWeightsAndGradBias
  }

  def getParameters(): (Tensor[T], Tensor[T]) = (weights, gradWeights)

  def getName() : String = this.name

  def getOp() : IROperator[T] = this.op
}

object IRElement {
  /**
   * create IRElement
   * @param name element name
   * @param op element operation, like IRSpatialMaxPooling, IRBlasModule, etc.
   * @param weights weights & bias for IRElement
   * @param gradWeights gradWeight & gradbias for IRElement
   * @tparam T
   * @return
   */
  def apply[T: ClassTag](name: String, op: IROperator[T],
                         weights: Tensor[T] = null, gradWeights: Tensor[T] = null): IRElement[T] =
    new IRElement[T](name, op, weights, gradWeights)
}
