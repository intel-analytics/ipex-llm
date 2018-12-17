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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer._
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}

import scala.reflect.ClassTag

/**
 * Subtractive + divisive contrast normalization.
 *
 * @param nInputPlane
 * @param kernel
 * @param threshold
 * @param thresval
 */
@SerialVersionUID(- 5339890039498187188L)
class SpatialContrastiveNormalization[T: ClassTag](
  val nInputPlane: Int = 1,
  var kernel: Tensor[T] = null,
  val threshold: Double = 1e-4,
  val thresval: Double = 1e-4
)(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  if (null == kernel) kernel = Tensor.ones[T](9, 9)

  private val kdim = kernel.nDimension()
  require(kdim == 1 || kdim == 2, "averaging kernel must be 2D or 1D" +
    s"averaging kernel dimension ${kdim}")
  require(kernel.size(1) % 2 != 0, "averaging kernel must have ODD dimensions" +
    s"averaging kernel dimension ${kernel.size(1)}")
  if (kdim == 2) {
    require(kernel.size(2) % 2 != 0, "averaging kernel must have ODD dimensions" +
      s"averaging kernel dimension ${kernel.size(2)}")
  }

  // instantiate sub+div normalization
  private var normalizer = new Sequential[T]()
  normalizer.add(new SpatialSubtractiveNormalization(nInputPlane, kernel))
  normalizer.add(new SpatialDivisiveNormalization(nInputPlane, kernel, threshold, thresval))

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output = normalizer.forward(input).toTensor[T]
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput = normalizer.backward(input, gradOutput).toTensor[T]
    gradInput
  }

  override def toString(): String = {
    s"${getPrintName}($nInputPlane, kernelTensor, $threshold, $thresval)"
  }

  override def canEqual(other: Any): Boolean = {
    other.isInstanceOf[SpatialContrastiveNormalization[T]]
  }

  override def equals(other: Any): Boolean = other match {
    case that: SpatialContrastiveNormalization[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        kdim == that.kdim &&
        normalizer == that.normalizer &&
        nInputPlane == that.nInputPlane &&
        kernel == that.kernel &&
        threshold == that.threshold &&
        thresval == that.thresval
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), kdim, normalizer,
      nInputPlane, kernel, threshold, thresval)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }

  override def clearState() : this.type = {
    super.clearState()
    normalizer.clearState()
    this
  }
}

object SpatialContrastiveNormalization extends ModuleSerializable {
  def apply[@specialized(Float, Double) T: ClassTag](
      nInputPlane: Int = 1,
      kernel: Tensor[T] = null,
      threshold: Double = 1e-4,
      thresval: Double = 1e-4)(
      implicit ev: TensorNumeric[T]) : SpatialContrastiveNormalization[T] = {
    new SpatialContrastiveNormalization[T](nInputPlane, kernel, threshold, thresval)
  }

  override def doLoadModule[T: ClassTag](context : DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {

    val spatialContrastiveNormaModule = super.doLoadModule(context).
      asInstanceOf[SpatialContrastiveNormalization[T]]

    val attrMap = context.bigdlModule.getAttrMap

    spatialContrastiveNormaModule.normalizer = DataConverter.
      getAttributeValue(context, attrMap.get("normalizer")).
      asInstanceOf[Sequential[T]]

    spatialContrastiveNormaModule
  }

  override def doSerializeModule[T: ClassTag](context : SerializeContext[T],
                                            contrastiveNormBuilder : BigDLModule.Builder)
                                           (implicit ev: TensorNumeric[T]) : Unit = {
    super.doSerializeModule(context, contrastiveNormBuilder)
    val spatialContrastiveNormaModule = context.moduleData.module.
      asInstanceOf[SpatialContrastiveNormalization[T]]

    val normalizerBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, normalizerBuilder,
      spatialContrastiveNormaModule.normalizer,
      ModuleSerializer.tensorModuleType)
    contrastiveNormBuilder.putAttr("normalizer", normalizerBuilder.build)

  }
}
