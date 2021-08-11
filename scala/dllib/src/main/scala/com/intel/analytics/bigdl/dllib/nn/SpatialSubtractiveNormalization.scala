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
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer._
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}

import scala.reflect.ClassTag

/**
 * Applies a spatial subtraction operation on a series of 2D inputs using kernel for
 * computing the weighted average in a neighborhood. The neighborhood is defined for
 * a local spatial region that is the size as kernel and across all features. For a
 * an input image, since there is only one feature, the region is only spatial. For
 * an RGB image, the weighted average is taken over RGB channels and a spatial region.
 *
 * If the kernel is 1D, then it will be used for constructing and separable 2D kernel.
 * The operations will be much more efficient in this case.
 *
 * The kernel is generally chosen as a gaussian when it is believed that the correlation
 * of two pixel locations decrease with increasing distance. On the feature dimension,
 * a uniform average is used since the weighting across features is not known.
 *
 * @param nInputPlane number of input plane, default is 1.
 * @param kernel kernel tensor, default is a 9 x 9 tensor.
 */
@SerialVersionUID(2522324984775526595L)
class SpatialSubtractiveNormalization[T: ClassTag](
  val nInputPlane: Int = 1,
  var kernel: Tensor[T] = null
)(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  if (kernel == null) kernel = Tensor.ones[T](9, 9)

  private val kdim = kernel.nDimension()
  require(kdim == 1 || kdim == 2, "averaging kernel must be 2D or 1D" +
    s"averaging kernel dimension $kdim")
  require(kernel.size(1) % 2 != 0, "averaging kernel must have ODD dimensions" +
    s"averaging kernel dimension ${kernel.size(1)}")
  if (kdim == 2) {
    require(kernel.size(2) % 2 != 0, "averaging kernel must have ODD dimensions" +
      s"averaging kernel dimension ${kernel.size(2)}")
  }

  kernel.div(ev.times(kernel.sum(), ev.fromType[Int](nInputPlane)))

  private val padH = math.floor(kernel.size(1).toFloat/2).toInt
  private val padW = if (kdim == 2) {
    math.floor(kernel.size(2).toFloat/2).toInt
  } else {
    padH
  }

  // create convolutional mean extractor
  private var meanestimator = new Sequential[T]()
  meanestimator.add(new SpatialZeroPadding(padW, padW, padH, padH))
  if (kdim == 2) {
    meanestimator.add(new SpatialConvolution(nInputPlane, 1, kernel.size(2), kernel.size(1)))
  } else {
    meanestimator.add(new SpatialConvolutionMap(
      SpatialConvolutionMap.oneToOne(nInputPlane), kernel.size(1), 1))
    meanestimator.add(new SpatialConvolution(nInputPlane, 1, 1, kernel.size(1)))
  }
  meanestimator.add(new Replicate(nInputPlane, 1, 3))

  // set kernel(parameters._1(0)) and bias(parameters._1(1))
  if (kdim == 2) {
    for (i <- 1 to nInputPlane) {
      meanestimator.modules(1).parameters()._1(0)(1)(1)(i).copy(kernel)
    }
    meanestimator.modules(1).parameters()._1(1).zero()
  } else {
    for (i <- 1 to nInputPlane) {
      meanestimator.modules(1).parameters()._1(0)(i).copy(kernel)
      meanestimator.modules(2).parameters()._1(0)(1)(1)(i).copy(kernel)
    }
    meanestimator.modules(1).parameters()._1(1).zero()
    meanestimator.modules(2).parameters()._1(1).zero()
  }

  // other operation
  private var subtractor = new CSubTable()
  private var divider = new CDivTable()

  // coefficient array, to adjust side effects
  private var coef = Tensor(1, 1, 1)

  private val ones: Tensor[T] = Tensor[T]()
  private var adjustedsums: Tensor[T] = _
  private var localsums: Tensor[T] = _

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val dim = input.dim()
    if (input.dim() + 1 != coef.dim() || (input.size(dim) != coef.size(dim)) ||
      (input.size(dim-1) != coef.size(dim-1))) {

      if (dim == 4) {
        // batch mode
        ones.resizeAs(input(1)).fill(ev.fromType[Int](1))
        val _coef = meanestimator.forward(ones).toTensor[T]
        val size = Array(input.size(1)) ++ _coef.size()
        coef = coef.resizeAs(_coef).copy(_coef).view(Array(1) ++ _coef.size()).expand(size)
      } else {
        ones.resizeAs(input).fill(ev.fromType[Int](1))
        val _coef = meanestimator.forward(ones).toTensor[T]
        coef.resizeAs(_coef).copy(_coef)
      }
    }

    // compute mean
    localsums = meanestimator.forward(input).toTensor[T]
    adjustedsums = divider.forward(T(localsums, coef)).asInstanceOf[Tensor[T]]
    output = subtractor.forward(T(input, adjustedsums)).asInstanceOf[Tensor[T]]

    output
  }


  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    // resize grad
    gradInput.resizeAs(input).zero()

    // backprop through all modules
    val gradsub = subtractor.updateGradInput(T(input, adjustedsums), gradOutput)
    val graddiv = divider.updateGradInput(T(localsums, coef), gradsub(2))
    val size = meanestimator.updateGradInput(input, graddiv(1)).toTensor[T].size()
    gradInput.add(meanestimator.updateGradInput(input, graddiv(1)).toTensor[T])
    gradInput.add(gradsub[Tensor[T]](1))

    gradInput
  }

  override def toString(): String = {
    s"${getPrintName}($nInputPlane, kernelTensor)"
  }

  override def canEqual(other: Any): Boolean = {
    other.isInstanceOf[SpatialSubtractiveNormalization[T]]
  }

  override def equals(other: Any): Boolean = other match {
    case that: SpatialSubtractiveNormalization[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        kdim == that.kdim &&
        padH == that.padH &&
        padW == that.padW &&
        meanestimator == that.meanestimator &&
        subtractor == that.subtractor &&
        divider == that.divider &&
        nInputPlane == that.nInputPlane &&
        kernel == that.kernel
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(),
      kdim, padH, padW, meanestimator, subtractor, divider, nInputPlane, kernel)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }

  override def clearState() : this.type = {
    super.clearState()
    meanestimator.clearState()
    subtractor.clearState()
    divider.clearState()
    coef = Tensor(1, 1, 1)
    ones.set()
    adjustedsums = null
    localsums = null
    this
  }
}

object SpatialSubtractiveNormalization extends ModuleSerializable {
  def apply[@specialized(Float, Double) T: ClassTag](
      nInputPlane: Int = 1,
      kernel: Tensor[T] = null)(
      implicit ev: TensorNumeric[T]) : SpatialSubtractiveNormalization[T] = {
    new SpatialSubtractiveNormalization[T](nInputPlane, kernel)
  }

  override def doLoadModule[T: ClassTag](context : DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {

    val spatialSubtractiveNormModule = super.doLoadModule(context).
      asInstanceOf[SpatialSubtractiveNormalization[T]]

    val attrMap = context.bigdlModule.getAttrMap

    spatialSubtractiveNormModule.meanestimator = DataConverter.
      getAttributeValue(context, attrMap.get("meanestimator")).
      asInstanceOf[Sequential[T]]

    spatialSubtractiveNormModule.subtractor = DataConverter.
      getAttributeValue(context, attrMap.get("subtractor")).
      asInstanceOf[CSubTable[T]]

    spatialSubtractiveNormModule.divider = DataConverter.
      getAttributeValue(context, attrMap.get("divider")).
      asInstanceOf[CDivTable[T]]

    spatialSubtractiveNormModule
  }

  override def doSerializeModule[T: ClassTag](contetxt: SerializeContext[T],
                                            subtractiveNormBuilder : BigDLModule.Builder)
                                           (implicit ev: TensorNumeric[T]) : Unit = {
    super.doSerializeModule(contetxt, subtractiveNormBuilder)
    val spatialSubtractiveNormaModule = contetxt.moduleData.module.
      asInstanceOf[SpatialSubtractiveNormalization[T]]

    val meanestimatorBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(contetxt, meanestimatorBuilder,
      spatialSubtractiveNormaModule.meanestimator,
      ModuleSerializer.tensorModuleType)
    subtractiveNormBuilder.putAttr("meanestimator", meanestimatorBuilder.build)


    val thresholderBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(contetxt, thresholderBuilder,
      spatialSubtractiveNormaModule.subtractor, ModuleSerializer.tensorModuleType)
    subtractiveNormBuilder.putAttr("subtractor", thresholderBuilder.build)

    val dividerBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(contetxt, dividerBuilder,
      spatialSubtractiveNormaModule.divider, ModuleSerializer.tensorModuleType)
    subtractiveNormBuilder.putAttr("divider", dividerBuilder.build)

  }
}
