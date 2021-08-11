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
 * Applies a spatial division operation on a series of 2D inputs using kernel for
 * computing the weighted average in a neighborhood. The neighborhood is defined for
 * a local spatial region that is the size as kernel and across all features. For
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
 * @param threshold threshold
 * @param thresval threshhold value to replace with
 *                 if data is smaller than theshold
 */

@SerialVersionUID(6036047576084619110L)
class SpatialDivisiveNormalization[T: ClassTag](
  val nInputPlane: Int = 1,
  var kernel: Tensor[T] = null,
  val threshold: Double = 1e-4,
  val thresval: Double = 1e-4
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

  val padH = math.floor(kernel.size(1).toFloat/2).toInt
  val padW = if (kdim == 2) {
    math.floor(kernel.size(2).toFloat/2).toInt
  } else {
    padH
  }

  // create convolutional mean estimator
  private var meanestimator = new Sequential[T]()
  meanestimator.add(new SpatialZeroPadding(padW, padW, padH, padH))
  if (kdim == 2) {
    meanestimator.add(new SpatialConvolution(nInputPlane, 1, kernel.size(2), kernel.size(1)))
  } else {
    meanestimator.add(new SpatialConvolutionMap[T](
      SpatialConvolutionMap.oneToOne[T](nInputPlane), kernel.size(1), 1))
    meanestimator.add(new SpatialConvolution(nInputPlane, 1, 1, kernel.size(1)))
  }
  meanestimator.add(new Replicate(nInputPlane, 1, 3))

  // create convolutional std estimator
  private var stdestimator = new Sequential[T]()
  stdestimator.add(new Square())
  stdestimator.add(new SpatialZeroPadding(padW, padW, padH, padH))
  if (kdim == 2) {
    stdestimator.add(new SpatialConvolution(nInputPlane, 1, kernel.size(2), kernel.size(1)))
  } else {
    stdestimator.add(new SpatialConvolutionMap[T](
      SpatialConvolutionMap.oneToOne[T](nInputPlane), kernel.size(1), 1))
    stdestimator.add(new SpatialConvolution(nInputPlane, 1, 1, kernel.size(1)))
  }
  stdestimator.add(new Replicate(nInputPlane, 1, 3))
  stdestimator.add(new Sqrt())

  // set kernel(parameters._1(0)) and bias(parameters._1(1))
  if (kdim == 2) {
    kernel.div(ev.times(kernel.sum(), ev.fromType[Int](nInputPlane)))
    for (i <- 1 to nInputPlane) {
      meanestimator.modules(1).parameters()._1(0)(1)(1)(i).copy(kernel)
      stdestimator.modules(2).parameters()._1(0)(1)(1)(i).copy(kernel)
    }
    meanestimator.modules(1).parameters()._1(1).zero()
    stdestimator.modules(2).parameters()._1(1).zero()
  } else {
    kernel.div(ev.times(kernel.sum(), ev.sqrt(ev.fromType[Int](nInputPlane))))
    for (i <- 1 to nInputPlane) {
      meanestimator.modules(1).parameters()._1(0)(i).copy(kernel)
      meanestimator.modules(2).parameters()._1(0)(1)(1)(i).copy(kernel)
      stdestimator.modules(2).parameters()._1(0)(i).copy(kernel)
      stdestimator.modules(3).parameters()._1(0)(1)(1)(i).copy(kernel)
    }
    meanestimator.modules(1).parameters()._1(1).zero()
    meanestimator.modules(2).parameters()._1(1).zero()
    stdestimator.modules(2).parameters()._1(1).zero()
    stdestimator.modules(3).parameters()._1(1).zero()
  }

  // other operation
  private var normalizer = new CDivTable()
  private var divider = new CDivTable()
  private var thresholder = new Threshold(threshold, thresval)

  // coefficient array, to adjust side effects
  private var coef: Tensor[T] = Tensor(1, 1, 1)

  private val ones: Tensor[T] = Tensor[T]()
  private var adjustedstds: Tensor[T] = _
  private var thresholdedstds: Tensor[T] = _
  private var localstds: Tensor[T] = _

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    localstds = stdestimator.forward(input).toTensor[T]

    // compute side coefficients
    val dim = input.dim()
    if (localstds.dim() != coef.dim() || (input.size(dim) != coef.size(dim)) ||
      (input.size(dim-1) != coef.size(dim-1)) ) {
      if (dim == 4) {
        // batch mode
        ones.resizeAs(input(1)).fill(ev.fromType[Int](1))
        val _coef = meanestimator.forward(ones).toTensor[T]
        coef = coef.resizeAs(_coef).copy(_coef).view(Array(1) ++ _coef.size()).expandAs(localstds)
      } else {
        ones.resizeAs(input).fill(ev.fromType[Int](1))
        coef = meanestimator.forward(ones).toTensor[T]
      }
    }

    // normalize std dev
    adjustedstds = divider.forward(T(localstds, coef)).asInstanceOf[Tensor[T]]
    thresholdedstds = thresholder.forward(adjustedstds)
    output = normalizer.forward(T(input, thresholdedstds)).asInstanceOf[Tensor[T]]

    output
  }


  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    // resize grad
    gradInput.resizeAs(input).zero()

    // backprop through all modules
    val gradnorm = normalizer.updateGradInput(T(input, thresholdedstds), gradOutput)
    val gradadj = thresholder.updateGradInput(adjustedstds, gradnorm(2))
    val graddiv = divider.updateGradInput(T(localstds, coef), gradadj)
    gradInput.add(stdestimator.updateGradInput(input, graddiv(1)).toTensor[T])
    gradInput.add(gradnorm[Tensor[T]](1))
    gradInput
  }

  override def toString(): String = {
    s"${getPrintName}($nInputPlane, kernelTensor, $threshold, $thresval)"
  }

  override def canEqual(other: Any): Boolean = {
    other.isInstanceOf[SpatialDivisiveNormalization[T]]
  }

  override def equals(other: Any): Boolean = other match {
    case that: SpatialDivisiveNormalization[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        kdim == that.kdim &&
        padH == that.padH &&
        padW == that.padW &&
        meanestimator == that.meanestimator &&
        stdestimator == that.stdestimator &&
        normalizer == that.normalizer &&
        divider == that.divider &&
        thresholder == that.thresholder &&
        nInputPlane == that.nInputPlane &&
        kernel == that.kernel &&
        threshold == that.threshold &&
        thresval == that.thresval
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), kdim, padH, padW, meanestimator, stdestimator,
      normalizer, divider, thresholder, nInputPlane, kernel, threshold, thresval)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }


  override def clearState() : this.type = {
    super.clearState()
    meanestimator.clearState()
    stdestimator.clearState()
    normalizer.clearState()
    divider.clearState()
    coef = Tensor(1, 1, 1)
    ones.set()
    adjustedstds = null
    thresholdedstds = null
    localstds = null
    this
  }
}

object SpatialDivisiveNormalization extends ModuleSerializable {
  def apply[@specialized(Float, Double) T: ClassTag](
      nInputPlane: Int = 1,
      kernel: Tensor[T] = null,
      threshold: Double = 1e-4,
      thresval: Double = 1e-4)(
      implicit ev: TensorNumeric[T]) : SpatialDivisiveNormalization[T] = {
    new SpatialDivisiveNormalization[T](nInputPlane, kernel, threshold, thresval)
  }

  override def doLoadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {

    val spatialDivisiveNormModule = super.doLoadModule(context).
      asInstanceOf[SpatialDivisiveNormalization[T]]

    val attrMap = context.bigdlModule.getAttrMap

    spatialDivisiveNormModule.meanestimator = DataConverter.
      getAttributeValue(context, attrMap.get("meanestimator")).
      asInstanceOf[Sequential[T]]

    spatialDivisiveNormModule.stdestimator = DataConverter.
      getAttributeValue(context, attrMap.get("stdestimator")).
      asInstanceOf[Sequential[T]]

    spatialDivisiveNormModule.normalizer = DataConverter.
      getAttributeValue(context, attrMap.get("normalizer")).
      asInstanceOf[CDivTable[T]]

    spatialDivisiveNormModule.divider = DataConverter.
      getAttributeValue(context, attrMap.get("divider")).
      asInstanceOf[CDivTable[T]]

    spatialDivisiveNormModule.thresholder = DataConverter.
      getAttributeValue(context, attrMap.get("thresholder")).
      asInstanceOf[Threshold[T]]

    spatialDivisiveNormModule
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
                                              spatialDivisiveNormBuilder : BigDLModule.Builder)
                                           (implicit ev: TensorNumeric[T]) : Unit = {

    val spatialDivisiveNormModule = context.moduleData
      .module.asInstanceOf[SpatialDivisiveNormalization[T]]
    super.doSerializeModule(context, spatialDivisiveNormBuilder)

    val meanestimatorBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, meanestimatorBuilder,
      spatialDivisiveNormModule.meanestimator, ModuleSerializer.tensorModuleType)
    spatialDivisiveNormBuilder.putAttr("meanestimator", meanestimatorBuilder.build)

    val stdestimatorBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, stdestimatorBuilder,
      spatialDivisiveNormModule.stdestimator, ModuleSerializer.tensorModuleType)
    spatialDivisiveNormBuilder.putAttr("stdestimator", stdestimatorBuilder.build)

    val normalizerBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, normalizerBuilder,
      spatialDivisiveNormModule.normalizer, ModuleSerializer.tensorModuleType)
    spatialDivisiveNormBuilder.putAttr("normalizer", normalizerBuilder.build)

    val dividerBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, dividerBuilder,
      spatialDivisiveNormModule.divider, ModuleSerializer.tensorModuleType)
    spatialDivisiveNormBuilder.putAttr("divider", dividerBuilder.build)

    val thresholderBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, thresholderBuilder,
      spatialDivisiveNormModule.thresholder, ModuleSerializer.tensorModuleType)
    spatialDivisiveNormBuilder.putAttr("thresholder", thresholderBuilder.build)

  }
}
