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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{FloatType, Tensor}
import com.intel.analytics.bigdl.utils.serializer._
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.utils.{Engine, ParameterSynchronizer, T, Table}
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}

import scala.reflect.ClassTag

/**
 * This layer implements Batch Normalization as described in the paper:
 * "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
 * by Sergey Ioffe, Christian Szegedy https://arxiv.org/abs/1502.03167
 *
 * This implementation is useful for inputs NOT coming from convolution layers.
 * For convolution layers, use nn.SpatialBatchNormalization.
 *
 * The operation implemented is:
 *             ( x - mean(x) )
 *     y =  -------------------- * gamma + beta
 *             standard-deviation(x)
 * where gamma and beta are learnable parameters.The learning of gamma and beta is optional.
 * @param nOutput output feature map number
 * @param eps avoid divide zero
 * @param momentum momentum for weight update
 * @param affine affine operation on output or not
 * @param ev numeric operator
 * @tparam T numeric type
 */
@SerialVersionUID(- 3181824540272906068L)
class BatchNormalization[T: ClassTag](
  val nOutput: Int, // output feature map number
  val eps: Double = 1e-5, // avoid divde zero
  val momentum: Double = 0.1, // momentum for weight update
  val affine: Boolean = true, // affine operation on output or not
  private val initWeight: Tensor[T] = null,
  private val initBias: Tensor[T] = null,
  private val initGradWeight: Tensor[T] = null,
  private val initGradBias: Tensor[T] = null
)(implicit ev: TensorNumeric[T]) extends TensorModule[T] with Initializable
  with MklInt8Convertible {

  require(nOutput > 0, "output feature map number must be greater than zero")

  private var parallism : Option[Int] = None


  /**
   * Set parameter sync parallisim number
   * @param parallism Concurrent sync threads number
   */
  def setParallism(parallism: Int): Unit = {
    this.parallism = Some(parallism)
  }

  def getParallism(): Option[Int] = this.parallism

  val meanKey: String = s"${this.getName}_mean"
  val stdKey: String = s"${this.getName}_std"
  val gmKey: String = s"${this.getName}_gm"
  val gxmKey: String = s"${this.getName}_gxm"

  val nDim = 2
  val channelDim = 2
  var runningMean = if (affine) Tensor[T](nOutput) else Tensor[T]()
  var runningVar = if (affine) Tensor[T](nOutput).fill(ev.one) else Tensor[T]()
  var saveMean = if (affine) Tensor[T](nOutput) else Tensor[T]()
  var saveStd = if (affine) Tensor[T](nOutput).fill(ev.zero) else Tensor[T]()

  val weight: Tensor[T] =
    if (initWeight != null) initWeight else if (affine) Tensor[T](nOutput) else null
  val bias: Tensor[T] =
    if (initBias != null) initBias else if (affine) Tensor[T](nOutput) else null

  val gradWeight: Tensor[T] =
    if (initGradWeight != null) initGradWeight else if (affine) Tensor[T](nOutput) else null
  val gradBias: Tensor[T] =
    if (initGradBias != null) initGradBias else if (affine) Tensor[T](nOutput) else null

  @transient
  // BatchNormalization has internal parameters (saveMean, saveStd)
  // that changes at every forward, so a standard gradcheck won't work with this module.
  // if you want to do a gradcheck, you will need to fix those variables, otherwise not fix.
  protected var needFix: Boolean = false

  {
    val wInit = RandomUniform(0, 1)
    val bInit = Zeros
    setInitMethod(wInit, bInit)
  }

  override def reset(): Unit = {
    if (null != weight && initWeight == null) {
      weightInitMethod.init(weight, VariableFormat.ONE_D)
    }

    if (null != bias && initBias == null) {
      biasInitMethod.init(bias, VariableFormat.ONE_D)
    }

    zeroGradParameters()
  }

  @inline
  // to fix internal parameters (saveMean, saveStd)
  def setInit(status: Boolean = true): this.type = {
    needFix = status
    this
  }

  @inline
  protected def checkInputDim(input: Tensor[T]): Unit = {
    require(input.dim() == nDim || (input.dim() == nDim - 1 && train == false),
      s"only mini-batch supported (${nDim}D tensor), got ${input.dim()}D tensor instead")
  }

  @inline
  protected def makeBatch(input: Tensor[T]): Tensor[T] = {
    if (input.dim() == nDim - 1 && train == false) {
      input.addSingletonDimension()
    } else {
      input
    }
  }

  @inline
  protected def initializeBuffer(channels: Int): Unit = {
    runningMean.resize(channels).zero
    runningVar.resize(channels).fill(ev.one)
  }

  protected val gMean = Tensor[T]()
  protected val gxMean = Tensor[T]()
  protected val _input = Tensor[T]()
  protected val _gradOutput = Tensor[T]()

  var globalMean: Array[T] = new Array[T](0)

  var globalStd: Array[T] = new Array[T](0)

  var globalGMean: Array[T] = new Array[T](0)

  var globalGxmMean: Array[T] = new Array[T](0)

  override def clearState(): this.type = {
    super.clearState()
    gMean.set()
    gxMean.set()
    this
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    if (affine) {
      (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
    } else {
      null
    }
  }

  override def getExtraParameter(): Array[Tensor[T]] = {
    Array(runningMean, runningVar)
  }

  override def getParametersTable(): Table = {
    if (affine) {
      T(getName() -> T("weight" -> weight, "bias" -> bias,
        "gradWeight" -> gradWeight, "gradBias" -> gradBias,
        "runningMean" -> runningMean, "runningVar" -> runningVar))
    } else {
      T(getName() -> T("runningMean" -> runningMean, "runningVar" -> runningVar))
    }
  }

  override def toString(): String = {
    s"nn.BatchNormalization($nOutput, $eps, $momentum, $affine)"
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[BatchNormalization[T]]

  override def equals(other: Any): Boolean = other match {
    case that: BatchNormalization[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        nDim == that.nDim &&
        runningMean == that.runningMean &&
        runningVar == that.runningVar &&
        weight == that.weight &&
        bias == that.bias &&
        nOutput == that.nOutput &&
        eps == that.eps &&
        momentum == that.momentum &&
        affine == that.affine
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), nDim, runningMean, runningVar, weight, bias,
      nOutput, eps, momentum, affine)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {

    val parallism = getParallism().getOrElse(1)

    val meanKeyWithId = s"${this.meanKey}_${this.getId}"
    val stdKeyWithId = s"${this.stdKey}_${this.getId}"
    val gmKeyWithId = s"${this.gmKey}_${this.getId}"
    val gxmKeyWithId = s"${this.gxmKey}_${this.getId}"

    val needSync = if (parallism != 1) {
      ParameterSynchronizer.register(meanKeyWithId, parallism)
      ParameterSynchronizer.register(stdKeyWithId, parallism)
      ParameterSynchronizer.register(gmKeyWithId, parallism)
      ParameterSynchronizer.register(gxmKeyWithId, parallism)
      true
    } else false

    checkInputDim(input)
    output.resizeAs(input)

    _input.set(input)
    makeBatch(_input)
    _input.addSingletonDimension(_input, 3)
    _input.addSingletonDimension(_input, 4)
    val nInput = _input.size(channelDim)

    if (runningMean.nElement == 0 || runningMean.nElement < nInput) {
      initializeBuffer(nInput)
    }

    saveMean.resizeAs(runningMean).zero
    saveStd.resizeAs(runningVar).fill(ev.zero)

    val nChannels = _input.size(2)

    if (globalMean.size < nChannels) {
      globalMean = new Array[T](nChannels)
    }

    if (globalStd.size < nChannels) {
      globalStd = new Array[T](nChannels)
    }

    if (train) {
      if (ev.getType() == FloatType) {
        SpatialBatchNormalization.updateOutputNCHWTrainFloat(
          _input.asInstanceOf[Tensor[Float]], output.asInstanceOf[Tensor[Float]],
          saveMean.asInstanceOf[Tensor[Float]], saveStd.asInstanceOf[Tensor[Float]],
          runningMean.asInstanceOf[Tensor[Float]], runningVar.asInstanceOf[Tensor[Float]],
          weight.asInstanceOf[Tensor[Float]], bias.asInstanceOf[Tensor[Float]],
          eps.toFloat, momentum.toFloat,
          globalMean = globalMean.asInstanceOf[Array[Float]],
          globalStd = globalStd.asInstanceOf[Array[Float]],
          meanKey = meanKeyWithId, stdKey = stdKeyWithId, needSync = needSync)
      } else {
        SpatialBatchNormalization.updateOutputNCHWTrainDouble(
          _input.asInstanceOf[Tensor[Double]], output.asInstanceOf[Tensor[Double]],
          saveMean.asInstanceOf[Tensor[Double]], saveStd.asInstanceOf[Tensor[Double]],
          runningMean.asInstanceOf[Tensor[Double]], runningVar.asInstanceOf[Tensor[Double]],
          weight.asInstanceOf[Tensor[Double]], bias.asInstanceOf[Tensor[Double]],
          eps, momentum,
          globalMean = globalMean.asInstanceOf[Array[Double]],
          globalStd = globalStd.asInstanceOf[Array[Double]],
          meanKey = meanKeyWithId, stdKey = stdKeyWithId, needSync = needSync)
      }
    } else {
      if (ev.getType() == FloatType) {
        SpatialBatchNormalization.updateOutputNCHWInferFloat(
          _input.asInstanceOf[Tensor[Float]], output.asInstanceOf[Tensor[Float]],
          runningMean.asInstanceOf[Tensor[Float]], runningVar.asInstanceOf[Tensor[Float]],
          weight.asInstanceOf[Tensor[Float]], bias.asInstanceOf[Tensor[Float]], eps.toFloat)
      } else {
        SpatialBatchNormalization.updateOutputNCHWInferDouble(
          _input.asInstanceOf[Tensor[Double]], output.asInstanceOf[Tensor[Double]],
          runningMean.asInstanceOf[Tensor[Double]], runningVar.asInstanceOf[Tensor[Double]],
          weight.asInstanceOf[Tensor[Double]], bias.asInstanceOf[Tensor[Double]], eps)
      }
    }

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val gmKeyWithId = s"${this.gmKey}_${this.getId}"
    val gxmKeyWithId = s"${this.gxmKey}_${this.getId}"
    val needSync = getParallism() != None && getParallism().get > 1
    _gradOutput.set(gradOutput)
    makeBatch(_gradOutput)
    _gradOutput.addSingletonDimension(_gradOutput, 3)
    _gradOutput.addSingletonDimension(_gradOutput, 4)
    gxMean.zero()
    gMean.zero()
    val nChannel = _gradOutput.size(2)
    if (globalGMean.size < nChannel) {
      globalGMean = new Array[T](nChannel)
    }
    if (globalGxmMean.size < nChannel) {
      globalGxmMean = new Array[T](nChannel)
    }
    if (train) {
      if (ev.getType() == FloatType) {
        SpatialBatchNormalization.updateGradInputNCHWTrainFloat(
          _input.asInstanceOf[Tensor[Float]], _gradOutput.asInstanceOf[Tensor[Float]],
          gradInput.asInstanceOf[Tensor[Float]], weight.asInstanceOf[Tensor[Float]],
          saveMean.asInstanceOf[Tensor[Float]], saveStd.asInstanceOf[Tensor[Float]],
          gMean.asInstanceOf[Tensor[Float]], gxMean.asInstanceOf[Tensor[Float]],
          globalGMean.asInstanceOf[Array[Float]], globalGxmMean.asInstanceOf[Array[Float]],
          gMeanKey = gmKeyWithId, gxMeanKey = gxmKeyWithId, needSync = needSync)
      } else {
        SpatialBatchNormalization.updateGradInputNCHWTrainDouble(
          _input.asInstanceOf[Tensor[Double]], _gradOutput.asInstanceOf[Tensor[Double]],
          gradInput.asInstanceOf[Tensor[Double]], weight.asInstanceOf[Tensor[Double]],
          saveMean.asInstanceOf[Tensor[Double]], saveStd.asInstanceOf[Tensor[Double]],
          gMean.asInstanceOf[Tensor[Double]], gxMean.asInstanceOf[Tensor[Double]],
          globalGMean.asInstanceOf[Array[Double]], globalGxmMean.asInstanceOf[Array[Double]],
          gMeanKey = gmKeyWithId, gxMeanKey = gxmKeyWithId, needSync = needSync)
      }
    } else {
      if (ev.getType() == FloatType) {
        SpatialBatchNormalization.updateGradInputNCHWInferFloat(
          _gradOutput.asInstanceOf[Tensor[Float]],
          gradInput.asInstanceOf[Tensor[Float]], weight.asInstanceOf[Tensor[Float]],
          bias.asInstanceOf[Tensor[Float]])
      } else {
        SpatialBatchNormalization.updateGradInputNCHWInferDouble(
          _gradOutput.asInstanceOf[Tensor[Double]],
          gradInput.asInstanceOf[Tensor[Double]], weight.asInstanceOf[Tensor[Double]],
          bias.asInstanceOf[Tensor[Double]])
      }
    }
    gradInput.squeeze(4)
    gradInput.squeeze(3)
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    if (weight == null || scaleW == 0) {
      return
    }

    if (ev.getType() == FloatType) {
      SpatialBatchNormalization.accGradientNCHWFloat(_gradOutput.asInstanceOf[Tensor[Float]],
        gradWeight.asInstanceOf[Tensor[Float]], gradBias.asInstanceOf[Tensor[Float]],
        _input.asInstanceOf[Tensor[Float]], saveMean.asInstanceOf[Tensor[Float]],
        saveStd.asInstanceOf[Tensor[Float]], scaleW.toFloat, scaleB.toFloat)
    } else {
      SpatialBatchNormalization.accGradientNCHWDouble(_gradOutput.asInstanceOf[Tensor[Double]],
        gradWeight.asInstanceOf[Tensor[Double]], gradBias.asInstanceOf[Tensor[Double]],
        _input.asInstanceOf[Tensor[Double]], saveMean.asInstanceOf[Tensor[Double]],
        saveStd.asInstanceOf[Tensor[Double]], scaleW, scaleB)
    }
  }
}

object BatchNormalization extends ModuleSerializable {
  def apply[@specialized(Float, Double) T: ClassTag](
    nOutput: Int,
    eps: Double = 1e-5,
    momentum: Double = 0.1,
    affine: Boolean = true,
    initWeight: Tensor[T] = null,
    initBias: Tensor[T] = null,
    initGradWeight: Tensor[T] = null,
    initGradBias: Tensor[T] = null)
    (implicit ev: TensorNumeric[T]): BatchNormalization[T] = {

    new BatchNormalization[T](
      nOutput, eps, momentum, affine, initWeight, initBias, initGradWeight, initGradBias)
  }
  def apply[@specialized(Float, Double) T: ClassTag](
    affine: Option[Int])(implicit ev: TensorNumeric[T]): BatchNormalization[T] = {
    new BatchNormalization[T](nOutput = affine.getOrElse(1), affine = affine.isDefined)
  }

  override def doLoadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {
    val attrMap = context.bigdlModule.getAttrMap
    val batchNorm = super.doLoadModule(context).asInstanceOf[BatchNormalization[T]]

    batchNorm.runningMean = DataConverter.
      getAttributeValue(context, attrMap.get("runningMean")).
      asInstanceOf[Tensor[T]]

    batchNorm.runningVar = DataConverter.
      getAttributeValue(context, attrMap.get("runningVar")).
      asInstanceOf[Tensor[T]]

    batchNorm.saveMean = DataConverter.
      getAttributeValue(context, attrMap.get("saveMean")).
      asInstanceOf[Tensor[T]]

    batchNorm.saveStd = DataConverter.
      getAttributeValue(context, attrMap.get("saveStd")).
      asInstanceOf[Tensor[T]]

    batchNorm
  }
  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
                                              batchNormBuilder : BigDLModule.Builder)
                                             (implicit ev: TensorNumeric[T]) : Unit = {

    super.doSerializeModule(context, batchNormBuilder)

    val batchNorm = context.moduleData.module.asInstanceOf[BatchNormalization[T]]

    val runningMeanBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, runningMeanBuilder,
      batchNorm.runningMean, ModuleSerializer.tensorType)
    batchNormBuilder.putAttr("runningMean", runningMeanBuilder.build)

    val runningVarBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, runningVarBuilder,
      batchNorm.runningVar, ModuleSerializer.tensorType)
    batchNormBuilder.putAttr("runningVar", runningVarBuilder.build)

    val saveMeanBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, saveMeanBuilder,
      batchNorm.saveMean, ModuleSerializer.tensorType)
    batchNormBuilder.putAttr("saveMean", saveMeanBuilder.build)

    val saveStdBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, saveStdBuilder,
      batchNorm.saveStd, ModuleSerializer.tensorType)
    batchNormBuilder.putAttr("saveStd", saveStdBuilder.build)
  }
}
