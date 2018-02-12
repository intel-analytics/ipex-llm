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
import com.intel.analytics.bigdl.nn.abstractnn.{Initializable, TensorModule}
import com.intel.analytics.bigdl.optim.{L1L2Regularizer, L1Regularizer, L2Regularizer, Regularizer}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Shape, T, Table}
import com.intel.analytics.bigdl.nn.{InitMethodTag => Tag}

import scala.reflect.ClassTag
import scala.util.Try

/**
 * The `Linear` module applies a linear transformation to the input data,
 * i.e. `y = Wx + b`. The `input` given in `forward(input)` must be either
 * a vector (1D tensor) or matrix (2D tensor). If the input is a vector, it must
 * have the size of `inputSize`. If it is a matrix, then each row is assumed to be
 * an input sample of given batch (the number of rows means the batch size and
 * the number of columns should be equal to the `inputSize`).
 *
 * @param inputSize the size the each input sample
 * @param outputSize the size of the module output of each sample
 * @param wRegularizer: instance of [[Regularizer]]
 *                    (eg. L1 or L2 regularization), applied to the input weights matrices.
 * @param bRegularizer: instance of [[Regularizer]]
 *                    applied to the bias.
 */
@SerialVersionUID( 359656776803598943L)
class Linear[T: ClassTag](
  val inputSize: Int,
  val outputSize: Int,
  val withBias: Boolean = true,
  var wRegularizer: Regularizer[T] = null,
  var bRegularizer: Regularizer[T] = null,
  private val initWeight: Tensor[T] = null,
  private val initBias: Tensor[T] = null,
  private val initGradWeight: Tensor[T] = null,
  private val initGradBias: Tensor[T] = null
)(implicit ev: TensorNumeric[T]) extends TensorModule[T] with Initializable {
  val weight: Tensor[T] =
    if (initWeight != null) initWeight else Tensor[T](outputSize, inputSize)
  val bias: Tensor[T] =
    if (initBias != null) initBias else if (withBias) Tensor[T](outputSize) else null
  val addBuffer: Tensor[T] = Tensor[T]()

  val gradWeight: Tensor[T] =
    if (initGradWeight != null) initGradWeight else Tensor[T]()
  val gradBias: Tensor[T] =
    if (initGradBias != null) initGradBias else if (withBias) Tensor[T]() else null

  {
    val stdv = 1.0 / math.sqrt(weight.size(2))
    val wInit: InitializationMethod = RandomUniform(-stdv, stdv)
    val bInit: InitializationMethod = RandomUniform(-stdv, stdv)
    setInitMethod(wInit, bInit)
  }

  override def reset(): Unit = {
    if (initWeight == null) {
      weightInitMethod.init(weight, VariableFormat.OUT_IN)
    }
    if (initBias == null) {
      Option(bias).foreach(biasInitMethod.init(_, VariableFormat.ONE_D))
    }
    zeroGradParameters()
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    inputShape.copyAndUpdate(-1, outputSize)
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 1 || input.dim() == 2,
      "Linear: " + ErrorInfo.constrainInputAsVectorOrBatch +
        s"input dim ${input.dim()}")


    if (input.dim() == 1) {
      output.resize(Array(outputSize))
      if (withBias) output.copy(bias) else output.zero()
      output.addmv(ev.fromType[Int](1), weight, input)
    }
    else if (input.dim() == 2) {
      val nFrame = input.size(1)
      val nElement = output.nElement
      val t = Array(nFrame, weight.size(1))
      output.resize(t)
      if (output.nElement() != nElement) {
        output.zero()
      }

      if (addBuffer.nElement() != nFrame) {
        addBuffer.resize(Array(nFrame)).fill(ev.one)
      }

      output.addmm(ev.zero, output, ev.one, input, weight.t)
      if (withBias) output.addr(ev.one, addBuffer, bias)
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.dim() == 1 || input.dim() == 2,
      "Linear: " + ErrorInfo.constrainInputAsVectorOrBatch +
        s"input dim ${input.dim()}")

    val nElement = gradInput.nElement()
    gradInput.resizeAs(input)
    if (nElement != gradInput.nElement()) {
      gradInput.zero()
    }

    if (input.dim() == 1) {
      gradInput.addmv(ev.fromType[Int](0), ev.fromType[Int](1), weight.t(), gradOutput)
    } else if (input.dim() == 2) {
      gradInput.addmm(ev.fromType[Int](0), ev.fromType[Int](1), gradOutput, weight)
    }
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    require(input.dim() == 1 || input.dim() == 2,
      "Linear: " + ErrorInfo.constrainInputAsVectorOrBatch +
        s"input dim ${input.dim()}")

    gradWeight.resize(outputSize, inputSize)
    if (withBias) {
      gradBias.resize(outputSize)
    }

    if (input.dim() == 1) {
      if (scaleW != 0) {
        gradWeight.addr(ev.fromType[Double](scaleW), gradOutput, input)
      }

      if (withBias && scaleB != 0) {
        gradBias.add(ev.fromType[Double](scaleB), gradOutput)
      }
    }
    else if (input.dim() == 2) {
      if (scaleW != 0) {
        gradWeight.addmm(ev.fromType[Double](scaleW), gradOutput.t, input)
      }

      if (withBias && scaleB != 0) {
        gradBias.addmv(ev.fromType[Double](scaleB), gradOutput.t, addBuffer)
      }
    }

    if (null != wRegularizer && scaleW != 0) {
      wRegularizer.accRegularization(weight, gradWeight, scaleW)
    }
    if (null != bRegularizer && scaleB != 0) {
      bRegularizer.accRegularization(bias, gradBias, scaleB)
    }
  }

  override def clearState() : this.type = {
    super.clearState()
    addBuffer.set()
    this
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    if (null == bias) {
      (Array(this.weight), Array(this.gradWeight))
    } else {
      (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
    }
  }

  override def equals(obj: Any): Boolean = {

    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[Linear[T]]) {
      return false
    }
    val other = obj.asInstanceOf[Linear[T]]
    if (this.eq(other)) {
      return true
    }

    gradWeight == other.gradWeight &&
      gradBias == other.gradBias &&
      weight == other.weight &&
      bias == other.bias
  }

  override def hashCode() : Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + gradWeight.hashCode()
    hash = hash * seed + gradBias.hashCode()
    hash = hash * seed + weight.hashCode()
    hash = hash * seed + bias.hashCode()

    hash
  }

  override def toString(): String = {
    s"${getPrintName}($inputSize -> $outputSize)"
  }
}

object Linear extends quantized.Quantizable {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputSize: Int,
    outputSize: Int,
    withBias: Boolean = true,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    initWeight: Tensor[T] = null,
    initBias: Tensor[T] = null,
    initGradWeight: Tensor[T] = null,
    initGradBias: Tensor[T] = null
  )(implicit ev: TensorNumeric[T]) : Linear[T] = {
    new Linear[T](inputSize, outputSize,
      withBias, wRegularizer, bRegularizer, initWeight, initBias, initGradWeight, initGradBias)
  }

  /**
   * A Humanized Builder for Linear
   *
   * @param inputSize the size the each input sample
   * @param outputSize the size of the module output of each sample
   * @param withBias whether including bias(default: true)
   * @param l1Reg `lambda` of L1 Regularization(default: 0.0)
   * @param l2Reg `lambda` of L2 Regularization(default: 0.0)
   * @param initWeightMethod initialization method of weights/bias(default: RandomNormal)
   */
  def build[T: ClassTag](
    inputSize: Int,
    outputSize: Int,
    withBias: Boolean = true,
    l1Reg: Double = 0.0,
    l2Reg: Double = 0.0,
    initWeightMethod: Tag.Value = Tag.RandomNormal
  )(implicit ev: TensorNumeric[T]): Linear[T] = {
    require(inputSize > 0, s"found InputSize($inputSize) <= 0 in LinearBuilder!")
    require(outputSize > 0, s"found OutputSize($outputSize) <= 0 in LinearBuilder!")
    val state = T("inputSize" -> inputSize, "outputSize" -> outputSize, "withBias" -> withBias)
    val regBuilder = T("l1" -> l1Reg, "l2" -> l2Reg)
    state.update("wRegularizer", regBuilder)
    state.update("bRegularizer", regBuilder)
    state.update("initWeight", T("name" -> initWeightMethod))
    if (withBias) {
      state.update("initBias", T("name" -> initWeightMethod))
    }

    val ele = linearLikeBuilder[T](state)
    Linear[T](ele._1, ele._2, ele._3, ele._4, ele._5, ele._6, ele._7, ele._8, ele._9)
  }

  def buildRegularizer[T: ClassTag](
    l1: Double,
    l2: Double
  )(implicit ev: TensorNumeric[T]): Regularizer[T] = {
    if (l1 + l2 == 0.0) {
      null
    } else if (l1 * l2 > 0.0) {
      new L1L2Regularizer[T](l1, l2)
    } else if (l2 > 0.0) {
      new L2Regularizer[T](l2)
    } else {
      new L1Regularizer[T](l1)
    }
  }

  /**
   * Build linear-like members with [[com.intel.analytics.bigdl.utils.Table]].
   *
   * ======Schema of the Table======
   * {{{
   *  inputSize: Int
   *  outputSize: Int
   *  withBias: Boolean(default: true)
   *  wRegularizer, bRegularizer: Regularizer[T]/Table(default: null)
   *    params of Table:
   *      l1: Double(default: 0.0)
   *      l2: Double(default: 0.0)
   *  initWeight, initBias, initGradWeight, initGradBias: Tensor[T]/Table(default: null)
   *    params of Table:
   *      same params as buildInitTensor(Table)
   * }}}
   *
   * @param param a Table contains parameters
   */
  private[bigdl] def linearLikeBuilder[T: ClassTag](param: Table
  )(implicit ev: TensorNumeric[T]) = {
    val inputSize = param.get[Int]("inputSize") match {
      case Some(e) => e
      case _ => throw new IllegalArgumentException(
        "value(type: Int) for key(inputSize) is required!")
    }
    val outputSize = param.get[Int]("outputSize") match {
      case Some(e) => e
      case _ => throw new IllegalArgumentException(
        "value(type: Int) for key(outputSize) is required!")
    }
    val withBias = param.getOrElse("withBias", true)

    val getRegularizer = (key: String) => param.get[Any](key) match {
      case None => null
      case Some(value) => value match {
        case reg: Regularizer[T] => reg
        case state: Table =>
          buildRegularizer[T](state.getOrElse("l1", 0.0), state.getOrElse("l2", 0.0))
        case _ => throw new IllegalArgumentException(s"wrong value type for key($key)!")
      }
    }
    val (wReg, bReg) = getRegularizer("wRegularizer") -> getRegularizer("bRegularizer")

    val getTensor = (key: String) => param.get[Any](key) match {
      case None => null
      case Some(value) => value match {
        case tensor: Tensor[T] => tensor
        case state: Table =>
          // weights tensor of Linear Layer has a shape like: [outputSize, inputSize]
          if (!state.contains("shape")) key match {
            case k if k.endsWith("Weight") =>
              state.update("shape", Array(outputSize, inputSize))
            case k if k.endsWith("Bias") =>
              state.update("shape", Array(outputSize))
          }
          buildInitTensor[T](state)
        case _ => throw new IllegalArgumentException(s"wrong value type for key($key)!")
      }
    }

    val (initW, initB) = getTensor("initWeight") ->
      (if (withBias) getTensor("initBias") else null)
    val (initGradW, initGradB) = getTensor("initGradWeight") ->
      (if (withBias) getTensor("initGradBias") else null)

    (inputSize, outputSize, withBias, wReg, bReg, initW, initB, initGradW, initGradB)
  }

  /**
   * Init a `Tensor` with a [[com.intel.analytics.bigdl.nn.InitializationMethod]]
   * which is configured by a [[com.intel.analytics.bigdl.utils.Table]].
   *
   * ======Schema of the Table======
   * {{{
   *  shape: Array[Int], shape of Tensor to be initialized
   *  name: String, name of InitializationMethod(case-insensitive)
   *      InitializationMethods:
   *          RandomUniform
   *          RandomNormal
   *              mean: Double(default: 0)
   *              stdv: Double(default: 1/sqrt(size))
   *          Xavier
   *          Ones
   *          Zeros
   *          Const
   *              value: Double(no default)
   *          BilinearFiller
   *          MsraFiller
   *              varianceNormAverage: Boolean(default: true)
   * }}}
   *
   * @param param a Table contains parameters
   */
  def buildInitTensor[T: ClassTag](param: Table
  )(implicit ev: TensorNumeric[T]): Tensor[T] = {
    val shape = param.get[Array[Int]]("shape") match {
      case Some(s) => s
      case None => throw new IllegalArgumentException("missed param shape(type: Array[Int])!")
    }
    val tensor = Tensor[T](shape)

    require(param.contains("name"), "param(name) is missing!")

    val initMethod = Try(param.get[String]("name").get)
      .getOrElse(Try(param.get[Tag.Value]("name").get.toString).getOrElse(""))
      .toLowerCase

    initMethod match {
      case s if s.equals(Tag.RandomUniform.toString) =>
        RandomUniform.init(tensor)
      case s if s.equals(Tag.RandomNormal.toString) =>
        val mean = param.getOrElse("mean", 0.0)
        lazy val _stdv = 1.0 / math.sqrt(tensor.size().product.toDouble)
        val stdv = param.getOrElse("stdv", _stdv)
        RandomNormal(mean, stdv).init(tensor)
      case s if s.equals(Tag.Xavier.toString) =>
        Xavier.init(tensor, VariableFormat.Default)
      case s if s.equals(Tag.Ones.toString) =>
        Ones.init(tensor)
      case s if s.equals(Tag.Zeros.toString) =>
        Zeros.init(tensor)
      case s if s.equals(Tag.Const.toString) =>
        ConstInitMethod(param.get[Double]("value").get).init(tensor)
      case s if s.equals(Tag.BilinearFiller.toString) =>
        BilinearFiller.init(tensor)
      case s if s.equals(Tag.MsraFiller.toString) =>
        MsraFiller(param.getOrElse("varianceNormAverage", true)).init(tensor)
      case _ =>
        throw new IllegalArgumentException("Error parsing InitMethod name!")
    }

    tensor
  }

  override def quantize[T: ClassTag](module: Module[T])(
    implicit ev: TensorNumeric[T]): Module[T] = {
    val linear = module.asInstanceOf[Linear[T]]
    val quantizedLinear = quantized.Linear[T](linear.weight.size(2), linear.weight.size(1),
      initWeight = linear.weight, initBias = linear.bias)
    quantizedLinear.setName(linear.getName())
  }
}
