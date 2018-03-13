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

import com.intel.analytics.bigdl.nn.abstractnn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer._
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}

import scala.reflect.ClassTag

/**
 * S-shaped Rectified Linear Unit.
 *  It follows:
 *  `f(x) = t^r + a^r(x - t^r) for x >= t^r`,
 *  `f(x) = x for t^r > x > t^l`,
 *  `f(x) = t^l + a^l(x - t^l) for x <= t^l`.
 *
 * [Deep Learning with S-shaped Rectified Linear Activation Units](http://arxiv.org/abs/1512.07030)
 *
 * @param shape shape for tleft, aleft, tright, aright.
 *              E.g. for a 4-D input, the shape is the last 3-D
 * @param sharedAxes the axes along which to share learnable parameters
 *                    for the activation function.
 *                    For example, if the incoming feature maps are from a 2D convolution
 *                    with output shape `(batch, height, width, channels)`,
 *                    and you wish to share parameters across space
 *                    so that each filter only has one set of parameters,
 *                    set `shared_axes=[1, 2]`.
 */

@SerialVersionUID(7173457290010080259L)
class SReLU[T: ClassTag](val shape: Array[Int], val sharedAxes: Array[Int] = null)(
  implicit ev: TensorNumeric[T]) extends TensorModule[T]
    with Initializable {
  import SReLU._
  val weightsLen = 4
  val weights: Array[Tensor[T]] = Array.fill[Tensor[T]](4)(Tensor[T]())
  val gradWeights: Array[Tensor[T]] = Array.fill[Tensor[T]](4)(Tensor[T]())

  val weightsInit: Array[InitializationMethod] = Array(Zeros, Xavier, Xavier, Ones)

  // this attribute for computing the offset in weight because of sharedAxes
  private var indexes: Array[Int] = null

  init(shape).reset()

  private def init(shape: Array[Int]): this.type = {
    if (sharedAxes != null) {
      var i = 0
      while (i < sharedAxes.length) {
        shape(sharedAxes(i) - 1) = 1
        i += 1
      }
    }

    val variableFormat = shape.length match {
      case 2 => VariableFormat.IN_OUT
      case 4 => VariableFormat.OUT_IN_KW_KH
      case 5 => VariableFormat.OUT_IN_KT_KH_KW
      case _ => VariableFormat.Default
    }

    var i = 0
    while (i < weightsLen) {
      weights(i).resize(shape)
      weightsInit(i).init(weights(i), variableFormat)

      gradWeights(i).resize(shape)
      gradWeights(i).resizeAs(weights(i)).zero()

      i += 1
    }

    // ensure the the right part is always to the right of the left
    weights(tRight).abs().add(weights(tLeft))
    this
  }

  override def reset(): Unit = {
    for ((initMethod, weight) <- weightsInit.zip(weights)) {
      initMethod.init(weight)
    }
    zeroGradParameters()
  }

  private def getIndex(indexes: Array[Int], stride: Array[Int], ndim: Int, offset: Int): Unit = {
    var i = 0
    var tmp = offset
    while (i < ndim) {
      indexes(i) = tmp / stride(i) + 1 // 1 based
      tmp = tmp % stride(i)
      i += 1
    }

    // set back the shared axes
    if (sharedAxes != null) {
      i = 0
      while (i < sharedAxes.length) {
        indexes(sharedAxes(i) - 1) = 1
        i += 1
      }
    }
  }

  private def setValue(w: Array[Tensor[T]], i: Int, t: Int, v: T): Unit = {
    w(t).storage().array()(w(t).storageOffset() - 1 + i) = ev.plus(
      w(t).storage().array()(w(t).storageOffset() - 1 + i),
      v)
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.isContiguous(), s"the input of SReLU must be contiguous")
    // ensure the the right part is always to the right of the left
    weights(tRight).abs().add(weights(tLeft))
    output.resizeAs(input)

    // temp buf for indexes
    if (indexes == null) {
      indexes = new Array[Int](weights(tRight).nDimension())
    }

    var batch = 0
    while (batch < input.size(1)) {
      val sliceInput = input.select(1, batch + 1)
      val sliceOutput = output.select(1, batch + 1)

      val xArr = sliceInput.storage().array()
      val yArr = sliceOutput.storage().array()
      val yOffset = sliceOutput.storageOffset() - 1
      val xOffset = sliceInput.storageOffset() - 1

      var i = 0
      while (i < sliceInput.nElement()) {
        getIndex(indexes, sliceInput.stride(), sliceInput.nDimension(), i)

        val tr = weights(tRight).apply(indexes)
        val ar = weights(aRight).apply(indexes)
        val tl = weights(tLeft).apply(indexes)
        val al = weights(aLeft).apply(indexes)

        val x = xArr(xOffset + i)

        yArr(yOffset + i) = if (ev.isGreaterEq(x, tr)) {
          // right: x_i >= t_i^r
          ev.plus(tr, ev.times(ar, ev.minus(x, tr)))
        } else if (ev.isGreaterEq(tl, x)) {
          // left: x_i <= t_i^l
          ev.plus(tl, ev.times(al, ev.minus(x, tl)))
        } else {
          // else x_i = x_i
          x
        }

        i += 1
      }

      batch += 1
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.isContiguous(), s"the input of SReLU must be contiguous")
    require(gradOutput.isContiguous(), s"the gradOutput of SReLU must be contiguous")
    gradInput.resizeAs(input)

    var batch = 0
    while (batch < gradInput.size(1)) {
      val sliceInput = input.select(1, batch + 1)
      val sliceGradInput = gradInput.select(1, batch + 1)
      val sliceGradOutput = gradOutput.select(1, batch + 1)

      val xArr = sliceInput.storage().array()
      var xOffset = sliceInput.storageOffset() - 1

      val yArr = sliceGradInput.storage().array()
      var yOffset = sliceGradInput.storageOffset() - 1

      val zArr = sliceGradOutput.storage().array()
      var zOffset = sliceGradOutput.storageOffset() - 1

      var i = 0

      while (i < sliceGradInput.nElement()) {
        getIndex(indexes, sliceInput.stride(), sliceInput.nDimension(), i)

        val tr = weights(tRight).apply(indexes)
        val ar = weights(aRight).apply(indexes)
        val tl = weights(tLeft).apply(indexes)
        val al = weights(aLeft).apply(indexes)
        val x = xArr(xOffset + i)

        val t = if (ev.isGreaterEq(x, tr)) {
          ev.times(ar, zArr(zOffset + i))
        } else if (ev.isGreaterEq(tl, x)) {
          ev.times(al, zArr(zOffset + i))
        } else {
          zArr(zOffset + i)
        }
        yArr(yOffset + i) = ev.plus(yArr(yOffset + i), t)
        i += 1
      }

      batch += 1
    }

    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    var batch = 0
    while (batch < gradInput.size(1)) {
      val sliceInput = input.select(1, batch + 1)
      val sliceGradOutput = gradOutput.select(1, batch + 1)

      val xArr = sliceInput.storage().array()
      val xOffset = sliceInput.storageOffset() - 1

      val zArr = sliceGradOutput.storage().array()
      val zOffset = sliceGradOutput.storageOffset() - 1

      var i = 0
      while (i < sliceInput.nElement()) {
        getIndex(indexes, sliceInput.stride(), sliceInput.nDimension(), i)

        // weight offset
        var wOffset = 0
        var j = 0
        while (j < indexes.length) {
          // because indexes is 1 based, so we should minus 1 here
          wOffset += (indexes(j) - 1) * gradWeights(tLeft).stride(j + 1)
          j += 1
        }

        val tr = weights(tRight).apply(indexes)
        val ar = weights(aRight).apply(indexes)
        val tl = weights(tLeft).apply(indexes)
        val al = weights(aLeft).apply(indexes)
        val x = xArr(xOffset + i)

        if (ev.isGreaterEq(x, tr)) {
          setValue(gradWeights, wOffset, tRight, ev.times(ev.minus(ev.fromType(1), ar),
            zArr(zOffset + i)))
          setValue(gradWeights, wOffset, aRight, ev.times(ev.minus(x, tr),
            zArr(zOffset + i)))
        } else {
          setValue(gradWeights, wOffset, tRight, ev.fromType(0))
          setValue(gradWeights, wOffset, aRight, ev.fromType(0))
        }

        if (ev.isGreaterEq(tl, x)) {
          setValue(gradWeights, wOffset, tLeft, ev.times(ev.minus(ev.fromType(1), al),
            zArr(zOffset + i)))
          setValue(gradWeights, wOffset, aLeft, ev.times(ev.minus(xArr(xOffset + i), tl),
            zArr(zOffset + i)))
        } else {
          setValue(gradWeights, wOffset, tLeft, ev.fromType(0))
          setValue(gradWeights, wOffset, aLeft, ev.fromType(0))
        }

        i += 1
      }

      batch += 1
    }
  }

  override def getParametersTable(): Table = {
    T(getName() -> T(
      "tLeft" -> weights(tLeft),
      "aLeft" -> weights(aLeft),
      "tRight" -> weights(tRight),
      "aRight" -> weights(aRight)))
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (weights, gradWeights)
  }

  override def setInitMethod(initMethods: Array[InitializationMethod]): this.type = {
    for (i <- Array(tLeft, aLeft, tRight, aRight)) {
      if (initMethods(i) != null) {
        weightsInit(i) = initMethods(i)
      }
    }
    reset()
    this
  }

  override def setInitMethod(weightInitMethod: InitializationMethod = null,
    biasInitMethod: InitializationMethod = null): this.type = {
    throw new UnsupportedOperationException(
      s"SReLU should call setInitMethod(initMethods: Array[InitializationMethod])")
  }
}


object SReLU extends ModuleSerializable {
  def apply[T: ClassTag](shape: Array[Int], shareAxes: Array[Int] = null)(
    implicit ev: TensorNumeric[T]): SReLU[T] = {
    new SReLU[T](shape, shareAxes)
  }

  val (tLeft, aLeft, tRight, aRight) = (0, 1, 2, 3)

  override def doLoadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {
    val attrMap = context.bigdlModule.getAttrMap
    val srelu = super.doLoadModule(context).asInstanceOf[SReLU[T]]

    srelu.weights(tLeft) = DataConverter.
      getAttributeValue(context, attrMap.get("tLeft")).
      asInstanceOf[Tensor[T]]

    srelu.weights(aLeft) = DataConverter.
      getAttributeValue(context, attrMap.get("aLeft")).
      asInstanceOf[Tensor[T]]

    srelu.weights(tRight) = DataConverter.
      getAttributeValue(context, attrMap.get("tRight")).
      asInstanceOf[Tensor[T]]

    srelu.weights(aRight) = DataConverter.
      getAttributeValue(context, attrMap.get("aRight")).
      asInstanceOf[Tensor[T]]


    srelu
  }
  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
    sreluBuilder : BigDLModule.Builder)
    (implicit ev: TensorNumeric[T]) : Unit = {

    super.doSerializeModule(context, sreluBuilder)

    val srelu = context.moduleData.module.asInstanceOf[SReLU[T]]

    val runningMeanBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, runningMeanBuilder,
      srelu.weights(tLeft), ModuleSerializer.tensorType)
    sreluBuilder.putAttr("tLeft", runningMeanBuilder.build)

    val runningVarBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, runningVarBuilder,
      srelu.weights(aLeft), ModuleSerializer.tensorType)
    sreluBuilder.putAttr("aLeft", runningVarBuilder.build)

    val saveMeanBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, saveMeanBuilder,
      srelu.weights(tRight), ModuleSerializer.tensorType)
    sreluBuilder.putAttr("tRight", saveMeanBuilder.build)

    val saveStdBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, saveStdBuilder,
      srelu.weights(aRight), ModuleSerializer.tensorType)
    sreluBuilder.putAttr("aRight", saveStdBuilder.build)
  }
}

