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
import com.intel.analytics.bigdl.utils.{T, Table}
import serialization.Bigdl.{AttrValue, BigDLModule}

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
 * @param shared_axes the axes along which to share learnable parameters
 *                    for the activation function.
 *                    For example, if the incoming feature maps are from a 2D convolution
 *                    with output shape `(batch, height, width, channels)`,
 *                    and you wish to share parameters across space
 *                    so that each filter only has one set of parameters,
 *                    set `shared_axes=[1, 2]`.
 */

@SerialVersionUID(7173457290010080259L)
class SReLU[T: ClassTag](shared_axes: Array[Int] = null)(
  implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  import SReLU._
  val weightsLen = 4
  val weights: Array[Tensor[T]] = Array.fill[Tensor[T]](4)(Tensor[T]())
  val gradWeights: Array[Tensor[T]] = Array.fill[Tensor[T]](4)(Tensor[T]())

  val weightsInit: Array[InitializationMethod] = Array(Zeros, Xavier, Xavier, Ones)

  private def init(input: Tensor[T]): Unit = {
    val shape = input.size().slice(1, input.size().length)
    if (shared_axes != null) {
      var i = 1
      while (i < shape.length) {
        if (shared_axes.contains(i)) {
          shape(i) = 1
        }
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
  }

  private def getValue(w: Array[Tensor[T]], i: Int, t: Int): T = {
    w(t).storage().array()(w(t).storageOffset() - 1 + i)
  }

  private def setValue(w: Array[Tensor[T]], i: Int, t: Int, v: T): Unit = {
    w(t).storage().array()(w(t).storageOffset() - 1 + i) = ev.plus(
      w(t).storage().array()(w(t).storageOffset() - 1 + i),
      v)
  }


  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input)

    // the weight's size depends on the input
    if (weights.exists(_.isEmpty)) {
      init(input)
    }

    var batch = 0
    while (batch < input.size(1)) {
      val sliceInput = input.select(1, batch + 1)
      val sliceOutput = output.select(1, batch + 1)

      val xArr = sliceInput.storage().array()
      val yArr = sliceOutput.storage().array()
      var yOffset = sliceOutput.storageOffset() - 1
      var xOffset = sliceInput.storageOffset() - 1

      // if share axes array is not null, do some groups
      val groups = sliceInput.nElement() / weights(tLeft).nElement()

      var g = 0
      while (g < groups) {
        var i = 0
        val len = weights(tLeft).nElement()

        xOffset += (i * len)
        yOffset += (i * len)

        while (i < len) {
          val tr = getValue(weights, i, tRight)
          val ar = getValue(weights, i, aRight)
          val tl = getValue(weights, i, tLeft)
          val al = getValue(weights, i, aLeft)
          val x = xArr(xOffset + i)

          yArr(yOffset + i) = if (ev.isGreaterEq(x, getValue(weights, i, tRight))) {
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
        g += 1
      }

      batch += 1
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
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

      // if share axes array is not null, do some groups
      val groups = sliceInput.nElement() / weights(tLeft).nElement()

      var g = 0
      while (g < groups) {
        var i = 0
        val len = weights(tLeft).nElement()

        xOffset += (i * len)
        yOffset += (i * len)
        zOffset += (i * len)

        while (i < len) {
          val tr = getValue(weights, i, tRight)
          val ar = getValue(weights, i, aRight)
          val tl = getValue(weights, i, tLeft)
          val al = getValue(weights, i, aLeft)
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
        g += 1
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

      val x = sliceInput.storage().array()
      val xOffset = sliceInput.storageOffset() - 1

      val z = sliceGradOutput.storage().array()
      val zOffset = sliceGradOutput.storageOffset() - 1

      var i = 0
      while (i < sliceInput.nElement()) {

        if (ev.isGreaterEq(x(xOffset + i), getValue(weights, i, tRight))) {
          setValue(gradWeights, i, tRight, ev.times(ev.minus(ev.fromType(1),
            getValue(weights, i, aRight)),
            z(zOffset + i)))
          setValue(gradWeights, i, aRight, ev.times(ev.minus(x(xOffset + i),
            getValue(weights, i, tRight)),
            z(zOffset + i)))
        } else {
          setValue(gradWeights, i, tRight, ev.fromType(0))
          setValue(gradWeights, i, aRight, ev.fromType(0))
        }

        if (ev.isGreaterEq(getValue(weights, i, tLeft), x(xOffset + i))) {
          setValue(gradWeights, i, tLeft, ev.times(ev.minus(ev.fromType(1),
            getValue(weights, i, aLeft)),
            z(zOffset + i)))
          setValue(gradWeights, i, aLeft, ev.times(ev.minus(x(xOffset + i),
            getValue(weights, i, tLeft)),
            z(zOffset + i)))
        } else {
          setValue(gradWeights, i, tLeft, ev.fromType(0))
          setValue(gradWeights, i, aLeft, ev.fromType(0))
        }

        i += 1
      }

      batch += 1
    }
  }

  override def setWeightsBias(newWeights: Array[Tensor[T]]): this.type = {
    // SReLU will split the weights from a tensor
    if (!newWeights.isEmpty) {
      var i = 0
      while (i < weightsLen) {
        val weight = newWeights(i)
        weights(i).resizeAs(weight).set(weight)
        gradWeights(i) = Tensor[T]().resizeAs(weight)

        i += 1
      }

      // ensure the the right part is always to the right of the left
      weights(tRight).abs().add(weights(tLeft))
    }

    this
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

  def setInitMethod(
    tLeftInit: InitializationMethod = null,
    aLeftInit: InitializationMethod = null,
    tRightInit: InitializationMethod = null,
    aRightInit: InitializationMethod = null): this.type = {
    val inits = Array(tLeftInit, aLeftInit, tRightInit, aRightInit)

    for (i <- Array(tLeft, aLeft, tRight, aRight)) {
      if (inits(i) != null) {
        weightsInit(i) = inits(i)
      }
    }

    this
  }
}


object SReLU extends ModuleSerializable {
  def apply[T: ClassTag](share_axes: Array[Int] = null)(implicit ev: TensorNumeric[T])
  : SReLU[T] = {
    new SReLU[T](share_axes)
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

