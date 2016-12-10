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

import com.intel.analytics.bigdl.nn.ModuleType._
import com.intel.analytics.bigdl.mkl.MklDnnFloat
import com.intel.analytics.bigdl.tensor.{MklTensor, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag
import scala.language.implicitConversions

class LRN[@specialized(Float, Double) T: ClassTag](
    val size: Int = 5,
    val alpha: Double = 1.0,
    val beta: Double = 0.75,
    val k: Double = 1.0)(implicit ev: TensorNumeric[T])
    extends MklModule[T] {

  class PoolRef extends Ref {
    val workspace = new MklTensor[T]()
  }
  class PoolPrimitive extends Primitive {}

  val refs = new PoolRef
  val primitive = new PoolPrimitive

  private[this] def initLayerAttributes(input: Tensor[T]): Unit = {
    // set dimension = 4 forcily. because it seems that only support dimension 4 in mkl dnn
    val dimension = 4

    val inputLayout = new MklLayout(4, Array(
      input.size(input.dim()), // width
      input.size(input.dim() - 1), // height
      if (input.dim() < 3) 1 else input.size(input.dim() - 2), // channels
      if (input.dim() < 4) 1 else input.size(input.dim() - 3) // number
    ))

    val outputLayout = new MklLayout(4, inputLayout.size)

    // layout ptr in JNI, it may be confused.
    val layout = if (input.isMklTensor()) {
      input.asInstanceOf[MklTensor[T]].layoutMkl
    } else {
      refs.input.createUsrLayout(dimension, inputLayout.size, inputLayout.strides)
      refs.input.layoutUsr
    }

    ev.getType() match {
      case "Float" =>
        this.primitive.forward = MklDnnFloat.lrnCreateForward(
          layout,
          size,
          alpha.toFloat,
          beta.toFloat,
          k.toFloat
        )
        require(this.primitive.forward != 0, "create convolution primitive failed.")

        this.primitive.backward = MklDnnFloat.lrnCreateBackward(
          layout,
          layout,
          size,
          alpha.toFloat,
          beta.toFloat,
          k.toFloat
        )
        require(this.primitive.backward != 0, "create convolution primitive failed.")
      case _ =>
        throw new UnsupportedOperationException(s"Only Float supported")
    }

    for (i <- List(refs.input, refs.gradInput, this.gradInput)) {
      i.resizeAs(input)
    }

    for (i <- List(refs.output, refs.gradOutput, this.output)) {
      i.resize(outputLayout.size.reverse.map(_.toInt), outputLayout.strides.reverse.map(_.toInt))
    }

    refs.input.createConversion(inputLayout, primitive.forward, ResourceType.dnnResourceSrc)
    refs.output.createConversion(outputLayout, primitive.forward, ResourceType.dnnResourceDst)
    // we may create a usr layout and a conversion between usr and mkl layout
    refs.workspace.createMklLayout(primitive.forward, ResourceType.dnnResourceWorkspace)

    refs.gradInput.createConversion(inputLayout, primitive.backward, ResourceType.dnnResourceDiffSrc)
    refs.gradOutput.createConversion(outputLayout, primitive.backward, ResourceType.dnnResourceDiffDst)

    isInited_=(true)
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (!isInited) {
      initLayerAttributes(input)
    }
    refs.input.set(input)

    ev.getType() match {
      case "Float" => MklDnnFloat.lrnForwardExecute(
        refs.input.getConvertedStorage().array().asInstanceOf[Array[Float]],
        refs.output.mklStorage.array().asInstanceOf[Array[Float]],
        refs.workspace.mklStorage.array().asInstanceOf[Array[Float]],
        primitive.forward
      )
      case _ =>
        throw new UnsupportedOperationException(s"Only Float supported")
    }

    if (this.nextModuleType() == DNN) {
      this.output = refs.output
    } else {
      refs.output.backToUsr(this.output.storage(), this.output.storageOffset())
    }

    this.output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    refs.gradOutput.set(gradOutput)
    refs.input.set(input)

    ev.getType() match {
      case "Float" => MklDnnFloat.lrnBackwardExecute(
        refs.input.getConvertedStorage().array().asInstanceOf[Array[Float]],
        refs.gradInput.mklStorage.array().asInstanceOf[Array[Float]],
        refs.gradOutput.getConvertedStorage().array().asInstanceOf[Array[Float]],
        refs.workspace.mklStorage.array().asInstanceOf[Array[Float]],
        primitive.backward
      )
    }

    if (this.prevModuleType() == DNN) {
      this.gradInput = refs.gradInput
    } else {
      refs.gradInput.backToUsr(this.gradInput.storage(), this.gradInput.storageOffset())
    }

    this.gradInput
  }

  override def equals(obj: Any): Boolean = {
    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[SpatialCrossMapLRN[T]]) { return false }
    val other = obj.asInstanceOf[SpatialCrossMapLRN[T]]
    if (this.eq(other)) { return true }

    size == other.size &&
      alpha == other.alpha && beta == other.beta && k == other.k
  }

  override def hashCode(): Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + size.hashCode()
    hash = hash * seed + alpha.hashCode()
    hash = hash * seed + beta.hashCode()
    hash = hash * seed + k.hashCode()

    hash
  }

  override def toString(): String = {
    s"mkl.SpatialCrossMapLRN($size, $alpha, $beta, $k)"
  }

}
