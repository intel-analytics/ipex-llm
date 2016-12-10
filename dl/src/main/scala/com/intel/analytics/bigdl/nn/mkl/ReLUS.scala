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

import com.intel.analytics.bigdl.nn.ModuleType
import com.intel.analytics.bigdl.mkl.MklDnnFloat
import com.intel.analytics.bigdl.tensor.{MklTensor, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.language.implicitConversions
import scala.reflect.ClassTag

class ReLUS[@specialized(Float, Double) T: ClassTag](ip: Boolean = false)(
  implicit ev: TensorNumeric[T]) extends MklModule[T] {

  class ReLURef extends Ref {}
  class ReLUPrimitive extends Primitive {}

  val refs = new ReLURef
  val primitive = new ReLUPrimitive

  private[this] def initLayerAttributes(input: Tensor[T]): Unit = {
    val dimension = input.dim()
    // input and output layout
    val ioLayout = new MklLayout(dimension, Array(
      input.size(input.dim()), // width
      input.size(input.dim() - 1), // height
      if (input.dim() < 3) 1 else input.size(input.dim() - 2), // channels
      if (input.dim() < 4) 1 else input.size(input.dim() - 3) // number
    ))

    // layout ptr in JNI, it may be confused.
    val layout = if (input.isMklTensor()) {
      input.asInstanceOf[MklTensor[T]].layoutMkl
    } else {
      refs.input.createUsrLayout(dimension, ioLayout.size, ioLayout.strides)
      refs.input.layoutUsr
    }

    // nagtive slope set to 0 defaultly.
    val nagtiveSlope = ev.fromType(0.0)

    ev.getType() match {
      case "Float" =>
        this.primitive.forward = MklDnnFloat.reluCreateForward(layout,
          ev.toType[Float](nagtiveSlope))
        require(this.primitive.forward != 0, "create convolution primitive failed.")

        // TODO why are there 2 layout? Is it the same, or helpful for performance?
        this.primitive.backward = MklDnnFloat.reluCreateBackward(layout,
          layout, ev.toType[Float](nagtiveSlope))
        require(this.primitive.backward != 0, "create convolution primitive failed.")
      case _ =>
        throw new UnsupportedOperationException(s"Only Float supported")
    }

    // the size of all below are same
    for (tensor <- List(refs.input, refs.gradInput, refs.output, refs.gradOutput,
      this.output, this.gradInput)) {
      tensor.resizeAs(input)
    }

    refs.input.createConversion(ioLayout, primitive.forward, ResourceType.dnnResourceSrc)
    refs.output.createConversion(ioLayout, primitive.forward, ResourceType.dnnResourceDst)
    refs.gradInput.createConversion(ioLayout, primitive.forward, ResourceType.dnnResourceSrc)
    refs.gradOutput.createConversion(ioLayout, primitive.forward, ResourceType.dnnResourceDst)

    isInited_=(true)
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (!isInited) {
      initLayerAttributes(input)
    }
    refs.input.set(input, ip)
//    refs.output.mklStorage.fill(ev.fromType(0), 1, refs.output.mklStorage().size)

    ev.getType() match {
      case "Float" => MklDnnFloat.reluForwardExecute(
        refs.input.getConvertedStorage().array().asInstanceOf[Array[Float]],
        refs.output.mklStorage.array().asInstanceOf[Array[Float]],
        primitive.forward
      )
      case _ =>
        throw new UnsupportedOperationException(s"Only Float supported")
    }

    if (this.nextModuleType() == ModuleType.DNN) {
      this.output = refs.output
    } else {
      refs.output.backToUsr(this.output.storage(), this.output.storageOffset())
    }

    this.output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    refs.gradOutput.set(gradOutput)
    refs.input.set(input, ip)

    ev.getType() match {
      case "Float" => MklDnnFloat.reluBackwardExecute(
        refs.input.getConvertedStorage().array().asInstanceOf[Array[Float]],
        refs.gradInput.mklStorage.array().asInstanceOf[Array[Float]],
        refs.gradOutput.getConvertedStorage().array().asInstanceOf[Array[Float]],
        primitive.backward
      )
    }

    if (this.prevModuleType() == ModuleType.DNN) {
      this.gradInput = refs.gradInput
    } else {
      refs.gradInput.backToUsr(this.gradInput.storage(), this.gradInput.storageOffset())
    }

    this.gradInput
  }

  override def toString(): String = {
    s"mkl.ReLU"
  }
}
