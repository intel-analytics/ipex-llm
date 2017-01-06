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

package com.intel.analytics.bigdl.nn.dnn

import com.intel.analytics.bigdl.mkl.MklDnnFloat
import com.intel.analytics.bigdl.nn.abstractnn.ModuleType._
import com.intel.analytics.bigdl.tensor.{FloatType, MklTensor, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag
import scala.language.implicitConversions

class SpatialCrossMapLRN[T: ClassTag](val size: Int = 5,
                                      val alpha: Double = 1.0,
                                      val beta: Double = 0.75,
                                      val k: Double = 1.0)(implicit ev: TensorNumeric[T])
  extends MklModule[T] {

  class LRNRef extends Ref {
    val workspace = new MklTensor[T]()
  }
  class LRNPrimitive extends Primitive {}

  val refs = new LRNRef
  val primitive = new LRNPrimitive
  val resources = new Array[Long](ResourceType.dnnResourceNumber)

  private[this] def initLayerAttributes(input: Tensor[T]): Unit = {
    val dimension = 4

    val inputLayout = new MklLayout(4, Utils.getSize(input, dimension))

    // layout ptr in JNI. It's maybe a little confused.
    val layout = if (input.isMklTensor()) {
      input.asInstanceOf[MklTensor[T]].layoutMkl
    } else {
      refs.input.createUsrLayout(dimension, inputLayout.size, inputLayout.strides)
      refs.input.layoutUsr
    }

    ev.getType() match {
      case FloatType =>
        this.primitive.forward = MklDnnFloat.lrnCreateForward(
          layout,
          size.toLong,
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

    for (i <- List(refs.input, refs.gradInput)) {
      i.resizeAs(input)
    }

    for (i <- List(refs.output, refs.gradOutput)) {
      i.resize(Utils.reverseAndToInt(inputLayout.size),
        Utils.reverseAndToInt(inputLayout.strides))
    }

    if (nextModuleType != DNN) {
      this.output.resizeAs(refs.output)
    }

    if (prevModuleType != DNN) {
      this.gradInput.resizeAs(input)
    }

    refs.input.createConversion(inputLayout, primitive.forward, ResourceType.dnnResourceSrc)
    refs.output.createConversion(inputLayout, primitive.forward, ResourceType.dnnResourceDst)
    refs.workspace.createMklLayout(primitive.forward, ResourceType.dnnResourceWorkspace)

    refs.gradInput.createConversion(inputLayout, primitive.backward,
      ResourceType.dnnResourceDiffSrc)
    refs.gradOutput.createConversion(inputLayout, primitive.backward,
      ResourceType.dnnResourceDiffDst)

    setInit(true)
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (!isInited) {
      initLayerAttributes(input)
    }
    refs.input.set(input)

    java.util.Arrays.fill(resources, 0)
    resources(ResourceType.dnnResourceSrc) = refs.input.getConvertedStorage()
    resources(ResourceType.dnnResourceWorkspace) = refs.workspace.mklStorage()
    resources(ResourceType.dnnResourceDst) = refs.output.mklStorage()

    execute(resources, primitive.forward)

    if (this.nextModuleType == DNN) {
      this.output = refs.output
    } else {
      refs.output.backToUsr(output)
    }

    this.output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    refs.gradOutput.set(gradOutput)
    refs.input.set(input)

    java.util.Arrays.fill(resources, 0)
    resources(ResourceType.dnnResourceSrc) = refs.input.getConvertedStorage()
    resources(ResourceType.dnnResourceDiffDst) = refs.gradOutput.getConvertedStorage()
    resources(ResourceType.dnnResourceDiffSrc) = refs.gradInput.mklStorage()
    resources(ResourceType.dnnResourceWorkspace) = refs.workspace.mklStorage()

    execute(resources, primitive.backward)

    if (this.prevModuleType == DNN) {
      this.gradInput = refs.gradInput
    } else {
      refs.gradInput.backToUsr(gradInput)
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

  override def toString: String = {
    s"mkl.SpatialCrossMapLRN($size, $alpha, $beta, $k)"
  }

}
