/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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

import com.intel.analytics.bigdl.nn.AbstractReLU
import com.intel.analytics.bigdl.mkl.MklDnnFloat
import com.intel.analytics.bigdl.nn.abstractnn.ModuleType._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.{FloatType, MklTensor, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.language.implicitConversions
import scala.reflect.ClassTag

@SerialVersionUID(9153872979086169351L)
class ReLU[T: ClassTag](ip: Boolean = false)
                       (implicit ev: TensorNumeric[T])
  extends AbstractReLU[T](ip) with MklModuleMethods {
  class ReLURef extends Ref[T] {}
  class ReLUPrimitive extends Primitive {}

  @transient
  var refs: ReLURef = null
  @transient
  var primitive: ReLUPrimitive = null
  val resources = new Array[Long](ResourceType.dnnResourceNumber)

  private[this] def initLayerAttributes(input: Tensor[T]): Unit = {
    if (refs == null) { refs = new ReLURef }
    if (primitive == null) { primitive = new ReLUPrimitive }

    savedSize = Some(input.size().clone())

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
      case FloatType =>
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
    refs.gradInput.createConversion(ioLayout, primitive.backward, ResourceType.dnnResourceDiffSrc)
    refs.gradOutput.createConversion(ioLayout, primitive.backward, ResourceType.dnnResourceDiffDst)

    setInit(true)
  }

  def releaseAll(): Unit = {
    if (refs != null && primitive != null) {
      refs.release()
      primitive.release()

      setInit(false)
    }
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (input.size().deep != savedSize.getOrElse(Array()).deep || ! isInited) {
      releaseAll()
      initLayerAttributes(input)
    }
    refs.input.set(input, ip)

    java.util.Arrays.fill(resources, 0)
    resources(ResourceType.dnnResourceSrc) = refs.input.getConvertedStorage()
    resources(ResourceType.dnnResourceDst) = refs.output.mklStorage()

    execute(resources, primitive.forward)

    if (this.nextModuleType == DNN) {
      this.output = refs.output
    } else {
      output.resizeAs(refs.output)
      refs.output.backToUsr(output)
    }

    if (this.isTraining()) {
      refs.input.setConverted(true)
    }

    this.output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    refs.gradOutput.set(gradOutput)
    refs.input.set(input, ip)

    java.util.Arrays.fill(resources, 0)
    resources(ResourceType.dnnResourceSrc) = refs.input.getConvertedStorage()
    resources(ResourceType.dnnResourceDiffDst) = refs.gradOutput.getConvertedStorage()
    resources(ResourceType.dnnResourceDiffSrc) = refs.gradInput.mklStorage()

    execute(resources, primitive.backward)

    if (this.prevModuleType == DNN) {
      this.gradInput = refs.gradInput
    } else {
      gradInput.resizeAs(refs.gradInput)
      refs.gradInput.backToUsr(gradInput)
    }

    if (this.isTraining()) {
      refs.input.setConverted(false)
    }

    this.gradInput
  }

  override def toString: String = {
    s"mkl.ReLU"
  }

  override def convertToMklDnn(prevModule: Option[AbstractModule[Activity, Activity, T]] = None)
  : (ModuleType, AbstractModule[Activity, Activity, T]) =
    super[MklModuleMethods].convertToMklDnn(prevModule)

  override def setNextModuleType(value: ModuleType): Unit =
    super[MklModuleMethods].setNextModuleType(value)

  override def setPrevModuleType(value: ModuleType): Unit =
    super[MklModuleMethods].setPrevModuleType(value)

  override def nextModuleType: ModuleType = super[MklModuleMethods].nextModuleType

  override def prevModuleType: ModuleType = super[MklModuleMethods].prevModuleType

  override def moduleType(): ModuleType = super[MklModuleMethods].moduleType()
}
