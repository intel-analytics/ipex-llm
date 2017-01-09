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
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{FloatType, MklTensor, Tensor}

import scala.reflect.ClassTag

class Pool[T: ClassTag](val kW: Int,
                        val kH: Int,
                        val dW: Int,
                        val dH: Int,
                        val padW: Int = 0,
                        val padH: Int = 0,
                        val algorithm: Int)(implicit ev: TensorNumeric[T])
  extends TensorModule[T] with MklModuleMethods {
  class PoolRef extends Ref {
    val workspace = new MklTensor[T]()
  }
  class PoolPrimitive extends Primitive {}

  val refs = new PoolRef
  val primitive = new PoolPrimitive
  val resources = new Array[Long](ResourceType.dnnResourceNumber)

  def ceil(): Pool[T] = {
    this
  }

  private[this] def initLayerAttributes(input: Tensor[T]): Unit = {
    val strides = Array[Long](dW, dH)
    val pads = Array[Int](-padW, -padH)

    // set dimension = 4 forcily. because it seems that only support dimension 4 in mkl dnn
    val dimension = 4

    val inputLayout = new MklLayout(4, Array(
      input.size(input.dim()), // width
      input.size(input.dim() - 1), // height
      if (input.dim() < 3) 1 else input.size(input.dim() - 2), // channels
      if (input.dim() < 4) 1 else input.size(input.dim() - 3) // number
    ))

    def computeOutput(input: Int, pad: Int, kernel: Int, stride: Int): Int = {
        math.ceil(1.0 * (input + 2 * pad - kernel) / stride).toInt + 1
    }

    val outputLayout = new MklLayout(4, Array(
      computeOutput(inputLayout.size(0).toInt, padW, kW, dW), // width
      computeOutput(inputLayout.size(1).toInt, padH, kH, dH), // height
      if (input.dim() < 3) 1 else input.size(input.dim() - 2), // channels
      if (input.dim() < 4) 1 else input.size(input.dim() - 3) // number
    ))

    val weightLayout = new MklLayout(dimension, Array(
      kW, kH
    ))

    // layout ptr in JNI, it may be confused.
    val layout = if (input.isMklTensor()) {
      input.asInstanceOf[MklTensor[T]].layoutMkl
    } else {
      refs.input.createUsrLayout(dimension, inputLayout.size, inputLayout.strides)
      refs.input.layoutUsr
    }

    ev.getType() match {
      case FloatType =>
        this.primitive.forward = MklDnnFloat.poolCreateForward(
          algorithm,
          layout,
          weightLayout.size,
          strides,
          pads,
          Border.dnnBorderZeros
        )
        require(this.primitive.forward != 0, "create convolution primitive failed.")

        this.primitive.backward = MklDnnFloat.poolCreateBackward(
          algorithm,
          layout,
          weightLayout.size,
          strides,
          pads,
          Border.dnnBorderZeros
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
    refs.gradInput.createConversion(inputLayout, primitive.backward,
      ResourceType.dnnResourceDiffSrc)
    refs.gradOutput.createConversion(outputLayout, primitive.backward,
      ResourceType.dnnResourceDiffDst)

    refs.workspace.createMklLayout(primitive.forward, ResourceType.dnnResourceWorkspace)

    setInit(true)
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (!isInited) {
      initLayerAttributes(input)
    }

    refs.input.set(input)

    java.util.Arrays.fill(resources, 0)
    resources(ResourceType.dnnResourceSrc) = refs.input.getConvertedStorage()
    resources(ResourceType.dnnResourceDst) = refs.output.mklStorage()
    resources(ResourceType.dnnResourceWorkspace) = refs.workspace.mklStorage()

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
    resources(ResourceType.dnnResourceDiffSrc) = refs.gradInput.mklStorage()
    resources(ResourceType.dnnResourceDiffDst) = refs.gradOutput.getConvertedStorage()
    resources(ResourceType.dnnResourceWorkspace) = refs.workspace.mklStorage()

    refs.gradInput.zero()

    execute(resources, primitive.backward)

    if (this.prevModuleType == DNN) {
      this.gradInput = refs.gradInput
    } else {
      this.gradInput.zero()
      refs.gradInput.backToUsr(gradInput)
    }

    this.gradInput
  }

  override def toString: String = {
    s"mkl.Pooling"
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

class SpatialMaxPooling[T: ClassTag](kW: Int,
                                     kH: Int,
                                     dW: Int,
                                     dH: Int,
                                     padW: Int = 0,
                                     padH: Int = 0)(
    implicit ev: TensorNumeric[T])
    extends Pool[T](kW, kH, dW, dH, padW, padH, Algorithm.dnnAlgorithmPoolingMax) {

  def this(kW: Int, kH: Int)(implicit ev: TensorNumeric[T]) {
    this(kW, kH, kW, kH, 0, 0)
  }

  override def toString: String = {
    s"""mkl.SpatialMaxPooling($kW, $kH, $dW, $dH,
        |$padW, $padH)""".stripMargin.replaceAll("\n", " ")
  }
}

class SpatialAveragePooling[T: ClassTag](kW: Int,
                                         kH: Int,
                                         dW: Int,
                                         dH: Int,
                                         padW: Int = 0,
                                         padH: Int = 0)(
    implicit ev: TensorNumeric[T])
    extends Pool[T](kW, kH, dW, dH, padW, padH, Algorithm.dnnAlgorithmPoolingAvg) {

  def this(kW: Int, kH: Int)(implicit ev: TensorNumeric[T]) {
    this(kW, kH, kW, kH, 0, 0)
  }

  override def toString: String = {
    s"""mkl.SpatialAveragePooling($kW, $kH, $dW, $dH,
        |$padW, $padH)""".stripMargin.replaceAll("\n", " ")
  }

}
