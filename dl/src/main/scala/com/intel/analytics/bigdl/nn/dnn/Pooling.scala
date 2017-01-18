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

import com.intel.analytics.bigdl.nn.Pooling
import com.intel.analytics.bigdl.mkl.MklDnnFloat
import com.intel.analytics.bigdl.nn.abstractnn.ModuleType._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{FloatType, MklTensor, Tensor}

import scala.reflect.ClassTag

class Pool[T: ClassTag](kW: Int,
                        kH: Int,
                        dW: Int,
                        dH: Int,
                        padW: Int = 0,
                        padH: Int = 0,
                        algorithm: Int)(implicit ev: TensorNumeric[T])
  extends Pooling[T](kW, kH, dW, dH, padW, padH) with MklModuleMethods {

  class PoolRef extends Ref {
    val workspace = new MklTensor[T]()

    override def release(): Unit = {
      super.release()

      workspace.release()
    }
  }

  class PoolPrimitive extends Primitive {}

  @transient
  var refs: PoolRef = null
  @transient
  var primitive: PoolPrimitive = null
  val resources = new Array[Long](ResourceType.dnnResourceNumber)

  override def ceil(): Pool[T] = {
    ceilMode = true
    this
  }

  override def floor(): Pooling[T] = {
    ceilMode = false
    this
  }

  var ceilMode = false
  var ceilLayout: MklLayout = _
  var outputLayout: MklLayout = _

  private[this] def initLayerAttributes(input: Tensor[T]): Unit = {
    if (refs == null) {
      refs = new PoolRef
    }
    if (primitive == null) {
      primitive = new PoolPrimitive
    }

    savedSize = Some(input.size().clone())

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

    /**
     * return the output width / height based on input width / height and other attributes
     *
     * ow = (iw - kw + 2 * padw) / stridew + 1
     * oh = (ih - kh + 2 * padh) / strideh + 1
     *
     *
     * @param input input width / height
     * @param pad pad width / height
     * @param kernel kernel width / height
     * @param stride stride width / height
     * @param ceil ceil mode or not
     * @return output width / height
     */
    def computeOutput(input: Int, pad: Int, kernel: Int, stride: Int, ceil: Boolean): Int = {
      if (ceil) {
        math.ceil(1.0 * (input + 2 * pad - kernel) / stride).toInt + 1
      } else {
        math.floor(1.0 * (input + 2 * pad - kernel) / stride).toInt + 1
      }
    }

    // supports ceil mode
    val widthCeil = computeOutput(inputLayout.size(0).toInt, padW, kW, dW, ceil = true)
    val heightCeil = computeOutput(inputLayout.size(1).toInt, padH, kH, dH, ceil = true)
    val widthFloor = computeOutput(inputLayout.size(0).toInt, padW, kW, dW, ceil = false)
    val heightFloor = computeOutput(inputLayout.size(1).toInt, padH, kH, dH, ceil = false)

    if ((widthCeil == widthFloor) && (heightCeil == heightFloor)) {
      // for convenience, ceilMode will be set true even though user sets it to false
      ceilMode = true
    }

    outputLayout = new MklLayout(4, Array(
      // the ceil mode must be true, it only supports this
      computeOutput(inputLayout.size(0).toInt, padW, kW, dW, ceilMode), // width
      computeOutput(inputLayout.size(1).toInt, padH, kH, dH, ceilMode), // height
      if (input.dim() < 3) 1 else input.size(input.dim() - 2), // channels
      if (input.dim() < 4) 1 else input.size(input.dim() - 3) // number
    ))

    ceilLayout = new MklLayout(4, Array(
      // the ceil mode must be true, it only supports this
      computeOutput(inputLayout.size(0).toInt, padW, kW, dW, ceil = true), // width
      computeOutput(inputLayout.size(1).toInt, padH, kH, dH, ceil = true), // height
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
    refs.output.createConversion(ceilLayout, primitive.forward, ResourceType.dnnResourceDst)
    refs.gradInput.createConversion(inputLayout, primitive.backward,
      ResourceType.dnnResourceDiffSrc)
    refs.gradOutput.createConversion(ceilLayout, primitive.backward,
      ResourceType.dnnResourceDiffDst)

    refs.workspace.createMklLayout(primitive.forward, ResourceType.dnnResourceWorkspace)

    setInit(true)
  }

  def releaseAll(): Unit = {
    if (refs != null && primitive != null) {
      refs.release()
      primitive.release()

      setInit(false)
    }
  }

  private[this] def padding(input: Tensor[T], storage: Long): Long = {
    wrapper {
      MklDnnFloat.padding(
        input.storage().array().asInstanceOf[Array[Float]],
        input.storageOffset() - 1,
        storage,
        outputLayout.size,
        outputLayout.strides,
        ceilLayout.size,
        ceilLayout.strides)
    }

    storage
  }

  private[this] def unPadding(input: Tensor[T], storage: Long): Unit = {
    wrapper {
      MklDnnFloat.unPadding(
        input.storage().array().asInstanceOf[Array[Float]],
        input.storageOffset() - 1,
        storage,
        ceilLayout.strides,
        outputLayout.size,
        outputLayout.strides
      )
    }
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (input.size().deep != savedSize.getOrElse(Array()).deep || ! isInited) {
      releaseAll()
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
      output.resizeAs(refs.output)
      if (!ceilMode) {
        unPadding(output, refs.output.mklStorage())
      } else {
        refs.output.backToUsr(output)
      }
    }

    if (this.isTraining()) {
      refs.input.setConverted(true)
    }

    this.output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    refs.gradOutput.set(gradOutput)
    refs.input.set(input)

    val gradOutputStorage = if (!ceilMode) {
      padding(gradOutput, refs.gradOutput.mklStorage())
    } else {
      refs.gradOutput.getConvertedStorage()
    }

    java.util.Arrays.fill(resources, 0)
    resources(ResourceType.dnnResourceDiffSrc) = refs.gradInput.mklStorage()
    resources(ResourceType.dnnResourceWorkspace) = refs.workspace.mklStorage()
    resources(ResourceType.dnnResourceDiffDst) = gradOutputStorage

    refs.gradInput.zero()

    execute(resources, primitive.backward)

    if (this.prevModuleType == DNN) {
      this.gradInput = refs.gradInput
    } else {
      gradInput.resizeAs(refs.gradInput)
      gradInput.zero()
      refs.gradInput.backToUsr(gradInput)
    }

    if (this.isTraining()) {
      refs.input.setConverted(false)
    }

    this.gradInput
  }

  override def toString: String = {
    s"mkl.Pooling"
  }

  override def convertToMklDnn(prevModule: Option[AbstractModule[Activity, Activity, T]] = None)
  : (ModuleType, AbstractModule[Activity, Activity, T]) =
    super[MklModuleMethods].convertToMklDnn(prevModule)

  override def setNextModuleType(value: ModuleType): Unit = {
    val moduleType = if (!ceilMode) {
      BLAS
    } else {
      value
    }

    super[MklModuleMethods].setNextModuleType(moduleType)
  }

  override def setPrevModuleType(value: ModuleType): Unit =
    super[MklModuleMethods].setPrevModuleType(value)

  override def nextModuleType: ModuleType = super[MklModuleMethods].nextModuleType

  override def prevModuleType: ModuleType = super[MklModuleMethods].prevModuleType

  override def moduleType(): ModuleType = {
    if (!ceilMode) {
      BLAS
    } else {
      DNN
    }
  }
}

@SerialVersionUID(- 9140346840084610380L)
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

@SerialVersionUID(2161491765598561395L)
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
