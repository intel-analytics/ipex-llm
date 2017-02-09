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

import com.intel.analytics.bigdl.mkl.MklDnnFloat
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.nn.abstractnn.ModuleType._
import com.intel.analytics.bigdl.nn.AbstractBatchNormalization
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.{FloatType, MklTensor, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

@SerialVersionUID(796580269069270433L)
class BatchNormalization[T: ClassTag](nOutput: Int,
                                      eps: Double = 1e-5,
                                      momentum: Double = 0.1,
                                      affine: Boolean = true)(implicit ev: TensorNumeric[T])
  extends AbstractBatchNormalization[T](nOutput, eps, momentum, affine) with MklModuleMethods {

  require(nOutput > 0,
          "To set affine=false call SpatialBatchNormalization(nFeature,  eps, momentum, false)")

  class BNRef extends Ref[T] {
    val workspace = new MklTensor[T]()
    val scaleShift = new MklTensor[T]()

    override def release(): Unit = {
      super.release()
      
      workspace.release()
      scaleShift.release()
    }
  }

  class BNPrimitive extends Primitive {
    var scaleShift = 0L

    override def release(): Unit = {
      super.release()

      wrapper {
        require(scaleShift != 0, s"scaleShift should not 0")
        MklDnnFloat.deletePrimitive(scaleShift)

        scaleShift = 0L
      }
    }
  }

  @transient
  var refs: BNRef = null
  @transient
  var primitive: BNPrimitive = null
  val resources = new Array[Long](ResourceType.dnnResourceNumber)

  reset()

  override def reset(): Unit = {
    weight.apply1(_ => ev.fromType[Double](RNG.uniform(0, 1)))
    bias.fill(ev.fromType[Int](0))
  }

  private[this] def initLayerAttributes(input: Tensor[T]): Unit = {
    if (refs == null) { refs = new BNRef }
    if (primitive == null) { primitive = new BNPrimitive }

    savedSize = Some(input.size().clone())

    val dimension = 4
    val inputLayout = new MklLayout(dimension, Utils.getSize(input, dimension))

    val layout = if (input.isMklTensor()) {
      input.asInstanceOf[MklTensor[T]].layoutMkl
    } else {
      refs.input.createUsrLayout(inputLayout.dimension, inputLayout.size, inputLayout.strides)
      refs.input.layoutUsr
    }

    ev.getType() match {
      case FloatType =>
        primitive.forward = MklDnnFloat.batchNormCreateForward(layout, eps.toFloat)
        require(this.primitive.forward != 0, "create batchnorm primitive failed.")

        // TODO layout should be gradOutput
        primitive.backward = MklDnnFloat.batchNormCreateBackward(layout, eps.toFloat)
        require(this.primitive.backward != 0, "create batchnorm primitive failed.")

        if (affine) {
          primitive.scaleShift = MklDnnFloat.batchNormCreateScaleShift(layout, eps.toFloat)
          require(this.primitive.scaleShift != 0, "create batchnorm primitive failed.")
        }
      case _ =>
        throw new UnsupportedOperationException(s"Only Float supported")
    }

    refs.input.createConversion(inputLayout, primitive.forward, ResourceType.dnnResourceSrc)
    refs.output.createConversion(inputLayout, primitive.forward, ResourceType.dnnResourceDst)
    refs.gradOutput.createConversion(inputLayout, primitive.backward,
      ResourceType.dnnResourceDiffDst)
    refs.gradInput.createConversion(inputLayout, primitive.backward,
      ResourceType.dnnResourceDiffSrc)

    refs.scaleShift.createMklLayout(primitive.forward, ResourceType.dnnResourceScaleShift)
    refs.workspace.createMklLayout(primitive.forward, ResourceType.dnnResourceWorkspace)

    refs.workspace.zero()
    refs.scaleShift.zero()

    for (tensor <- List(refs.input, refs.output, refs.gradOutput, refs.gradInput)) {
      tensor.resizeAs(input)
    }

    if (nextModuleType != DNN) {
      this.output.resizeAs(input)
    }

    if (prevModuleType != DNN) {
      this.gradInput.resizeAs(input)
    }

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

    refs.input.set(input)

    ev.getType() match {
      case FloatType =>
        val weightArray = weight.storage().array().asInstanceOf[Array[Float]]
        val biasArray = bias.storage().array().asInstanceOf[Array[Float]]

        MklDnnFloat.setScaleShift(bool2int(affine),
          weightArray, weight.storageOffset() - 1,
          biasArray, bias.storageOffset() - 1,
          refs.scaleShift.mklStorage(), nOutput)
      case _ => throw new UnsupportedOperationException(s"Only Float supported")
    }

    java.util.Arrays.fill(resources, 0)
    resources(ResourceType.dnnResourceSrc) = refs.input.getConvertedStorage()
    resources(ResourceType.dnnResourceScaleShift) = refs.scaleShift.mklStorage()
    resources(ResourceType.dnnResourceWorkspace) = refs.workspace.mklStorage()
    resources(ResourceType.dnnResourceDst) = refs.output.mklStorage()

    execute(resources, primitive.forward)

    if (this.nextModuleType == DNN) {
      this.output = refs.output
    } else {
      output.resizeAs(refs.output)
      refs.output.backToUsr(output)
    }

    this.output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    refs.input.set(input)
    refs.gradOutput.set(gradOutput)

    java.util.Arrays.fill(resources, 0)
    resources(ResourceType.dnnResourceSrc) = refs.input.getConvertedStorage()
    resources(ResourceType.dnnResourceDiffDst) = refs.gradOutput.getConvertedStorage()
    resources(ResourceType.dnnResourceScaleShift) = refs.scaleShift.mklStorage()
    resources(ResourceType.dnnResourceWorkspace) = refs.workspace.mklStorage()
    resources(ResourceType.dnnResourceDiffSrc) = refs.gradInput.mklStorage()

    execute(resources, primitive.backward)

    if (this.prevModuleType == DNN) {
      this.gradInput = this.refs.gradInput
    } else {
      gradInput.resizeAs(refs.gradInput)
      refs.gradInput.backToUsr(gradInput)
    }

    this.gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T], scale: Double): Unit = {
    refs.input.set(input)
    refs.gradOutput.set(gradOutput)

    if (affine) {
      java.util.Arrays.fill(resources, 0)
      resources(ResourceType.dnnResourceSrc) = refs.input.getConvertedStorage()
      resources(ResourceType.dnnResourceDiffDst) = refs.gradOutput.getConvertedStorage()
      resources(ResourceType.dnnResourceWorkspace) = refs.workspace.mklStorage()
      resources(ResourceType.dnnResourceDiffScaleShift) = refs.scaleShift.mklStorage()

      execute(resources, primitive.scaleShift)

      ev.getType() match {
        case FloatType =>
          val weightArray = gradWeight.storage().array().asInstanceOf[Array[Float]]
          val biasArray = gradBias.storage().array().asInstanceOf[Array[Float]]

          MklDnnFloat.setGradScaleShift(bool2int(affine),
            weightArray, gradWeight.storageOffset() - 1,
            biasArray, gradBias.storageOffset() - 1,
            refs.scaleShift.mklStorage(), nOutput)
        case _ => throw new UnsupportedOperationException(s"Only Float supported")
      }
    }
  }

  override def updateParameters(learningRate: T): Unit = {
    weight.map(gradWeight, (a, b) => ev.minus(a, ev.times(learningRate, b)))
    bias.map(gradBias, (a, b) => ev.minus(a, ev.times(learningRate, b)))
  }

  override def zeroGradParameters(): Unit = {
    gradWeight.zero()
    gradBias.zero()
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
  }

  override def toString: String = {
    s"mkl.BatchNormalization[${ev.getType()}]($nOutput, $eps, $momentum, $affine)"
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

@SerialVersionUID(7103870054854933642L)
class SpatialBatchNormalization[T: ClassTag](nOutput: Int,
                                             eps: Double = 1e-5,
                                             momentum: Double = 0.1,
                                             affine: Boolean = true)(implicit ev: TensorNumeric[T])
  extends BatchNormalization[T](nOutput, eps, momentum, affine) {
  override def toString: String = {
    s"mkl.SpatialBatchNormalization[${ev.getType()}]($nOutput, $eps, $momentum, $affine)"
  }
}
