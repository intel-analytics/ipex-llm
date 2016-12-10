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
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.tensor.{MklTensor, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.nn.{Constant, Default, InitializationMethod, Xavier}

import scala.reflect.ClassTag

class Linears[@specialized(Float, Double) T: ClassTag](
    inputSize: Int,
    outputSize: Int,
    val needCompute: Boolean = true,
    private var initMethod: InitializationMethod = Default
)(implicit ev: TensorNumeric[T])
    extends MklModule[T] {

  class LinearRef extends Ref {
    var weight = new MklTensor[T]()
    var bias = new MklTensor[T]()

    var gradWeight = new MklTensor[T]()
    var gradBias = new MklTensor[T]()
  }

  class LinearPrimitive extends Primitive {
    var backWeight = 0L
    var backBias = 0L
  }

  val refs = new LinearRef
  val primitive = new LinearPrimitive

  val weight: Tensor[T] = Tensor[T](outputSize, inputSize)
  val gradWeight = Tensor[T](outputSize, inputSize)
  val bias: Tensor[T] = Tensor[T](outputSize)
  val gradBias = Tensor[T](outputSize)

  def setInitMethod(initMethod: InitializationMethod): this.type = {
    this.initMethod = initMethod
    this
  }

  override def reset(): Unit = {
    initMethod match {
      case Default =>
        val stdv = 1.0 / math.sqrt(weight.size(2))
        weight.apply1(_ => ev.fromType[Double](RNG.uniform(0, 1) * 2 * stdv - stdv))
        bias.apply1(_ => ev.fromType[Double](RNG.uniform(0, 1) * 2 * stdv - stdv))
      case Xavier =>
        val fanIn = weight.size(2)
        val fanOut = weight.size(1)
        val stdv = math.sqrt(6.0 / (fanIn + fanOut))
        weight.apply1(_ => ev.fromType[Double](RNG.uniform(-stdv, stdv)))
        bias.fill(ev.fromType(0))
      case Constant =>
        weight.fill(ev.fromType(0.1))
        bias.fill(ev.fromType(0))
    }
  }

  private[this] def initLayerAttributes(input: Tensor[T]): Unit = {
    val dimension = 2

    val inputLayout = new MklLayout(
      dimension,
      Array(
        input.size(input.dim()), // width
        if (input.dim() < 2) 1 else input.size(input.dim() - 1) // number
      ))

    val outputLayout = new MklLayout(dimension,
      Array(
        outputSize,
        if (input.dim() < 2) 1 else input.size(input.dim() - 1) // number
      ))

    val weightLayout = new MklLayout(dimension,
      Array(
        inputSize,
        outputSize
      ))

    val biasLayout = new MklLayout(1,
      Array(
        outputSize
      ))

    refs.input.resizeAs(input)
    refs.gradInput.resizeAs(input)

    refs.output.resize(outputLayout.size.reverse.map(_.toInt),
      outputLayout.strides.reverse.map(_.toInt))
    refs.gradOutput.resizeAs(refs.output)

    this.output.resizeAs(refs.output)
    this.gradInput.resizeAs(refs.input)


    for (i <- List(refs.weight, refs.gradWeight)) i.resizeAs(weight)
    for (i <- List(refs.bias, refs.gradBias)) i.resizeAs(bias)

    initForward(inputLayout, outputLayout, weightLayout, biasLayout, outputSize)
    initBackwardData(inputLayout, outputLayout, weightLayout, biasLayout, outputSize)
    initBackwardWeight(inputLayout, outputLayout, weightLayout, biasLayout, outputSize)
    initBackwardBias(outputLayout, biasLayout)

    isInited_=(true)
  }

  private[this] def initForward(inputLayout: MklLayout,
                                outputLayout: MklLayout,
                                weightLayout: MklLayout,
                                biasLayout: MklLayout,
                                outputChannel: Long): Unit = {
    ev.getType() match {
      case "Float" =>
        this.primitive.forward = MklDnnFloat
          .linearCreateForwardBias(inputLayout.dimension, inputLayout.size, outputChannel)
        require(this.primitive.forward != 0, "create convolution primitive failed.")
      case _ =>
        throw new UnsupportedOperationException(s"Only Float supported")
    }

    refs.input.createConversion(inputLayout, primitive.forward, ResourceType.dnnResourceSrc)
    refs.weight.createConversion(weightLayout, primitive.forward, ResourceType.dnnResourceFilter)
    refs.bias.createConversion(biasLayout, primitive.forward, ResourceType.dnnResourceBias)
    refs.output.createConversion(outputLayout, primitive.forward, ResourceType.dnnResourceDst)
  }

  private[this] def initBackwardData(inputLayout: MklLayout,
                                     outputLayout: MklLayout,
                                     weightLayout: MklLayout,
                                     biasLayout: MklLayout,
                                     outputChannel: Long): Unit = {
    ev.getType() match {
      case "Float" =>
        this.primitive.backward = MklDnnFloat.linearCreateBackData(
          inputLayout.dimension, inputLayout.size, outputChannel)
        require(this.primitive.backward != 0, "create convolution primitive failed.")
      case _ =>
        throw new UnsupportedOperationException(s"Only Float supported")
    }

    refs.gradOutput.createConversion(outputLayout,primitive.backward,
      ResourceType.dnnResourceDiffDst)
    refs.weight.createConversion(weightLayout, primitive.backward,
      ResourceType.dnnResourceFilter)
    refs.gradInput.createConversion(inputLayout, primitive.backward,
      ResourceType.dnnResourceDiffSrc)
  }

  def initBackwardWeight(inputLayout: MklLayout,
                         outputLayout: MklLayout,
                         weightLayout: MklLayout,
                         biasLayout: MklLayout,
                         outputChannel: Long): Unit = {
    ev.getType() match {
      case "Float" =>
        this.primitive.backWeight = MklDnnFloat.linearCreateBackWeight(
          inputLayout.dimension, inputLayout.size, outputChannel)
        require(this.primitive.backWeight != 0, "create convolution primitive failed.")
      case _ =>
        throw new UnsupportedOperationException(s"Only Float supported")
    }

    refs.gradWeight.createConversion(weightLayout, primitive.backWeight,
      ResourceType.dnnResourceDiffFilter)
  }

  def initBackwardBias(outputLayout: MklLayout, biasLayout: MklLayout): Unit = {
    ev.getType() match {
      case "Float" =>
        this.primitive.backBias = MklDnnFloat.linearCreateBackBias(
          outputLayout.dimension, outputLayout.size)
        require(this.primitive.backBias != 0, "create convolution primitive failed.")
      case _ =>
        throw new UnsupportedOperationException(s"Only Float supported")
    }

    refs.gradBias.createConversion(biasLayout, primitive.backBias,
      ResourceType.dnnResourceDiffBias)
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (!isInited) {
      initLayerAttributes(input)
    }
    refs.input.set(input)
    refs.weight.set(weight)
    refs.bias.set(bias)

    ev.getType() match {
      case "Float" =>
        MklDnnFloat.linearForwardExecute(
          refs.input.getConvertedStorage().array().asInstanceOf[Array[Float]],
          refs.weight.getConvertedStorage().array().asInstanceOf[Array[Float]],
          refs.bias.getConvertedStorage().array().asInstanceOf[Array[Float]],
          refs.output.mklStorage().array().asInstanceOf[Array[Float]],
          primitive.forward
        )
      case _ =>
        throw new UnsupportedOperationException(s"Only Float supported")
    }

    if (this.nextModuleType() == DNN) {
      this.output = refs.output
    } else {
      refs.output.backToUsr(output.storage(), output.storageOffset())
    }

    this.output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    refs.gradOutput.set(gradOutput)

    ev.getType() match {
      case "Float" =>
        MklDnnFloat.linearBackDataExecute(
          refs.gradInput.mklStorage().array().asInstanceOf[Array[Float]],
          refs.gradOutput.getConvertedStorage().array().asInstanceOf[Array[Float]],
          refs.weight.getConvertedStorage().array().asInstanceOf[Array[Float]],
          primitive.backward)
    }

    if (this.prevModuleType() == DNN) {
      this.gradInput = this.refs.gradInput
    } else {
      refs.gradInput.backToUsr(this.gradInput.storage(), this.gradInput.storageOffset())
    }

    this.gradInput
  }

  override def accGradParameters(input: Tensor[T],
                                 gradOutput: Tensor[T],
                                 scale: Double = 1.0): Unit = {
    refs.input.set(input)
    refs.gradOutput.set(gradOutput)

    ev.getType() match {
      case "Float" =>
        MklDnnFloat.linearBackWeightExecute(
          refs.input.getConvertedStorage().array().asInstanceOf[Array[Float]],
          refs.gradOutput.getConvertedStorage().array().asInstanceOf[Array[Float]],
          refs.gradWeight.mklStorage().array().asInstanceOf[Array[Float]],
          primitive.backWeight
        )
        MklDnnFloat.linearBackBiasExecute(
          refs.gradOutput.getConvertedStorage().array().asInstanceOf[Array[Float]],
          refs.gradBias.mklStorage().array().asInstanceOf[Array[Float]],
          primitive.backBias
        )
    }

    refs.gradWeight.backToUsr(gradWeight.storage(), gradWeight.storageOffset())
    refs.gradBias.backToUsr(gradBias.storage(), gradBias.storageOffset())

  }

  override def updateParameters(learningRate: T): Unit = {
    weight.add(ev.negative(learningRate), gradWeight)
    bias.add(ev.negative(learningRate), gradBias)
  }

  override def zeroGradParameters(): Unit = {
    gradWeight.zero()
    gradBias.zero()
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
  }

  override def equals(obj: Any): Boolean = {
    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[Linear[T]]) { return false }
    val other = obj.asInstanceOf[Linear[T]]
    if (this.eq(other)) { return true }

    gradWeight == other.gradWeight &&
      gradBias == other.gradBias &&
      weight == other.weight &&
      bias == other.bias
  }

  override def hashCode(): Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + gradWeight.hashCode()
    hash = hash * seed + gradBias.hashCode()
    hash = hash * seed + weight.hashCode()
    hash = hash * seed + bias.hashCode()

    hash
  }

  override def toString(): String = {
    s"mkl.Linear($inputSize -> $outputSize)"
  }
}
