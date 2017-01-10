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
import com.intel.analytics.bigdl.nn.AbstractConcat
import com.intel.analytics.bigdl.nn.abstractnn.ModuleType._
import com.intel.analytics.bigdl.tensor.{FloatType, MklTensor, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}

import scala.reflect.ClassTag

@SerialVersionUID(- 4157856938968907891L)
class Concat[T: ClassTag](dimension: Int)(implicit ev: TensorNumeric[T])
  extends AbstractConcat[T](dimension) {

  class ConcatRef {
    var inputs: Array[MklTensor[T]] = null //  we don't known the length of modules
    val output = new MklTensor[T]()
  }

  class SplitRef {
    val gradOutput = new MklTensor[T]()
    var gradInputs: Array[MklTensor[T]] = null
    var gradInputsUsr: Array[Tensor[T]] = null
  }

  class SumRef {
    val gradInput = new MklTensor[T]()
    var gradOutputs: Array[MklTensor[T]] = null
  }

  class Ref {
    val concat = new ConcatRef
    val split = new SplitRef
    val sum = new SumRef
  }

  class Primitive {
    var concat = 0L
    var split = 0L
    var sum = 0L
  }

  @transient
  val primitive = new Primitive
  @transient
  val refs = new Ref
  val resources = new Array[Long](ResourceType.dnnResourceNumber)

  var coefficients: Array[T] = null

  private[this] var _isConcatInited: Boolean = false

  def isConcatInited: Boolean = _isConcatInited
  def setConcatInit(value: Boolean): Unit = {
    _isConcatInited = value
  }

  private[this] var _isSplitInited: Boolean = false

  def isSplitInited: Boolean = _isSplitInited
  def setSplitInit(value: Boolean): Unit = {
    _isSplitInited = value
  }

  private[this] var _isSumInited: Boolean = false

  def isSumInited: Boolean = _isSumInited
  def setSumInit(value: Boolean): Unit = {
    _isSumInited = value
  }

  private[this] var _nextModuleType: ModuleType = BLAS
  private[this] var _prevModuleType: ModuleType = BLAS

  override def moduleType: ModuleType = DNN

  override def nextModuleType: ModuleType = _nextModuleType
  override def setNextModuleType(value: ModuleType): Unit = {
    value match {
      case DNN => _nextModuleType = DNN
      case _ =>
    }
  }

  override def prevModuleType: ModuleType = _prevModuleType
  override def setPrevModuleType(value: ModuleType): Unit = {
    value match {
      case DNN => _prevModuleType = DNN
      case _ =>
    }
  }

  override def reset(): Unit = {
    require(this.modules.length <= 4 && this.modules.nonEmpty)
  }

  def execute(resources: Array[Long], primitive: Long): Unit = {
    ev.getType() match {
      case FloatType => MklDnnFloat.execute(resources, primitive)
      case _ => throw new UnsupportedOperationException(s"Only Float supported")
    }
  }

  private[this] def initConcat(inputs: Array[Tensor[T]]): Unit = {
    require(inputs.length > 0, s"input of concat is not satisfied.")

    var channels = 0 // output channels
    val numConcats = inputs.length
    val nDimension = inputs(0).nDimension()
    // layoutPtr means the layout pointer create by MKL-DNN in JNI
    // layout means the number, channel, ... discription in scala
    val layoutsPtr = new Array[Long](numConcats)
    val layouts = new Array[MklLayout](numConcats) // for creating conversion

    refs.concat.inputs = new Array[MklTensor[T]](numConcats)
    for (i <- 0 until numConcats) {
      refs.concat.inputs(i) = new MklTensor[T]()
    }

    for (i <- 0 until numConcats) {
      val one = inputs(i)
//      require(nDimension == one.nDimension(), s"the dimension of outputs in concats must be same")

      val size = one.size().reverse.map(_.toLong).slice(0, one.nDimension())
      val layout = if (one.nDimension() <= 2) {
        new MklLayout(4, Array(1, 1, size(0), size(1)))
      } else {
        new MklLayout(4, size)
      }
      layouts(i) = layout

      // mkl concat only supports 4-D
      val layoutPtr = if (one.isMklTensor() && one.nDimension() >= 4) {
        one.asInstanceOf[MklTensor[T]].layoutMkl
      } else {
        refs.concat.inputs(i).createUsrLayout(layout.dimension, layout.size, layout.strides)
        refs.concat.inputs(i).layoutUsr
      }

      layoutsPtr(i) = layoutPtr

      channels += one.size(this.dimension)
    }

    // create primitive
    ev.getType() match {
      case FloatType =>
        primitive.concat = MklDnnFloat.concatCreate(numConcats, layoutsPtr)
        require(primitive.concat != 0, s"create primitive of concat failed.")
      case _ =>
        throw new UnsupportedOperationException(s"Only Float supported")
    }

    // create conversion
    for (i <- 0 until numConcats) {
      refs.concat.inputs(i).resizeAs(inputs(i))
      refs.concat.inputs(i).createConversion(layouts(i), primitive.concat,
        ResourceType.dnnResourceMultipleSrc + i)
    }

    val tmp = inputs(0).size().reverse.map(_.toLong).slice(0, nDimension)
    val outputSize = new Array[Long](4)
    if (nDimension > 2) {
      for (i <- 0 until 4) {
        outputSize(i) = tmp(i)
      }
    } else {
      outputSize(0) = 1
      outputSize(1) = 1
      outputSize(2) = tmp(0)
      outputSize(3) = tmp(1)
    }
    outputSize(this.dimension) = channels
    val outputLayout = new MklLayout(4, outputSize)

    if (nDimension > 2) {
      refs.concat.output.resize(outputSize.reverse.map(_.toInt))
      this.output.resize(outputSize.reverse.map(_.toInt))
    } else {
      refs.concat.output.resize(outputSize.reverse.map(_.toInt).slice(0, nDimension))
      this.output.resize(outputSize.reverse.map(_.toInt).slice(0, nDimension))
    }

    refs.concat.output.createConversion(outputLayout, primitive.concat, ResourceType.dnnResourceDst)
    setConcatInit(true)
  }

  private[this] def initSplit(inputs: Array[Tensor[T]], gradOutput: Tensor[T]): Unit = {
    val layouts = new Array[MklLayout](modules.length)
    val numSplits = inputs.length
    val distriSplits = new Array[Long](modules.length)

    for (i <- 0 until numSplits) {
      val one = inputs(i)
//      require(nDimension == one.nDimension(),
//        s"the dimension of outputs in concats must be same")

      distriSplits(i) = one.size(this.dimension)

      val oneSize = new Array[Int](4)
      Array.copy(one.size(), 0, oneSize, 0, one.size().length)
      for (i <- one.size().length until 4) {
        oneSize(i) = 1
      }

      val layout = new MklLayout(4, oneSize.reverse.map(_.toLong))
//      val layout = new MklLayout(one.nDimension, size)
      layouts(i) = layout
    }

    val outputSize = new Array[Int](4)
    Array.copy(gradOutput.size(), 0, outputSize, 0, gradOutput.size().length)
    for (i <- gradOutput.size().length until 4) {
      outputSize(i) = 1
    }

    val gradOutputSize = new MklLayout(4, outputSize.reverse.map(_.toLong))

    val gradOutputLayoutPtr = if (gradOutput.isMklTensor()) {
      gradOutput.asInstanceOf[MklTensor[T]].layoutMkl
    } else {
      refs.split.gradOutput.createUsrLayout(gradOutputSize.dimension, gradOutputSize.size,
        gradOutputSize.strides)
      refs.split.gradOutput.layoutUsr
    }

    // create primitive
    ev.getType() match {
      case FloatType =>
        primitive.split = MklDnnFloat.splitCreate(numSplits, gradOutputLayoutPtr, distriSplits)
        require(primitive.split != 0, s"create primitive of split failed.")
      case _ =>
        throw new UnsupportedOperationException(s"Only Float supported")
    }

    refs.split.gradInputs = new Array[MklTensor[T]](numSplits)
    refs.split.gradInputsUsr = new Array[Tensor[T]](numSplits)
    for (i <- 0 until numSplits) {
      refs.split.gradInputs(i) = new MklTensor[T]()
      refs.split.gradInputs(i).resizeAs(inputs(i))

      refs.split.gradInputsUsr(i) = Tensor[T]()
      refs.split.gradInputsUsr(i).resizeAs(inputs(i))
    }

    refs.split.gradOutput.resize(gradOutput.size())

    // create conversion
    refs.split.gradOutput.createConversion(gradOutputSize, primitive.split,
      ResourceType.dnnResourceSrc)

    for (i <- modules.indices) {
      refs.split.gradInputs(i).createConversion(layouts(i), primitive.split,
        ResourceType.dnnResourceMultipleDst + i)
    }

    setSplitInit(true)
  }

  private[this] def initSum(input: Tensor[T], gradOutputs: Array[Tensor[T]]): Unit = {
    val nDimension = gradOutputs(0).nDimension()
    val numSums = gradOutputs.length
    val size = gradOutputs(0).size().reverse.map(_.toLong).slice(0, nDimension)
    val inputSize = new MklLayout(nDimension, size)

    coefficients = new Array[T](gradOutputs.length)
    for (i <- coefficients.indices) {
      coefficients(i) = ev.fromType(1)
    }

    val layout = if (input.isMklTensor()) {
      input.asInstanceOf[MklTensor[T]].layoutMkl
    } else {
      refs.sum.gradInput.createUsrLayout(inputSize.dimension, inputSize.size, inputSize.strides)
      refs.sum.gradInput.layoutUsr
    }

    ev.getType() match {
      case FloatType =>
        primitive.sum = MklDnnFloat.sumCreate(numSums, layout,
          coefficients.asInstanceOf[Array[Float]])
        require(primitive.sum != 0, s"create primitive of sum failed.")
      case _ =>
        throw new UnsupportedOperationException(s"Only Float supported")
    }

    refs.sum.gradOutputs = new Array[MklTensor[T]](numSums)
    for (i <- 0 until numSums) {
      refs.sum.gradOutputs(i) = new MklTensor[T]()
      refs.sum.gradOutputs(i).resizeAs(gradOutputs(i))
    }
    refs.sum.gradInput.resizeAs(input)
    this.gradInput.resizeAs(input)

    for (i <- gradOutputs.indices) {
      refs.sum.gradOutputs(i).createConversion(inputSize, primitive.sum,
        ResourceType.dnnResourceMultipleSrc + i)
    }
    refs.sum.gradInput.createConversion(inputSize, primitive.sum, ResourceType.dnnResourceDst)

    setSumInit(true)
  }

  private[this] def concat(): Tensor[T] = {
    // init concat
    if (!isConcatInited) {
      val leaves = new Array[Tensor[T]](modules.length)
      for (i <- modules.indices) {
        leaves(i) = modules(i).output.asInstanceOf[Tensor[T]]
      }
      initConcat(leaves)
    }

    for (i <- modules.indices) {
      refs.concat.inputs(i).set(modules(i).output.asInstanceOf[Tensor[T]], usrOnly = true)
    }

    java.util.Arrays.fill(resources, 0)
    for (i <- modules.indices) {
      resources(ResourceType.dnnResourceMultipleSrc + i) =
        refs.concat.inputs(i).getConvertedStorage()
    }
    resources(ResourceType.dnnResourceDst) = refs.concat.output.mklStorage()

    execute(resources, primitive.concat)

    if (this.nextModuleType == DNN) {
      this.output = refs.concat.output
    } else {
      refs.concat.output.backToUsr(this.output)
    }

    this.output
  }

  private[this] def split(gradOutput: Tensor[T]): Array[Tensor[T]] = {
    // init
    if (!isSplitInited) {
      val leaves = new Array[Tensor[T]](modules.length)
      for (i <- modules.indices) {
        leaves(i) = modules(i).output.toTensor
      }
      val root = gradOutput
      initSplit(leaves, root)
    }

    refs.split.gradOutput.set(gradOutput)

    java.util.Arrays.fill(resources, 0)
    for (i <- modules.indices) {
      resources(ResourceType.dnnResourceMultipleDst + i) = refs.split.gradInputs(i).mklStorage()
    }
    resources(ResourceType.dnnResourceSrc) = refs.split.gradOutput.getConvertedStorage()

    execute(resources, primitive.split)

    // conversion depends on modules(i).output type
    val leaves = new Array[Tensor[T]](modules.length)
    for (i <- modules.indices) {
      if (modules(i).output.asInstanceOf[Tensor[T]].isMklTensor() &&
        modules(i).output.toTensor.dim() >= 4) {
        leaves(i) = refs.split.gradInputs(i)
      } else {
        leaves(i) = Tensor[T]().resizeAs(refs.split.gradInputs(i))
        refs.split.gradInputs(i).backToUsr(leaves(i))
      }
//      leaves(i) = refs.split.gradInputsUsr(i)
//      refs.split.gradInputs(i).backToUsr(leaves(i))
    }

    leaves
  }

  private[this] def sum(input: Tensor[T]): Tensor[T] = {
    // sum
    if (!isSumInited) {
      val leaves = new Array[Tensor[T]](modules.length)
      for (i <- modules.indices) {
        leaves(i) = modules(i).gradInput.asInstanceOf[Tensor[T]]
      }
      initSum(input, leaves)
    }

    for (i <- modules.indices) {
      refs.sum.gradOutputs(i).set(modules(i).gradInput.asInstanceOf[Tensor[T]])
    }

    java.util.Arrays.fill(resources, 0)
    for (i <- modules.indices) {
      resources(ResourceType.dnnResourceMultipleSrc + i) =
        refs.sum.gradOutputs(i).getConvertedStorage()
    }
    resources(ResourceType.dnnResourceDst) = refs.sum.gradInput.mklStorage()

    execute(resources, primitive.sum)

    if (this.prevModuleType == DNN) {
      this.gradInput = refs.sum.gradInput
    } else {
      refs.sum.gradInput.backToUsr(this.gradInput)
    }

    this.gradInput
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    // submodules forward, DO NOT bind core
    // TODO change while -> for

    for (module <- modules) { module.updateOutput(input) }

    concat()
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val gradOutputs = split(gradOutput)

    for (i <- modules.indices) {
      modules(i).backward(input, gradOutputs(i))
    }

    sum(input)
  }

  override def equals(obj: Any): Boolean = {
    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[Concat[T]]) {
      return false
    }
    val other = obj.asInstanceOf[Concat[T]]
    if (this.eq(other)) {
      return true
    }
    if (dimension != other.dimension) {
      return false
    }

    if (this.modules.length != other.modules.length) {
      return false
    }

    val moduleLength = modules.length
    var i = 0
    while (i < moduleLength) {
      if (modules(i) != other.modules(i)) {
        return false
      }
      i += 1
    }

    true
  }
  override def hashCode(): Int = {

    val seed = 37
    var hash = super.hashCode()
    var i = 0
    val moduleLength = modules.length
    while (i < moduleLength) {
      hash = hash * seed + modules(i).hashCode()
      i += 1
    }

    hash
  }

  override def toString: String = {
    val tab = "  "
    val next = "  |`-> "
    val last = "   ... -> "
    val ext = "  |    "
    val extlast = "       "
    s"mkl.Concat {$line${tab}input$line${
      modules.zipWithIndex
        .map { case (model: AbstractModule[Activity, Activity, T], index: Int)
        => s"$tab$next(${index + 1}): ${
          if (index == modules.length - 1) {
            model.setLine(line + tab + extlast)
          } else {
            model.setLine(line + tab + ext)
          }
        }"
        }
        .mkString(line)
    }$line$tab${last}output$line$tab}"
  }

  override def convertToMklDnn(prevModule: Option[AbstractModule[Activity, Activity, T]] = None)
  : (ModuleType, AbstractModule[Activity, Activity, T]) = {
    val modulesType = new Array[(ModuleType, AbstractModule[Activity, Activity, T])](modules.length)

    for (i <- modules.indices) {
      modulesType(i) = modules(i).convertToMklDnn(prevModule)
    }

    var isAllMklDnnLayers = true
    for (module <- modulesType) {
      if (module._1 != DNN) {
        isAllMklDnnLayers = false
      }
    }

    val head = if (isAllMklDnnLayers) DNN else BLAS

    if (isAllMklDnnLayers) {
      prevModule match {
        case Some(x) => x.setNextModuleType(head)
        case _ =>
      }
    }

    for (module <- modulesType) {
      if (module._2.moduleType == DNN) {
        module._2.setNextModuleType(DNN)
      }
    }

    prevModule match {
      case Some(x) => if (x.moduleType == DNN) setPrevModuleType(DNN)
      case _ =>
    }

    (head, this.asInstanceOf[AbstractModule[Activity, Activity, T]])
  }
}
