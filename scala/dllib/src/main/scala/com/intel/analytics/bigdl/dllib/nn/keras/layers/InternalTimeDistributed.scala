/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.api.keras.layers.internal

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.bigdl.utils.serializer.{DeserializeContext, ModuleSerializable}
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag


/**
 * NB: This implementation includes some bug fixes for
 * "com.intel.analytics.bigdl.nn.Timedistributed". Also it accepts input as Tensor
 * or Tables whose elements are Tensor. Nested Tables are not supported currently.
 *
 * This layer is intended to apply contained layer to each temporal time slice
 * of input tensor.
 *
 * For instance, The TimeDistributed Layer can feed each time slice of input tensor
 * to the Linear layer.
 *
 * The input data format is [Batch, Time, Other dims]. For the contained layer, it must not change
 * the Other dims length.
 *
 * @param maskZero: if `maskZero` is set to true, if the input including zero vectors, the
 *                corresponding output will be set to zero vecotrs.
 * @tparam T data type, which can be [[Double]] or [[Float]]
 */

private[zoo] class InternalTimeDistributed[T: ClassTag](
    val layer: AbstractModule[Activity, Tensor[T], T],
    maskZero: Boolean = false)
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Activity, Tensor[T], T] {

  private var internalInputSize: Array[Array[Int]] = _
  private var gradOutputSize: Array[Int] = _
  private var outputSize: Array[Int] = _
  private val timeBuffer =
    new ArrayBuffer[(AbstractModule[_ <: Activity, _ <: Activity, T], Long, Long)]
  private var maskBuffer: Tensor[T] = _
  private var indexBuffer: Tensor[T] = _
  private var inputBuffer: Tensor[T] = _

  /**
   * combine: [B, T, D] => [B * T, D]
   * split:   [B * T, D] => [B, T, D]
   */
  private def combine(src: Array[Int], target: Array[Int]): Unit = {
    require(src.length == target.length + 1,
      "TimeDistributed: combine method requires src.length == target.length + 1" +
        s" Current src.length = ${src.length}" +
        s" Current target.length = ${target.length}")

    target(0) = src(0) * src(1)
    var j = 1
    while (j < target.length) {
      target(j) = src(j + 1)
      j += 1
    }
  }

  private def resizeActivity(input: Activity, targetSizes: Array[Array[Int]]): Unit = {
    if (input.isTensor) input.toTensor.resize(targetSizes.head)
    else {
      input.toTable.foreach { case ((key: Int, value: Tensor[T])) =>
          value.resize(targetSizes(key - 1))
      }
    }
  }

  private def getActivitySize(input: Activity): Array[Array[Int]] = {
    if (input.isTensor) {
      Array(input.toTensor.size())
    } else {
      val sizes = new Array[Array[Int]](input.toTable.length)
      input.toTable.foreach { case ((key: Int, value: Tensor[T])) =>
        sizes(key - 1) = (value.size())
      }
      sizes
    }
  }

  override def updateOutput(input: Activity): Tensor[T] = {
    val oriSizes = getActivitySize(input)
    if (internalInputSize == null) {
      internalInputSize = new Array[Array[Int]](oriSizes.length)
      for (i <- 0 until oriSizes.length) {
        internalInputSize(i) = new Array[Int](oriSizes(i).length - 1)
      }
    }
    for ((srcSizes, tgtSizes) <- oriSizes zip internalInputSize) {
      combine(srcSizes, tgtSizes)
    }
    resizeActivity(input, internalInputSize)

    val _output = layer.forward(input).toTensor[T]

    val combinedShape = _output.size()
    // in case of singleton
    var i = 0
    while (i < oriSizes.length && oriSizes(i)(0) * oriSizes(i)(1) != combinedShape(0)) {
      i += 1
    }
    require(i < oriSizes.length,
      s"combined batch: ${combinedShape(0)} should match ${oriSizes(i)(0)} * ${oriSizes(i)(1)}")
    outputSize = Array(oriSizes(i)(0), oriSizes(i)(1)) ++ combinedShape.drop(1)

    resizeActivity(input, oriSizes)
    output.set(_output).resize(outputSize)

    if (maskZero) {
      require(input.isTensor, "only support mask with tensor")
      if (maskBuffer == null) {
        maskBuffer = Tensor()
      }
      if (indexBuffer == null) {
        indexBuffer = Tensor()
      }
      if (inputBuffer == null) {
        inputBuffer = Tensor()
      }
      inputBuffer.resizeAs(input.toTensor).abs(input.toTensor).max(maskBuffer, indexBuffer, 3)._1
      for (i <- 1 to maskBuffer.size(1)) {
        for (j <- 1 to maskBuffer.size(2)) {
          if (maskBuffer(Array(i, j, 1)) == ev.zero) {
            output.select(1, i).select(1, j).zero()
          }
        }
      }
    }

    output
  }

  override def updateGradInput(input: Activity, gradOutput: Tensor[T]): Activity = {
    if (gradOutputSize == null) {
      gradOutputSize = new Array[Int](gradOutput.size.length - 1)
    }
    val oriSizes = getActivitySize(input)
    val _gradOutputSize = gradOutput.size
    combine(_gradOutputSize, gradOutputSize)
    resizeActivity(input, internalInputSize)
    gradOutput.resize(gradOutputSize)
    val _gradInput = layer.updateGradInput(input, gradOutput)
    if (_gradInput.isTensor) {
      gradInput = Tensor()
      gradInput.toTensor.set(_gradInput.toTensor)
    } else {
      gradInput = T()
      var i = 1
      while (i <= _gradInput.toTable.length()) {
        gradInput.toTable.insert(_gradInput.toTable[Tensor[T]](i))
        i += 1
      }
    }

    resizeActivity(gradInput, oriSizes)
    resizeActivity(input, oriSizes)
    gradOutput.resize(_gradOutputSize)

    if (maskZero) {
      require(gradInput.isTensor, "only support mask with Tensor")
      for (i <- 1 to maskBuffer.size(1)) {
        for (j <- 1 to maskBuffer.size(2)) {
          if (maskBuffer(Array(i, j, 1)) == ev.zero) {
            gradInput.toTensor.select(1, i).select(1, j).zero()
          }
        }
      }
    }
    gradInput
  }

  override def accGradParameters(input: Activity, gradOutput: Tensor[T]): Unit = {
    val oriSizes = getActivitySize(input)
    val _gradOutputSize = gradOutput.size
    resizeActivity(input, internalInputSize)
    gradOutput.resize(gradOutputSize)
    layer.accGradParameters(input, gradOutput)
    resizeActivity(input, oriSizes)
    gradOutput.resize(_gradOutputSize)
  }

  override def backward(input: Activity, gradOutput: Tensor[T]): Activity = {
    val st = System.nanoTime
    updateGradInput(input, gradOutput)
    accGradParameters(input, gradOutput)
    backwardTime += System.nanoTime - st
    gradInput
  }

  override def reset(): Unit = layer.reset()

  override def training(): InternalTimeDistributed.this.type = {
    layer.training()
    super.training()
  }

  override def resetTimes(): Unit = {
    layer.resetTimes()
    this.forwardTime = 0
    this.backwardTime = 0
  }

  override def getTimes(): Array[(AbstractModule[_ <: Activity, _ <: Activity, T], Long, Long)] = {
    timeBuffer.clear
    var modulesForwardTime = 0L
    var modulesBackwardTime = 0L
    layer.getTimes.foreach(x => {
      timeBuffer.append(x)
      modulesForwardTime += x._2
      modulesBackwardTime += x._3
    })
    timeBuffer.append((this,
      this.forwardTime - modulesForwardTime,
      this.backwardTime - modulesBackwardTime))
    timeBuffer.toArray
  }

  override def evaluate(): InternalTimeDistributed.this.type = {
    layer.evaluate()
    super.evaluate()
  }

  /**
   * This function returns two arrays. One for the weights and the other the gradients
   * Custom modules should override this function if they have parameters
   *
   * @return (Array of weights, Array of grad)
   */
  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = layer.parameters()

  /**
   * This method will return a table indicating the name and corresponding parameters.
   * @return Table
   */
  override def getParametersTable(): Table = layer.getParametersTable()

  override def getExtraParameter(): Array[Tensor[T]] = {
    layer.getExtraParameter()
  }

  override def clearState(): InternalTimeDistributed.this.type = {
    super.clearState()
    layer.clearState()
    internalInputSize = null
    gradOutputSize = null
    outputSize = null
    timeBuffer.clear
    maskBuffer = null
    inputBuffer = null
    indexBuffer = null
    this
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[InternalTimeDistributed[T]]

  override def equals(other: Any): Boolean = other match {
    case that: InternalTimeDistributed[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        layer.equals(layer) &&
        internalInputSize == that.internalInputSize &&
        gradOutputSize == that.gradOutputSize &&
        outputSize == that.outputSize
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(),
      layer, internalInputSize, gradOutputSize, outputSize)
    state.filter(_ != null).map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }

  override def toString(): String = s"${getPrintName}${layer}"
}

object InternalTimeDistributed extends ModuleSerializable {
  def apply[@specialized(Float, Double) T: ClassTag](
      layer: AbstractModule[Activity, Tensor[T], T],
      maskZero: Boolean = false
  )(implicit ev: TensorNumeric[T]): InternalTimeDistributed[T] = {
    new InternalTimeDistributed[T](layer, maskZero)
  }

  // To make ti compatible with release 0.4
  override def doLoadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {
    val attrMap = context.bigdlModule.getAttrMap
    val layerAttr = attrMap.get("layer")
    val layer = DataConverter.getAttributeValue(context, layerAttr).
      asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
    var maskZero = false
    if (attrMap.containsKey("maskZero")) {
      maskZero = DataConverter.getAttributeValue(context, attrMap.get("maskZero")).
        asInstanceOf[Boolean]
    }
    InternalTimeDistributed(layer, maskZero)
  }
}
