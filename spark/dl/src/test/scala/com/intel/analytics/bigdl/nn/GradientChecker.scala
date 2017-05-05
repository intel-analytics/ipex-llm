/*
 * Copyright 2016 The BigDL Authors.
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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.TensorCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.commons.lang.StringUtils

import scala.reflect.ClassTag
import scala.util.Random

sealed trait modelCheck
case class PartCheck(data : Int) extends modelCheck
case class FullCheck() extends modelCheck

@com.intel.analytics.bigdl.tags.Parallel
class GradientChecker(stepSize: Double, threshold: Double = 1e-2) {

  private val defaultNum = 50
  private var checkModel = false

  // status for checking model, only support positive integer and "all"
  // all means check all points, others mean the checked points number
  private val status = System.getProperty("modelCheck")
  private val modelCheckType = if (status == null) {
    PartCheck(defaultNum)
  } else if (status == "all") {
    FullCheck()
  } else if (StringUtils.isNumeric(status)) {
    PartCheck(status.toInt)
  } else {
    throw new IllegalArgumentException(s"input wrong check number ${status}")
  }

  def setType(isModel: Boolean = false): this.type = {
    checkModel = isModel
    this
  }

  def checkLayer[T: ClassTag](
    layer: Module[T],
    input: Tensor[T],
    epsilon: Double = 0.001)
  (implicit ev: TensorNumeric[T]): Boolean = {

    val gradOutput = lossAndGradient(layer.forward(input).toTensor[T])._2
    val computedGrad = layer.updateGradInput(input, gradOutput).toTensor[T]
    computedGrad.resize(Array(computedGrad.nElement()))

    val perturbation = Tensor[T]()
    perturbation.set(input)
    perturbation.resize(input.nElement())
    var result = true
    var j = 1
    var i = 1

    val length = if (checkModel) {
      modelCheckType match {
        case FullCheck() => input.nElement()
        case PartCheck(n) => n
      }
    } else input.nElement()

    var scalaTime: Long = 0
    while (j <= length) {
      i = Random.nextInt(input.nElement()) + 1
      val curValue = perturbation.valueAt(i)
      perturbation.setValue(i, ev.fromType(ev.toType[Double](curValue) + stepSize))
      val positiveLoss = lossAndGradient(layer.forward(input).toTensor[T])._1
      perturbation.setValue(i, ev.fromType(ev.toType[Double](curValue) - stepSize))
      val start = System.nanoTime()
      val negativeLoss = lossAndGradient(layer.forward(input).toTensor[T])._1
      scalaTime = System.nanoTime() - start
      val estimatedGradient = (positiveLoss - negativeLoss) / stepSize / 2.0

      val errDiff = math.abs(estimatedGradient - ev.toType[Double](computedGrad.valueAt(i)))
      if (errDiff > epsilon) {
        println("input: greater " + i + ":" + errDiff + " " +
          estimatedGradient + " " + computedGrad.valueAt(i))
      }
      result = result & (errDiff < epsilon)
      perturbation.setValue(i, curValue)

      j += 1
    }
    println("forward time: " + scalaTime / 1e9 + " s")
    result
  }


  def checkCriterion[T: ClassTag](
      criterion: TensorCriterion[T],
      input: Tensor[T],
      target: Tensor[T],
      epsilon: Double = 0.001)
      (implicit ev: TensorNumeric[T]): Boolean = {

    val loss = criterion.forward(input, target)
    val gradOutput = criterion.backward(input, target)
    val computedGrad = gradOutput
    computedGrad.resize(Array(computedGrad.nElement()))

    val perturbation = Tensor[T]()
    perturbation.set(input)
    perturbation.resize(input.nElement())
    var result = true
    var j = 1
    var i = 1

    val length = if (checkModel) {
      modelCheckType match {
        case FullCheck() => input.nElement()
        case PartCheck(n) => n
      }
    } else input.nElement()

    var scalaTime: Long = 0
    while (j <= length) {
      i = Random.nextInt(input.nElement()) + 1
      val curValue = perturbation.valueAt(i)
      perturbation.setValue(i, ev.fromType(ev.toType[Double](curValue) + stepSize))
      val positiveLoss = criterion.forward(input, target)
      // val positiveLoss = lossAndGradient(layer.forward(input).toTensor[T])._1
      perturbation.setValue(i, ev.fromType(ev.toType[Double](curValue) - stepSize))
      val start = System.nanoTime()
      val negativeLoss = criterion.forward(input, target)
      // val negativeLoss = lossAndGradient(layer.forward(input).toTensor[T])._1
      scalaTime = System.nanoTime() - start
      val estimatedGradient =
        0.5 * ev.toType[Double](ev.divide(
          ev.minus(positiveLoss, negativeLoss), ev.fromType[Double](stepSize)))

      val errDiff = math.abs(estimatedGradient - ev.toType[Double](computedGrad.valueAt(i)))
      if (errDiff > epsilon) {
        println("input: greater " + i + ": " + errDiff + " " +
          estimatedGradient + " " + computedGrad.valueAt(i))
      }
      result = result & (errDiff < epsilon)
      perturbation.setValue(i, curValue)

      j += 1
    }
    println("forward time: " + scalaTime / 1e9 + " s")
    result
  }

  def checkWeight[T: ClassTag](
    layer: Module[T],
    input: Tensor[T],
    epsilon: Double = 0.001,
    num: Int = -1)
  (implicit ev: TensorNumeric[T]): Boolean = {

    var (weights, gradWeights) = layer.getParameters()
    val gradOutput = lossAndGradient(layer.forward(input).toTensor[T])._2
    val computedGrad = layer.backward(input, gradOutput).toTensor[T]

    weights = layer.getParameters()._1
    gradWeights = layer.getParameters()._2
    gradWeights.resize(Array(gradWeights.nElement()))

    val perturbation = Tensor[T]()
    perturbation.set(weights)
    perturbation.resize(weights.nElement())
    var result = true
    var j = 1
    var i = 1

    val length = if (checkModel) {
      modelCheckType match {
        case FullCheck() => weights.nElement()
        case PartCheck(n) => n
      }
    } else weights.nElement()

    while (j <= length) {
      i = Random.nextInt(weights.nElement()) + 1
      val curValue = perturbation.valueAt(i)
      perturbation.setValue(i, ev.fromType(ev.toType[Double](curValue) + stepSize))
      val positiveLoss = lossAndGradient(layer.forward(input).toTensor[T])._1

      perturbation.setValue(i, ev.fromType(ev.toType[Double](curValue) - stepSize))
      val negativeLoss = lossAndGradient(layer.forward(input).toTensor[T])._1

      val estimatedGradient = (positiveLoss - negativeLoss) / stepSize / 2.0
      val errDiff = math.abs(estimatedGradient - ev.toType[Double](gradWeights.valueAt(i)))

      if (errDiff > epsilon) {
        println("weight: greater " + i + ":" + errDiff + " " +
          estimatedGradient + " " + gradWeights.valueAt(i))
      }

      result = result & (errDiff < epsilon)
      perturbation.setValue(i, curValue)
      j += 1
    }
    result
  }

  def lossAndGradient[T: ClassTag](output: Tensor[T])(
    implicit ev: TensorNumeric[T]): (Double, Tensor[T]) = {
    val gradOutput = Tensor[T]().resizeAs(output).copy(output)
    var loss = 0.0
    gradOutput.apply1(a => {
      val aDouble = ev.toType[Double](a)
      loss += 0.5 * aDouble * aDouble
      a
    })
    (loss, gradOutput)
  }
}
