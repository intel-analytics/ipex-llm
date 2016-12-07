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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.Tensor

import scala.reflect.ClassTag
import scala.util.Random

class GradientChecker(stepSize: Double, threshold: Double = 1e-2) {

  private val maxPrNum = 50
  private var checkModel = false

  // add status for checking model, default or 0 means check maxPrNum points,
  // 1 means check all points, others mean the checked points number
  private val status = if ((null != System.getProperty("modelGradcheck")) &&
    (System.getProperty("modelGradcheck").toInt > 0)) {
    System.getProperty("modelGradcheck").toInt
  } else 0

  def setType(isModle: Boolean = false): this.type = {
    checkModel = isModle
    this
  }

  def checkLayer[T: ClassTag](
    layer: Module[T],
    input: Tensor[T],
    epsilon: Double = 0.001)
    (implicit ev: TensorNumeric[T]): Boolean = {
    val gradOutput = lossAndGradient(layer.updateOutput(input).toTensor[T])._2
    val computedGrad = layer.updateGradInput(input, gradOutput).toTensor[T]
    computedGrad.toTensor[Double].resize(Array(computedGrad.nElement()))

    val perturbation = Tensor[T]()
    perturbation.set(input)
    perturbation.resize(input.nElement())
    var result = true
    var j = 1
    var i = 1

    val length = if (checkModel) {
      status match {
        case 0 => math.min(maxPrNum, input.nElement())
        case 1 => input.nElement()
        case _ => math.min(status, input.nElement())
      }
    } else input.nElement()

    while (j <= length) {
      i = Random.nextInt(input.nElement()) + 1
      val curValue = perturbation.valueAt(i)
      perturbation.setValue(i, ev.fromType(ev.toType[Double](curValue) + stepSize))
      val positiveLoss = lossAndGradient(layer.updateOutput(input).toTensor[T])._1
      val positiveLoss = lossAndGradient(layer.forward(input))._1
      perturbation.setValue(i, ev.fromType(ev.toType[Double](curValue) - stepSize))
      val negativeLoss = lossAndGradient(layer.updateOutput(input).toTensor[T])._1
      val negativeLoss = lossAndGradient(layer.forward(input))._1
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

    result
  }


  def checkWeight[T: ClassTag](
                                layer: Module[T],
                                input: Tensor[T],
                                epsilon: Double = 0.001,
                                num: Int = -1)
                              (implicit ev: TensorNumeric[T]): Boolean = {
    // add status
    var status = System.getProperty("checkAllGradient").asInstanceOf[Int]

    var (weights, gradWeights) = layer.getParameters()
    val gradOutput = lossAndGradient(layer.forward(input))._2
    val computedGrad = layer.backward(input, gradOutput)

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
      status match {
        case 0 => math.min(maxPrNum, weights.nElement())
        case 1 => weights.nElement()
        case _ => math.min(status, weights.nElement())
      }
    } else weights.nElement()

    while (j <= length) {
      i = Random.nextInt(weights.nElement()) + 1
      val curValue = perturbation.valueAt(i)
      perturbation.setValue(i, ev.fromType(ev.toType[Double](curValue) + stepSize))
      val positiveLoss = lossAndGradient(layer.forward(input))._1

      perturbation.setValue(i, ev.fromType(ev.toType[Double](curValue) - stepSize))
      val negativeLoss = lossAndGradient(layer.forward(input))._1

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
