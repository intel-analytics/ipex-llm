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

  private val maxNum = 2000 //5000

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

    status = 1
    val length = if (status == 1) math.min(maxNum, input.nElement()) else input.nElement()
    while (j <= length) {
      i = if (status == 1) Random.nextInt(input.nElement()) + 1 else j
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
        println(errDiff + " " + estimatedGradient + " " + computedGrad.valueAt(i) )
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

    val gradOutput = lossAndGradient(layer.forward(input))._2
    val computedGrad = layer.backward(input, gradOutput)

    val (weights, gradWeights) = layer.getParameters()
    gradWeights.resize(Array(gradWeights.nElement()))

    val perturbation = Tensor[T]()
    perturbation.set(weights)
    perturbation.resize(weights.nElement())
    var result = true
    var j = 1
    var i = 1

    status = 1
    val length = if (status == 1) math.min(maxNum, weights.nElement()) else weights.nElement()

    while (j <= length) {
      i = 4606088 //if (status == 1) Random.nextInt(weights.nElement()) + 1 else j
      val curValue = perturbation.valueAt(i)
      perturbation.setValue(i, ev.fromType(ev.toType[Double](curValue) + stepSize))
      val positiveLoss = lossAndGradient(layer.forward(input))._1
      perturbation.setValue(i, ev.fromType(ev.toType[Double](curValue) - stepSize))
      val negativeLoss = lossAndGradient(layer.forward(input))._1
      val estimatedGradient = (positiveLoss - negativeLoss) / stepSize / 2.0

      val errDiff = math.abs(estimatedGradient - ev.toType[Double](gradWeights.valueAt(i)))
      if (errDiff > epsilon) {
        println(errDiff + " " + estimatedGradient + " " + gradWeights.valueAt(i) )
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
