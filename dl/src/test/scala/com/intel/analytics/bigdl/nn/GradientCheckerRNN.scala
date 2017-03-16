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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.optim.SGD
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T

import scala.reflect.ClassTag

@com.intel.analytics.bigdl.tags.Parallel
class GradientCheckerRNN(stepSize: Double = 0.01, threshold: Double = 0.01) {

  def checkLayer[T: ClassTag](
    layer: Module[Double],
    input: Tensor[Double],
    label: Tensor[Double],
    epsilon: Double = 0.01): Boolean = {

    val criterion = new CrossEntropyCriterion[Double]()
    val (weights, grad) = layer.getParameters()

    val state = T("learningRate" -> 0.05, "momentum" -> 0.0, "weightDecay" -> 0.0,
      "dampening" -> 0.0)
    val sgd = new SGD[Double]
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      layer.forward(input)
      criterion.forward(layer.output.asInstanceOf[Tensor[Double]], label)
      layer.zeroGradParameters()
      val gradOutputTest = criterion.backward(layer.output.asInstanceOf[Tensor[Double]], label)
      layer.backward(input, gradOutputTest)
      (criterion.output, grad)
    }
    sgd.optimize(feval, weights, state)



    for (i <- 1 to grad.size(1)) {
      var originalValue = weights.valueAt(i)
      weights.setValue(i, originalValue + stepSize)
      layer.forward(input)
      criterion.forward(layer.output.asInstanceOf[Tensor[Double]], label)
      var gradPlus = criterion.output
      weights.setValue(i, originalValue - stepSize)
      layer.forward(input)
      criterion.forward(layer.output.asInstanceOf[Tensor[Double]], label)
      var gradMinus = criterion.output
      var estimatedGradient = (gradPlus - gradMinus) / (2*stepSize)
      weights.setValue(i, originalValue)
      var backpropGradient = grad.valueAt(i)
      var relativeError = if ((Math.abs(backpropGradient) + Math.abs(estimatedGradient)) == 0) 0
          else {Math.abs(backpropGradient-estimatedGradient) /
        (Math.abs(backpropGradient) + Math.abs(estimatedGradient))}

      println(s"parameter ${i}, EstimatedGradient = ${estimatedGradient}, " +
        s"BackpropGradient = ${backpropGradient}," +
        s"RelativeError = ${relativeError}")
    }
    false
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
