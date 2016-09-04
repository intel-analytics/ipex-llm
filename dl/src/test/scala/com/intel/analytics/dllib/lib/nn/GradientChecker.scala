package com.intel.analytics.dllib.lib.nn

import com.intel.analytics.dllib.lib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.dllib.lib.tensor.{torch, Tensor}

import scala.reflect.ClassTag

class GradientChecker(stepSize : Double, threshold : Double) {

  def checkLayer[T : ClassTag](layer : Module[T], input : Tensor[T], epsilon : Double = 0.001)
                   (implicit ev: TensorNumeric[T]) : Boolean = {
    val gradOutput = lossAndGradient(layer.updateOutput(input))._2
    val computedGrad = layer.updateGradInput(input, gradOutput)
    computedGrad.resize(Array(computedGrad.nElement()))

    val perturbation = torch.Tensor[T]()
    perturbation.set(input)
    perturbation.resize(input.nElement())
    var result = true
    var i = 1
    while(i <= input.nElement()) {
      val curValue = perturbation.valueAt(i)
      perturbation.setValue(i, ev.fromType(ev.toType[Double](curValue) + stepSize))
      val positiveLoss = lossAndGradient(layer.updateOutput(input))._1
      perturbation.setValue(i, ev.fromType(ev.toType[Double](curValue) - stepSize))
      val negativeLoss = lossAndGradient(layer.updateOutput(input))._1
      val estimatedGradient = (positiveLoss - negativeLoss) / stepSize / 2.0

      result = result & (math.abs(estimatedGradient - ev.toType[Double](computedGrad.valueAt(i))) < epsilon)
      perturbation.setValue(i, curValue)
      i += 1
    }

    result
  }

  def lossAndGradient[T : ClassTag](output : Tensor[T])(implicit ev: TensorNumeric[T]) : (Double, Tensor[T]) = {
    val gradOutput = torch.Tensor[T]().resizeAs(output).copy(output)
    var loss = 0.0
    gradOutput.apply1(a => {
      val aDouble = ev.toType[Double](a)
      loss += 0.5 * aDouble * aDouble
      a
    })
    (loss, gradOutput)
  }
}
