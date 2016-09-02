package com.intel.analytics.dllib.lib.optim

import com.intel.analytics.dllib.lib.tensor.{torch, Tensor}
import com.intel.analytics.dllib.lib.tensor.TensorNumericMath.TensorNumeric

object EvaluateMethods {
  def calcAccuracy[@specialized(Float, Double) T](output : Tensor[T], target : Tensor[T]) : (Int, Int) = {
    val indexer = Array(1)
    var correct = 0
    var count = 0
    if (output.dim() == 2) {
      output.max(2)._2.squeeze().map(target, (a, b) => {
        if(a == b) {
          correct += 1
        }
        a
      })
      count += output.size(1)
    } else if (output.dim == 1) {
      require(target.size(1) == 1)
      output.max(1)._2.map(target, (a, b) => {
        if(a == b) {
          correct += 1
        }
        a
      })
      count += 1
    } else {
      throw new IllegalArgumentException
    }

    (correct, count)
  }

  //for sparse LR use only
  def calcLrAccuracy[@specialized(Float, Double) T](ev: TensorNumeric[T])(output : Tensor[T], target : Tensor[T]) : (Int, Int) = {
    var correct = 0
    var count = 0
    require(output.dim == 1 && output.size(1) == target.size(1))
    output.map(target, (a, b) => {
      if(Math.round(ev.toType[Double](a)) == b){
        correct += 1
      }
      a
    })
    count += output.size(1)

    (correct, count)
  }
}
