package com.intel.webscaleml.nn.nn

import com.intel.webscaleml.nn.tensor.TensorNumericMath.TensorNumeric
import org.apache.commons.lang3.SerializationUtils

import com.intel.webscaleml.nn.tensor.{torch, Tensor}

import scala.reflect.ClassTag

class Criterion[@specialized(Float, Double) T: ClassTag](implicit ev: TensorNumeric[T]) extends Serializable{
  var output: T = ev.fromType[Int](0)

  def forward(input: Tensor[T], target: Tensor[T]): T = {
    updateOutput(input, target)
  }

  def backward(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    updateGradInput(input, target)
  }

  def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    this.output
  }

  def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = torch.Tensor[T]()

  def cloneCriterion() : Criterion[T] = {
    SerializationUtils.clone(this)
  }
}
