package com.intel.analytics.sparkdl.utils

import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect._
import scala.reflect.runtime.universe._

trait Activities {
  def toTensor[T](): Tensor[T] = {
      this.asInstanceOf[Tensor[T]]
  }

  def toTable(): Table = {
    this.asInstanceOf[Table]
  }
}

object Activities {
  def apply[A <: Activities: ClassTag, @specialized(Float, Double) T: ClassTag]()(
    implicit ev: TensorNumeric[T]): Activities = {
    var result:Activities = null

    if (classTag[A] == classTag[Tensor[T]])
      result = Tensor[T]()
    else if (classTag[A] == classTag[Tensor[T]])
      result = T()

    result
  }
}
