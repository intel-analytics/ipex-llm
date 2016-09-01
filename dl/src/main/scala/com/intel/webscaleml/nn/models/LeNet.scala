package com.intel.webscaleml.nn.models

import com.intel.webscaleml.nn.nn.{Linear, LogSoftMax, SpatialMaxPooling, _}
import com.intel.webscaleml.nn.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

object LeNet5 {
  def apply[T : ClassTag](classNum : Int)(implicit ev: TensorNumeric[T]) : Module[T] = {
    val model = new Sequential[T]()
    model.add(new Reshape[T](Array(1, 28, 28)))
    model.add(new SpatialConvolution[T](1,6,5,5))
    model.add(new Tanh[T]())
    model.add(new SpatialMaxPooling[T](2,2,2,2))
    model.add(new Tanh[T]())
    model.add(new SpatialConvolution[T](6,12,5,5))
    model.add(new SpatialMaxPooling[T](2,2,2,2))
    model.add(new Reshape[T](Array(12 * 4 * 4)))
    model.add(new Linear[T](12 * 4 * 4, 100))
    model.add(new Tanh[T]())
    model.add(new Linear[T](100, classNum))
    model.add(new LogSoftMax[T]())
    model
  }
}
