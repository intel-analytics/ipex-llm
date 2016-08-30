package com.intel.webscaleml.nn.nn

import com.intel.webscaleml.nn.tensor.{DenseTensorApply, Storage, Tensor}
import com.intel.webscaleml.nn.tensor.TensorNumericMath.TensorNumeric

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.reflect.ClassTag

class ReLU[@specialized(Float, Double) T: ClassTag](ip:Boolean = false)(implicit ev: TensorNumeric[T]) extends Threshold[T](0,0,ip){
  override def toString() : String = {
    s"nn.ReLU"
  }
}