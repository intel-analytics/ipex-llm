package com.intel.analytics.dllib.lib.nn

import com.intel.analytics.dllib.lib.tensor.{DenseTensorApply, Storage, Tensor}
import com.intel.analytics.dllib.lib.tensor.TensorNumericMath.TensorNumeric

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