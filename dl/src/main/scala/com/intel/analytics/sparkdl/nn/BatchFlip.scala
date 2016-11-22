package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.utils.Engine

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.reflect.ClassTag

class BatchFlip[T: ClassTag](implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  @transient
  private var results: Array[Future[Unit]] = null

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output = input
    if (train) {
      val bs = input.size(1)
      val flipMask = Tensor.randperm[Float](bs)
      val flipMaskArray = flipMask.storage().array()
      if (results == null || results.length != bs) {
        results = new Array[Future[Unit]](bs)
      }
      var i = 1
      while (i <= bs) {
        val _i = i
        results(_i - 1) = Future[Unit] {
          if (flipMaskArray(_i - 1) <= bs / 2) {
            Tensor.hflip(input(_i), input(_i))
          }
        }(Engine.getInstance())
        i += 1
      }

      i = 0
      while (i < bs / 2) {
        Await.result(results(i), Duration.Inf)
        i += 1
      }
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput = gradOutput
    gradInput
  }

}