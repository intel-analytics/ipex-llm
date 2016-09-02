package com.intel.analytics.dllib.lib.tensor

trait TensorFunc2[@specialized(Float, Double) T] { self =>
  def apply(v1: Array[T], v2: Int): Unit

  override def toString() = "<TensorFunction2>"
}

trait TensorFunc4[@specialized(Float, Double) T] { self =>
  def apply(v1: Array[T], v2: Int, v3: Array[T], v4: Int): Unit

  override def toString() = "<TensorFunction4>"
}

trait TensorFunc6[@specialized(Float, Double) T] { self =>
  def apply(v1: Array[T], v2: Int, v3: Array[T], v4: Int,
            v5: Array[T], v6: Int): Unit

  override def toString() = "<TensorFunction6>"
}