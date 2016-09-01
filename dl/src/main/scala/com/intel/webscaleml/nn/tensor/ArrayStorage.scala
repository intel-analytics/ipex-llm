package com.intel.webscaleml.nn.tensor

import java.util


import com.intel.webscaleml.nn.tensor.TensorType.{FloatType, DoubleType, DataType}

import scala.reflect.ClassTag

private[tensor] class ArrayStorage[@specialized(Double, Float) T : ClassTag](private[tensor] var values : Array[T]) extends Storage[T] {

  override def apply(index : Int) : T = values(index)
  override def update(index : Int, value : T) = values(index) = value
  override def length() : Int = values.length
  override def iterator: Iterator[T] = values.iterator
  override def array() : Array[T] = values
  override def copy(source: Storage[T], offset : Int, sourceOffset : Int, length : Int): this.type = {
    source match {
      case s : ArrayStorage[T] => System.arraycopy(s.values, sourceOffset, this.values, offset, length)
      case s : Storage[T] => {
        var i = 0
        while(i < length) {
          this.values(i + offset) = s(i + sourceOffset)
          i += 1
        }
      }
    }
    this
  }

  override def resize(size: Long): this.type = {
    values = new Array[T](size.toInt)
    this
  }
  override def fill(value: T, offset : Int, length : Int): this.type = {

    value match {
      case v : Double => util.Arrays.fill(values.asInstanceOf[Array[Double]], offset - 1, offset - 1 + length, v)
      case v : Float => util.Arrays.fill(values.asInstanceOf[Array[Float]], offset - 1, offset - 1 + length, v)
      case _ => ???
    }

    this
  }

  override def set(other: Storage[T]): this.type = {
    require(other.length() == this.length())
    this.values = other.array
    this
  }
}