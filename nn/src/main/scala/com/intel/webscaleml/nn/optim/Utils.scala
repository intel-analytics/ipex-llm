package com.intel.webscaleml.nn.optim

import java.nio.ByteBuffer

import com.intel.webscaleml.nn.tensor.{torch, Tensor}

import scala.util.Random

object Utils {
  def inPlaceShuffle[T](data : Array[T]) : Array[T] = {
    var i = 0
    val rand = new Random(System.nanoTime())
    val length = data.length
    while(i < length) {
      val exchange = rand.nextInt(length - i) + i
      val tmp = data(exchange)
      data(exchange) = data(i)
      data(i) = tmp
      i += 1
    }
    data
  }
}
