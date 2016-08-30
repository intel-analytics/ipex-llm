package com.intel.webscaleml.nn.tensor

object TensorRandom {
  private val r = scala.util.Random
//
//  def randDouble() : Double = r.nextDouble()
//  def randFloat() : Float = r.nextFloat()
//
//  def randnDouble() : Double = r.nextGaussian()
//  def randnFloat() : Float = r.nextGaussian().toFloat

  def randInt(n : Int) : Int = r.nextInt(n)
}
