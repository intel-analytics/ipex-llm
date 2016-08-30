package com.intel.webscaleml.nn.nn

import com.intel.webscaleml.nn.tensor.Tensor
import com.intel.webscaleml.nn.tensor.TensorNumericMath.TensorNumeric

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

private[nn] abstract class Container [@specialized(Float, Double) T: ClassTag](implicit ev: TensorNumeric[T]) extends Module[T] {

  def add(module: Module[T]): this.type = {
    modules += module
    this
  }

  override def zeroGradParameters(): Unit = {
    modules.foreach(_.zeroGradParameters())
  }

  override def updateParameters(learningRate: T): Unit = {
    modules.foreach(_.updateParameters(learningRate))
  }

  override def reset(): Unit ={
    modules.foreach(_.reset())
  }

  override def training() : this.type = {
    modules.foreach(_.training())
    this
  }

  override def evaluate(): this.type = {
    modules.foreach(_.evaluate())
    this
  }

  override def getTimes() : Array[(Module[T], Long, Long)] = {
    this.modules.map(_.getTimes()).flatten.toArray
  }

  override def resetTimes(): Unit ={
    modules.foreach(_.resetTimes())
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    val weights = new ArrayBuffer[Tensor[T]]()
    val gradWeights = new ArrayBuffer[Tensor[T]]()
    modules.foreach(m=>{
      val params = m.parameters()
      if(params != null) {
        params._1.foreach(weights += _)
        params._2.foreach(gradWeights += _)
      }
    })
    (weights.toArray, gradWeights.toArray)
  }

  override def findModel(paramOffset : Int, indexes : Array[Int]) : (Module[T], Int, Array[Int]) = {
    var offset = paramOffset
    var result : Module[T] = this
    var newIndexes = indexes
    var i = 0
    modules.foreach( m=> {
      if(result == this) {
        val r = m.findModel(offset, indexes ++ Array(i))
        if (r._2 <= 0) {
          result = r._1
          newIndexes = r._3
        }
        offset = r._2
        i += 1
      }
    })
    (result, offset, newIndexes)
  }
}
