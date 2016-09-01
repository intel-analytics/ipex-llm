package com.intel.webscaleml.nn.nn

import com.intel.webscaleml.nn.tensor.TensorNumericMath.TensorNumeric
import com.intel.webscaleml.nn.tensor.{torch, Tensor}
import com.intel.webscaleml.nn.tensor.RandomGenerator._

import scala.concurrent.duration.{Duration, DurationConversions}
import scala.concurrent.{Await, Future}
import scala.reflect.ClassTag

class Dropout[@specialized(Float, Double) T: ClassTag](val initP : Double = 0.5, val inplace : Boolean = false,
    var scale : Boolean = true)(implicit ev: TensorNumeric[T]) extends Module[T] {
  private var p = initP
  var noise = torch.Tensor[T]()
  protected val results = new Array[Future[Unit]](coresNum)

  def getP(): T ={
    return ev.fromType[Double](p)
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if(inplace) {
      this.output = input
    } else {
      this.output.resizeAs(input).copy(input)
    }

    if(train) {
      noise.resizeAs(input)
      //require(input.isContiguous())
      if(input.isContiguous()) {
        val noiseData = noise.storage().array()
        var taskSize = noise.nElement() / coresNum
        var extraTask = noise.nElement() % coresNum
        var allocated = 0
        val offset = this.output.storageOffset() - 1
        val data = this.output.storage.array()
        var i = 0
        while (allocated < noise.nElement()) {
          val start = allocated
          allocated += taskSize
          if (extraTask > 0) {
            allocated += 1
            extraTask -= 1
          }
          val end = allocated
          results(i) = Future {
            var k = start
            while (k < end) {
              noiseData(k) = if (RNG.bernoulli(1 - p)) {
                if (scale) {
                  data(offset + k) = ev.divide(data(offset + k), ev.fromType[Double](1 - p))
                  ev.fromType[Double](1.0 / (1 - p))
                } else {
                  ev.fromType[Int](1)
                }
              } else {
                data(offset + k) = ev.fromType[Int](0)
                ev.fromType[Int](0)
              }

              k += 1
            }
          }
          i += 1
        }

        val allocatedTask = i
        i = 0
        while (i < allocatedTask) {
          Await.result(results(i), Duration.Inf)
          i += 1
        }
        this.output
      } else {
        noise.bernoulli(1 - p)

        if (scale) {
          noise.div(ev.fromType[Double](1 - p))
        }
        this.output.cmul(noise)
      }
    } else if(!scale){
      this.output.mul(ev.fromType[Double](1 - p))
    } else {
      output
    }
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if(train) {
      if(inplace) {
        this.gradInput = gradOutput
      } else {
        this.gradInput.resizeAs(gradOutput).copy(gradOutput)
      }

      //require(gradInput.isContiguous())
      if(gradInput.isContiguous()) {
        val noiseData = noise.storage().array()
        var taskSize = noise.nElement() / coresNum
        var extraTask = noise.nElement() % coresNum
        val gradInputData = gradInput.storage().array()
        val gradInputOffset = gradInput.storageOffset() - 1
        var allocated = 0
        var i = 0
        while (allocated < noise.nElement()) {
          val start = allocated
          allocated += taskSize
          if (extraTask > 0) {
            allocated += 1
            extraTask -= 1
          }
          val end = allocated
          results(i) = Future {
            var k = start
            while (k < end) {
              gradInputData(gradInputOffset + k) = ev.times(gradInputData(gradInputOffset + k), noiseData(k))
              k += 1
            }
          }
          i += 1
        }

        val allocatedTask = i
        i = 0
        while (i < allocatedTask) {
          Await.result(results(i), Duration.Inf)
          i += 1
        }

        this.gradInput
      } else {
        this.gradInput.cmul(noise)
      }
    } else {
      throw new IllegalArgumentException("backprop only defined while training")
    }

    this.gradInput
  }

  def setP(p : Double) = {
    this.p = p
  }

  override def toString() : String = {
    s"nn.Dropout($p)"
  }
}
