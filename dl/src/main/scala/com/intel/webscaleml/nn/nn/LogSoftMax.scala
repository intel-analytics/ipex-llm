package com.intel.webscaleml.nn.nn

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import com.intel.webscaleml.nn.tensor.TensorNumericMath.TensorNumeric

import scala.collection.mutable.ArrayBuffer
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.math.{exp, log, max}
import com.intel.webscaleml.nn.tensor.Tensor

import scala.reflect.ClassTag

class LogSoftMax[@specialized(Float, Double) T: ClassTag](implicit ev: TensorNumeric[T]) extends Module[T]{
  @transient
  private var results : Array[Future[Unit]] = null

  override def updateOutput(input:Tensor[T]):Tensor[T] = {
    require(input.dim() == 1 || input.dim() == 2, "vector or matrix expected")
    output.resizeAs(input)
    val (nframe, dim) = if(input.nDimension() == 1) (1, input.size(1)) else (input.size(1), input.size(2))

    if(nframe == 1) {
      updateOutputFrame(input, output)
    } else {
      if(results == null || results.length != nframe) {
        results = new Array[Future[Unit]](nframe)
      }
      var t = 1
      while (t <= nframe) {
        val _t = t
        results(_t - 1) = Future {
          updateOutputFrame(input.select(1, _t), output.select(1, _t))
        }
        t += 1
      }

      t = 0
      while (t < nframe) {
        Await.result(results(t), Duration.Inf)
        t += 1
      }
    }

    output
  }

  private def updateOutputFrame(in : Tensor[T], out : Tensor[T]) : Unit = {
    var logsum = ev.fromType[Int](0)
    val maxInput = in.max()
    in.apply1(v => {logsum = ev.plus(logsum, ev.exp(ev.minus(v, maxInput))); v})
    logsum = ev.plus(maxInput, ev.log(logsum))

    out.map(in, (outData, inData) => {ev.minus(inData, logsum)})
  }

  override def updateGradInput(input:Tensor[T], gradOutput:Tensor[T]):Tensor[T] ={
    require(output.nDimension() == 1 || output.nDimension() == 2, "vector or matrix expected")
    gradInput.resizeAs(input)
    val (nframe, dim) = if(output.nDimension() == 1) (1, output.size(1)) else (output.size(1), output.size(2))

    if(results == null || results.length != nframe) {
      results = new Array[Future[Unit]](nframe)
    }

    var t = 1
    while(t <= nframe) {
      val _t = t
      results(_t - 1) = Future {
        var sum = 0.0
        var d = 1
        while (d <= dim) {
          sum += ev.toType[Double](gradOutput.valueAt(_t, d))
          d += 1
        }

        d = 1
        while (d <= dim) {
          gradInput.setValue(_t, d, ev.minus(gradOutput.valueAt(_t, d),
            ev.times(ev.exp(output.valueAt(_t, d)), ev.fromType[Double](sum))))
          d += 1
        }
      }
      t += 1
    }

    t = 0
    while(t < nframe) {
      Await.result(results(t), Duration.Inf)
      t += 1
    }

    gradInput
  }

  override def toString() : String = {
    s"nn.LogSoftMax"
  }
}

object LogSoftMax {
  private val A0 = 1.0
  private val A1 = 0.125
  private val A2 = 0.0078125
  private val A3 = 0.00032552083
  private val A4 = 1.0172526e-5

  def expMinusApprox(x : Double) : Double = {
    //require(x >= 0.0)
    if(x < 0) {
      return exp(-x)
    } else {
      var y = 0.0
      if (x < 13.0) {
        y = A0 + x * (A1 + x * (A2 + x * (A3 + x * A4)))
        y *= y
        y *= y
        y *= y
        y = 1 / y
        return y
      }
    }

    return 0.0
  }
}

