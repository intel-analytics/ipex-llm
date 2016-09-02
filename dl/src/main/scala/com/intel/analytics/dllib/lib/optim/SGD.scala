package com.intel.analytics.dllib.lib.optim

import com.intel.analytics.dllib.lib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.dllib.lib.tensor.{Table, torch, Tensor}

import scala.reflect.ClassTag

class SGD[@specialized(Float, Double) T:ClassTag](implicit ev: TensorNumeric[T]) extends OptimMethod[T] {

  override def optimize(feval: (Tensor[T]) => (T, Tensor[T]), x: Tensor[T], config: Table, state: Table = null)
      : (Tensor[T], Array[T]) = {

    val _state = if(state == null) config else state
    val lr = config.get[Double]("learningRate").getOrElse(1e-3)
    val lrd = config.get[Double]("learningRateDecay").getOrElse(0.0)
    val wd = config.get[Double]("weightDecay").getOrElse(0.0)
    val mom = config.get[Double]("momentum").getOrElse(0.0)
    val damp = config.get[Double]("dampening").getOrElse(mom)
    val nesterov = config.get[Boolean]("nesterov").getOrElse(false)
    val lrs = config.get[Tensor[T]]("learningRates").getOrElse(null)
    val wds = config.get[Tensor[T]]("weightDecays").getOrElse(null)
    val nevals = _state.get[Int]("evalCounter").getOrElse(0)

    require(!nesterov || (mom > 0 && damp == 0), "Nesterov momentum requires a momentum and zero dampening")

    var (fx,dfdx) = feval(x)

    if(wd != 0) {
      dfdx.add(ev.fromType[Double](wd), x)
    } else if(wds != null) {
      val decayParameters = _state.get[Tensor[T]]("decayParameters").getOrElse({
        val DP = torch.Tensor[T]().resizeAs(dfdx)
        _state("decayParameters") = DP
        DP
      })
      decayParameters.copy(wds).cmul(x)
      dfdx.add(decayParameters)
    }

    if(mom != 0) {
      val stateDFDX = _state.get[Tensor[T]]("dfdx") match {
        case None =>
          val DFDX = torch.Tensor[T]().resizeAs(dfdx).copy(dfdx)
          _state("dfdx") = DFDX
          DFDX
        case s : Some[Tensor[T]] => s.get.mul(ev.fromType[Double](mom)).add(ev.fromType[Double](1 - damp), dfdx)
      }

      if(nesterov) {
        dfdx.add(ev.fromType[Double](mom), stateDFDX)
      } else {
        dfdx = stateDFDX
      }
    }

    val clr = ev.fromType[Double](-lr / (1 + nevals * lrd))

    if(lrs != null) {
      val deltaParameters = _state.get[Tensor[T]]("deltaParameters").getOrElse({
          val deltaP = torch.Tensor[T]().resizeAs(dfdx)
          _state("deltaParameters") = deltaP
          deltaP
      })
      deltaParameters.copy(lrs).cmul(dfdx)
      x.add(clr, deltaParameters)
    } else {
      x.add(clr, dfdx)
    }

    _state("evalCounter") = nevals + 1

    (x, Array(fx))
  }
}
