package com.intel.analytics.dllib.lib.optim

import com.intel.analytics.dllib.lib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.dllib.lib.tensor.{T, Table, torch, Tensor}

import scala.reflect.ClassTag

class Adagrad[@specialized(Float, Double) T:ClassTag](implicit ev: TensorNumeric[T]) extends OptimMethod[T] {

  /**
   * Adagrad implementation for SGD
   *
   * @param feval a function that takes a single input (X), the point of a evaluation, and returns f(X) and df/dX
   * @param parameter the initial point
   * @param config a table with configuration parameters for the optimizer
   *               config("learningRate") : learning rate
   *               config("learningRateDecay") : learning rate decay
   *
   * @param state a table describing the state of the optimizer; after each call the state is modified
   *               state("paramVariance") : vector of temporal variances of paramters
   * @return the new x vector and the function list, evaluated before the update
   */
  override def optimize(feval: (Tensor[T]) => (T, Tensor[T]), parameter: Tensor[T], config: Table, state: Table)
    : (Tensor[T], Array[T]) = {

    val _config = if(config == null) T() else config
    val _state = if(state == null) _config else state

    val lr = _config.get[Double]("learningRate").getOrElse(1e-3)
    val lrd = _config.get[Double]("learningRateDecay").getOrElse(0.0)
    val nevals = _state.get[Int]("evalCounter").getOrElse(0)

    val (fx, dfdx) = feval(parameter)

    val clr = lr / (1 + nevals * lrd)

    val (_paramVariance, _paramStd) =
      if(_state.get[Tensor[T]]("paramVariance").isDefined) {
        (_state.get[Tensor[T]]("paramVariance").get, _state.get[Tensor[T]]("paramStd").get)
      } else {
        (torch.Tensor[T]().resizeAs(dfdx).zero(), torch.Tensor[T]().resizeAs(dfdx))
      }

    _paramVariance.addcmul(ev.fromType[Int](1), dfdx, dfdx)
    _paramStd.resizeAs(_paramVariance).copy(_paramVariance).sqrt()
    parameter.addcdiv(ev.fromType[Double](-clr), dfdx, _paramStd.add(ev.fromType[Double](1e-10)))

    _state("evalCounter") = nevals + 1
    _state("paramVariance") = _paramVariance
    _state("paramStd") = _paramStd

    (parameter, Array(fx))
  }
}
