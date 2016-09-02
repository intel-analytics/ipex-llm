package com.intel.analytics.dllib.lib.optim

import com.intel.analytics.dllib.lib.tensor.{Table, Tensor}

/**
 * Similar to torch Optim method, which is used to update the parameter
 */
trait OptimMethod[@specialized(Float, Double) T] extends Serializable {
  /**
   * Optimize the model parameter
   *
   * @param feval a function that takes a single input (X), the point of a evaluation, and returns f(X) and df/dX
   * @param parameter the initial point
   * @param config a table with configuration parameters for the optimizer
   * @param state a table describing the state of the optimizer; after each call the state is modified
   * @return the new x vector and the function list, evaluated before the update
   */
  def optimize(feval : (Tensor[T]) => (T, Tensor[T]), parameter : Tensor[T], config : Table, state : Table = null) : (Tensor[T], Array[T])
}

trait IterateByItself

trait FullBatchOptimMethod
