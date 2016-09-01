package com.intel.webscaleml.nn.optim

import com.intel.webscaleml.nn.tensor.{Table, Tensor}

trait LineSearch[@specialized(Float, Double) T] {
  /**
   * A Line Search function
   *
   * @param opfunc a function (the objective) that takes a single input (X), the point of evaluation, and
   *               returns f(X) and df/dX
   * @param x initial point / starting location
   * @param t initial step size
   * @param d descent direction
   * @param f initial function value
   * @param g gradient at initial location
   * @param gtd directional derivative at starting location
   * @param options linesearch options
   * @return
   *         f : Double function value at x+t*d
   *         g : Tensor gradient value at x+t*d
   *         x : Tensor the next x (=x+t*d)
   *         t : Double the step length
   *         lsFuncEval : Int the number of function evaluations
   */
  def apply(
    opfunc : (Tensor[T]) => (T, Tensor[T]),
    x : Tensor[T],
    t : T,
    d : Tensor[T],
    f : T,
    g : Tensor[T],
    gtd : T,
    options : Table
  ) : (T, Tensor[T], Tensor[T], T, Int)
}
