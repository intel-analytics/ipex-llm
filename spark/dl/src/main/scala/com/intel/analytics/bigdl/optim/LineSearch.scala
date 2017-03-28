/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Table

/**
 * Line Search strategy
 */
trait LineSearch[@specialized(Float, Double) T] {
  /**
   * A Line Search function
   *
   * @param opfunc  a function (the objective) that takes a single input (X),
   *                the point of evaluation, and
   *                returns f(X) and df/dX
   * @param x       initial point / starting location
   * @param t       initial step size
   * @param d       descent direction
   * @param f       initial function value
   * @param g       gradient at initial location
   * @param gtd     directional derivative at starting location
   * @param options linesearch options
   * @return
   * f : Double function value at x+t*d
   * g : Tensor gradient value at x+t*d
   * x : Tensor the next x (=x+t*d)
   * t : Double the step length
   * lsFuncEval : Int the number of function evaluations
   */
  def apply(
    opfunc: (Tensor[T]) => (T, Tensor[T]),
    x: Tensor[T],
    t: T,
    d: Tensor[T],
    f: T,
    g: Tensor[T],
    gtd: T,
    options: Table
  ): (T, Tensor[T], Tensor[T], T, Int)
}
