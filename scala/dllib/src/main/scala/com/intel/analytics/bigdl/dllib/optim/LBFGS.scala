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

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * This implementation of L-BFGS relies on a user-provided line
 * search function (state.lineSearch). If this function is not
 * provided, then a simple learningRate is used to produce fixed
 * size steps. Fixed size steps are much less costly than line
 * searches, and can be useful for stochastic problems.
 *
 * The learning rate is used even when a line search is provided.
 * This is also useful for large-scale stochastic problems, where
 * opfunc is a noisy approximation of f(x). In that case, the learning
 * rate allows a reduction of confidence in the step size.
 * @param maxIter Maximum number of iterations allowed
 * @param maxEval Maximum number of function evaluations
 * @param tolFun Termination tolerance on the first-order optimality
 * @param tolX Termination tol on progress in terms of func/param changes
 * @param nCorrection
 * @param learningRate
 * @param verbose
 * @param lineSearch A line search function
 * @param lineSearchOptions If no line search provided,
 *               then a fixed step size is used
 */
class LBFGS[@specialized(Float, Double) T: ClassTag](
    var maxIter: Int = 20,
    var maxEval: Double = Double.MaxValue,
    var tolFun: Double = 1e-5,
    var tolX: Double = 1e-9,
    var nCorrection: Int = 100,
    var learningRate: Double = 1.0,
    var verbose: Boolean = false,
    var lineSearch: Option[LineSearch[T]] = None,
    var lineSearchOptions: Option[Table] = None
  )(implicit ev: TensorNumeric[T]) extends OptimMethod[T] {

  /**
   * Optimize the model parameter
   *
   * @param opfunc a function that takes a single input (X), the point of a evaluation,
   *               and returns f(X) and df/dX
   * @param x      the initial point
   * @return the new x vector and the evaluate value list, evaluated before the update
   *         x : the new `x` vector, at the optimal point
   *         f : a table of all function values:
   *         `f[1]` is the value of the function before any optimization and
   *         `f[#f]` is the final fully optimized value, at `x*`
   */
  override def optimize(opfunc: (Tensor[T]) => (T, Tensor[T]), x: Tensor[T]
                       ): (Tensor[T], Array[T]) = {
    if (this.maxEval == Double.MaxValue) this.maxEval = 1.25 * this.maxIter
    val _state = state
    val maxIter = this.maxIter
    val maxEval = this.maxEval
    val tolFun = this.tolFun
    val tolX = this.tolX
    val nCorrection = this.nCorrection
    val learningRate = this.learningRate

    var funcEval = _state.get[Int]("funcEval").getOrElse(0)
    var nIter = _state.get[Int]("nIter").getOrElse(0)

    // evaluate initial f(x) and df/dx
    var (f, g) = opfunc(x)
    val f_hist = new ArrayBuffer[T]()
    f_hist.append(f)
    var currentFuncEval = 1
    funcEval += 1
    val p = g.size(1)

    // check optimality of initial point
    val tmp1 = _state.get[Tensor[T]]("tmp1").getOrElse(Tensor[T]().resizeAs(g))
    tmp1.copy(g).abs()
    if (ev.toType[Double](tmp1.sum()) < tolFun) {
      verbose("optimality condition below tolFun")
      return (x, f_hist.toArray)
    }

    val (dir_bufs, stp_bufs) =
      if (!_state.get[Table]("dir_bufs").isDefined) {
        verbose("creating recyclable direction/step/history buffers")
        val d = Tensor[T](nCorrection + 1, p).split(1)
        val s = Tensor[T](nCorrection + 1, p).split(1)
        var i = 0
        while (i < d.length) {
          d(i) = d(i).squeeze(1)
          i += 1
        }
        val dObjs = new Array[Any](d.length)
        System.arraycopy(d, 0, dObjs, 0, d.length)
        val sObjs = new Array[Any](s.length)
        System.arraycopy(s, 0, sObjs, 0, s.length)
        (T array dObjs, T array sObjs)
      } else {
        (_state.get[Table]("dir_bufs").get, _state.get[Table]("stp_bufs").get)
      }

    val d = _state.get[Tensor[T]]("d").getOrElse(g.clone().mul(ev.fromType[Int](-1)))
    var t = _state.get[Double]("t").getOrElse(learningRate)
    var Hdiag = _state.get[Double]("Hdiag").getOrElse(1.0)
    val old_dirs = _state.get[Table]("old_dirs").getOrElse(T())
    val old_stps = _state.get[Table]("old_stps").getOrElse(T())
    val g_old = _state.get[Tensor[T]]("g_old").getOrElse(g.clone())
    var f_old = _state.get[T]("f_old").getOrElse(f)
    val ro = _state.get[Tensor[T]]("ro").getOrElse(Tensor[T](nCorrection))
    val al = _state.get[Tensor[T]]("al").getOrElse(Tensor[T](nCorrection))

    var _nIter = 0
    var isBreak = false
    while (_nIter < maxIter && !isBreak) {
      nIter += 1
      _nIter += 1

      // compute gradient descent direction
      if (nIter != 1) {
        val y = dir_bufs.remove[Tensor[T]]().get
        val s = stp_bufs.remove[Tensor[T]]().get
        y.add(g, ev.fromType[Int](-1), g_old)
        s.mul(d, ev.fromType[Double](t))
        val ys = y.dot(s)
        if (ev.toType[Double](ys) > 1e-10) {
          if (old_dirs.length() == nCorrection) {
            // shift history by one (limited-memory)
            val removed1 = old_dirs.remove[Tensor[T]](1).get
            val removed2 = old_stps.remove[Tensor[T]](1).get
            dir_bufs.insert(removed1)
            stp_bufs.insert(removed2)
          }

          old_dirs.insert(s)
          old_stps.insert(y)

          Hdiag = ev.toType[Double](ev.divide(ys, y.dot(y))) // (y*y)
        } else {
          dir_bufs.insert(y)
          stp_bufs.insert(s)
        }

        val k = old_dirs.length()
        var i = 1
        while (i <= k) {
          ro.setValue(i, ev.divide(ev.fromType[Int](1), old_stps[Tensor[T]](i).dot(old_dirs(i))))
          i += 1
        }

        // iteration in L-BFGS loop collapsed to use just one buffer
        // reuse tmp1 for the q buffer
        val q = tmp1
        q.mul(g, ev.fromType[Int](-1))
        i = k
        while (i >= 1) {
          al.setValue(i, ev.times(old_dirs[Tensor[T]](i).dot(q), ro.valueAt(i)))
          q.add(ev.negative(al.valueAt(i)), old_stps[Tensor[T]](i))
          i -= 1
        }

        val r = d
        r.mul(q, ev.fromType[Double](Hdiag))
        i = 1
        while (i <= k) {
          val be_i = ev.times(old_stps[Tensor[T]](i).dot(r), ro.valueAt(i))
          r.add(ev.minus(al.valueAt(i), be_i), old_dirs[Tensor[T]](i))
          i += 1
        }
      }
      g_old.copy(g)
      f_old = f

      val gtd = g.dot(d) // directional derivative
      if (ev.toType[Double](gtd) > -tolX) {
        isBreak = true
      } else {
        t = if (nIter == 1) {
          tmp1.copy(g).abs()
          math.min(1.0, 1.0 / ev.toType[Double](tmp1.sum())) * learningRate
        } else {
          learningRate
        }

        // optional line search: user function
        var lsFuncEval = 0
        if (this.lineSearch.isDefined) {
          val lineSearch = this.lineSearch.get
          val lineSearchOpts = this.lineSearchOptions.get
          val result = lineSearch(opfunc, x, ev.fromType[Double](t), d, f, g, gtd, lineSearchOpts)
          f = result._1
          g = result._2
          x.copy(result._3)
          t = ev.toType[Double](result._4)
          lsFuncEval = result._5
          f_hist.append(f)
        } else {
          // no line search, simply move with fixed-step
          x.add(ev.fromType[Double](t), d)

          /* re-evaluate function only if not in last iteration
          the reason we do this: in a stochastic setting,
          no use to re-evaluate that function here */
          if (_nIter != maxIter) {
            val result = opfunc(x)
            f = result._1
            g = result._2
            lsFuncEval = 1
            f_hist.append(f)
          }
        }

        // update func eval
        currentFuncEval = currentFuncEval + lsFuncEval
        funcEval += 1

        // check conditions
        if (_nIter == maxIter) {
          verbose("reached max number of iterations")
          isBreak = true
        } else if (currentFuncEval >= maxEval) {
          verbose("max nb of function evals")
          isBreak = true
        } else if (ev.toType[Double](tmp1.copy(g).abs().sum()) <= tolFun) {
          verbose("optimality condition below tolFun")
          isBreak = true
        } else if (ev.toType[Double](tmp1.copy(d).mul(ev.fromType[Double](t)).abs().sum()) < tolX) {
          verbose("step size below tolX")
          isBreak = true
        } else if (ev.toType[Double](ev.abs(ev.minus(f, f_old))) < tolX) {
          verbose("function value changing less than tolX")
          isBreak = true
        }
      }
    }

    // save state
    _state("old_dirs") = old_dirs
    _state("old_stps") = old_stps
    _state("Hdiag") = Hdiag
    _state("tmp1") = tmp1
    _state("g_old") = g_old
    _state("f_old") = f_old
    _state("t") = t
    _state("d") = d
    _state("dir_bufs") = dir_bufs
    _state("stp_bufs") = stp_bufs
    _state("ro") = ro
    _state("al") = al
    _state("funcEval") = funcEval
    _state("nIter") = nIter

    (x, f_hist.toArray)
  }

  def verbose(msg: String): Unit = {
    if (verbose) {
      println(s"<optim.lbfgs> $msg")
    }
  }

  override def loadFromTable(config: Table): this.type = {
    this.maxIter = config.get[Int]("maxIter").getOrElse(this.maxIter)
    this.maxEval = config.get[Double]("maxEval").getOrElse(this.maxEval)
    this.tolFun = config.get[Double]("tolFun").getOrElse(this.tolFun)
    this.tolX = config.get[Double]("tolX").getOrElse(this.tolX)
    this.nCorrection = config.get[Int]("nCorrection").getOrElse(this.nCorrection)
    this.learningRate = config.get[Double]("learningRate").getOrElse(this.learningRate)
    this.verbose = config.get[Boolean]("verbose").getOrElse(this.verbose)
    this.lineSearch = config.get[Option[LineSearch[T]]]("lineSearch").getOrElse(this.lineSearch)
    this.lineSearchOptions = config.get[Option[Table]]("lineSearchOptions")
      .getOrElse(this.lineSearchOptions)
    this
  }

  override def clearHistory(): Unit = {
    state.delete("dir_bufs")
    state.delete("d")
    state.delete("t")
    state.delete("Hdiag")
    state.delete("old_dirs")
    state.delete("old_stps")
    state.delete("g_old")
    state.delete("f_old")
    state.delete("ro")
    state.delete("al")
  }

  override def getLearningRate(): Double = this.learningRate
}
