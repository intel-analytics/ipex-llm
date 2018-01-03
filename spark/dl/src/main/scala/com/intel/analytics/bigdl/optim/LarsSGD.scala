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

import com.intel.analytics.bigdl.optim.SGD.{Default, LearningRateSchedule}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.collection.immutable.HashMap.HashMap1
import scala.collection.mutable
import scala.reflect.ClassTag

/**
 * A implementation of SGD which support LARS
 * @param learningRate learning rate
 * @param weightDecay weight decay
 * @param momentum momentum
 * @tparam T
 */
class LarsSGD[@specialized(Float, Double) T: ClassTag](
  learningRate: Double = 1e-3,
  learningRateDecay: Double = 0.0,
  weightDecay: Double = 0.0,
  momentum: Double = 0.0,
  larsLearningRateSchedule: LearningRateSchedule = Default(),
  val gwRation: Double = 1.0)(implicit ev: TensorNumeric[T])
  extends SGD[T](learningRate, learningRateDecay, weightDecay, momentum,
    learningRateSchedule = larsLearningRateSchedule) {

  import SGD._

  /**
   *
   * @param feval a function that takes a single input (X), the point of a evaluation,
   * and returns f(X) and df/dX
   * @param x the initial point
   * @return the new x 1D tensor and the function list, evaluated before the update
   */
  override def optimize(feval: (Tensor[T]) => (T, Tensor[T]), x: Tensor[T])
    : (Tensor[T], Array[T]) = {

    this.updateHyperParameter()
    val wd = this.weightDecay
    val mom = this.momentum
    val clr = ev.fromType(this.learningRateSchedule.currentRate)

    val (fx, dfdx) = feval(x)

    require(state.get[Array[(Int, Int)]]("lookupList").isDefined,
      "LarsSGD needs to know start and length for each layer")
    val lookupList = state.get[mutable.HashMap[Int, (Int, Int)]]("lookupList").get
    val gwNorm2List = state.get[Array[(Int, (T, T))]]("gwNorm2List").get

    lookupList.foreach(e => {
      val layerId = e._1
      val start = e._2._1
      val length = e._2._2
      val wPerLayer = x.narrow(1, start, length)
      var dfdxPerLayer = dfdx.narrow(1, start, length)
      val (gNorm2, wNorm2) = gwNorm2List(layerId)._2
      val localLr = ev.times(ev.fromType(gwRation), ev.times(clr, ev.divide(wNorm2,
      ev.plus(gNorm2, ev.times(ev.fromType(wd), wNorm2)))))

      if (wd != 0) {
      require(!state.get[Boolean]("isLayerwiseScaled").getOrElse(false),
      "SGD: Can't set layerwise scale and weight decay at the same time")
      }
      if (wd != 0) {
      dfdxPerLayer.add(ev.fromType[Double](wd), wPerLayer)
      }

      if (mom != 0) {
        val stateDFDX = state.get[Tensor[T]](s"${layerId}ThDfdx") match {
        case None =>
          val DFDX = Tensor[T]().resizeAs(dfdxPerLayer).copy(dfdxPerLayer)
          DFDX.mul(localLr)
          state(s"${layerId}ThDfdx") = DFDX
          DFDX
        case s: Some[Tensor[T]] => s.get.mul(ev.fromType[Double](mom)).
          add(ev.negative(localLr), dfdxPerLayer)
        }
        dfdxPerLayer = stateDFDX
      }

      wPerLayer.add(ev.negative(ev.one), dfdxPerLayer)
    })

    (x, Array(fx))
  }
}