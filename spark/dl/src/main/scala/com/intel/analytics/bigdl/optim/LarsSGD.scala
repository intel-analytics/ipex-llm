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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Container
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.optim.SGD.{Default, LearningRateSchedule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import org.apache.log4j.{Level, Logger}

import scala.reflect.ClassTag


/**
 * An implementation of LARS https://arxiv.org/abs/1708.03888
 * Lars.createOptimForModule is recommended to be used to create LARS optim methods for multiple
 * layers
 *
 * @param lrScheduleOwner       if this optim method owns the learning rate scheduler.
 *                              A scheduler may be shared by multiple LARS scheduler
 * @param trust                 the trust on the learning rate scale, should be in 0 to 1
 * @param _learningRate         learning rate
 * @param _learningRateDecay    learning rate decay
 * @param _weightDecay          weight decay
 * @param _momentum             momentum
 * @param _learningRateSchedule the learning rate scheduler
 * @tparam T
 */
class LarsSGD[T: ClassTag](
                             lrScheduleOwner: Boolean,
                             trust: Double = 1.0,
                             _learningRate: Double = 1e-3,
                             _learningRateDecay: Double = 0.01,
                             _weightDecay: Double = 0.0005,
                             _momentum: Double = 0.5,
                             _learningRateSchedule: LearningRateSchedule
                             = Default()
                          )(implicit ev: TensorNumeric[T])
   extends SGD[T](_learningRate, _learningRateDecay, _weightDecay, _momentum,
     learningRateSchedule = _learningRateSchedule) {
  @transient
  private var buffer: Tensor[T] = null
  
  /**
   * @param feval     a function that takes a single input (X), the point of a evaluation, and
   *                  returns f(X) and df/dX
   * @param parameter the initial point
   * @return the new x vector and the function list {fx}, evaluated before the update
   */
  override def optimize(feval: Tensor[T] => (T, Tensor[T]),
                        parameter: Tensor[T]): (Tensor[T], Array[T]) = {

    val weightDecay = this.weightDecay
    val momentum = this.momentum
    val (fx, dfdx) = feval(parameter)
    if (buffer == null) buffer = Tensor[T]().resizeAs(dfdx)
    val _v =
      if (state.get[Tensor[T]]("v").isDefined) {
        state.get[Tensor[T]]("v").get
      } else {
        Tensor[T]().resizeAs(dfdx).zero()
      }
    learningRateSchedule.updateHyperParameter(this)
    val globalLr = -learningRateSchedule.currentRate * trust
    val normGradient = ev.sqrt(dfdx.sumSquare())
    val normParam = ev.sqrt(parameter.sumSquare())
    // scale = (normGradient + weightDecay * normParam) / normParam
    val scale = Tensor.scalar[T](normParam)
    scale.mul(ev.fromType[Double](weightDecay)).add(normGradient).div(normParam)
    val raw_scale_value = scale.value()
    val scale_value = if (ev.isInf(raw_scale_value)) {
      ev.fromType[Double](10000.0)
    } else if (ev.nearlyEqual(raw_scale_value, ev.fromType[Double](0.0), 0.0001)) {
      ev.fromType[Double](1e-4)
    } else if (ev.isNan(raw_scale_value)) {
      ev.fromType[Double](1.0)
    } else {
      raw_scale_value
    }
    // rate = globalLr / scale
    val rate = ev.divide(ev.fromType[Double](globalLr), scale_value)
    // _v = momentum * _v + rate * (dfdx + weightDecay * parameter)
    _v.mul(ev.fromType[Double](momentum))
    buffer.mul(parameter, ev.fromType[Double](weightDecay)).add(dfdx).mul(rate)
    _v.add(buffer)
    parameter.sub(_v)
    state("v") = _v
    (parameter, Array(fx))
  }

  /**
   * return an string of current hyperParameter.
   */
  override def getHyperParameter(): String = {
    if (lrScheduleOwner) {
      val clr = -this.learningRateSchedule.currentRate
      s"Current learning rate is $clr. "
    }
    else {
      ""
    }
  }

  /**
   * return an string of current hyperParameter.
   */
  override def getHyperParameter(config: Table): String = {
    if (lrScheduleOwner) {
      val clr = -config[Double]("clr")
      s"Current learning rate is $clr. "
    }
    else {
      ""
    }
  }

  override def updateHyperParameter(config: Table, state: Table): Unit = {
    val lrSchedule = config.get[LearningRateSchedule]("learningRateSchedule").getOrElse(Default())
    lrSchedule.updateHyperParameter(config, state)
  }

  override def getLearningRate(): Double = this.learningRateSchedule.currentRate
}

object LarsSGD {
  /**
   * Create a Map(String, OptimMethod) for a container. For each submodule in the container,
   * generate (module.getName(), new Lars[T]) pair in the returned map. The resulting map can be
   * used in setOptimMethods.
   * Note: each Lars optim uses the same LearningRateSchedule
   *
   * @param model                the container to build LARS optim method for
   * @param trust                the trust on the learning rate scale, should be in 0 to 1
   * @param learningRate         learning rate
   * @param learningRateDecay    learning rate decay
   * @param weightDecay          weight decay
   * @param momentum             momentum
   * @param learningRateSchedule the learning rate scheduler
   *
   */
  def createOptimForModule[T: ClassTag](model: Module[T],
                                        trust: Double = 1.0,
                                        learningRate: Double = 1e-3,
                                        learningRateDecay: Double = 0.01,
                                        weightDecay: Double = 0.005,
                                        momentum: Double = 0.5,
                                        learningRateSchedule: LearningRateSchedule = Default())
                                       (implicit ev: TensorNumeric[T]): Map[String,
     OptimMethod[T]] = {
    var isOwner = true
    // lrScheGenerator generates the same learningRateSchedule for each module
    // But it only returns isOwner = true for the first module
    val lrScheGenerator = (_: AbstractModule[Activity, Activity, T]) => {
      val _isOwner = isOwner
      isOwner = false
      (learningRateSchedule, _isOwner)
    }
    createOptimSeqForModule(model, lrScheGenerator,
      trust, learningRate, learningRateDecay, weightDecay, momentum).toMap

  }


  /**
   * Create a Map(String, OptimMethod) for a container. For each submodule in the container,
   * generate (module.getName(), new Lars[T]) pair in the returned map. The resulting map can be
   * used in setOptimMethods.
   * This function sets different LearningRateSchedules for different submodules
   *
   * @param model             the container to build LARS optim method for
   * @param lrScheGenerator   the learning rate schedule generator for each sub-module.
   *                          Generator accepts the sub-module that the schedule is linked to.
   *                          It should return a tuple (learningRateSchedule, isOwner), where
   *                          isOwner indicates whether the corresponding LARS optim method is
   *                          responsible for showing the learning rate in  getHyperParameter
   *                          (multiple LARS optim methods may share one learning rate scheduler)
   * @param trust             the trust on the learning rate scale, should be in 0 to 1
   * @param learningRate      learning rate
   * @param learningRateDecay learning rate decay
   * @param weightDecay       weight decay
   * @param momentum          momentum
   *
   */
  def createOptimLRSchedulerForModule[A <: Activity, B <: Activity, T: ClassTag]
  (model: Container[A, B, T],
   lrScheGenerator: AbstractModule[Activity, Activity, T] => (LearningRateSchedule, Boolean),
   trust: Double = 1.0,
   learningRate: Double = 1e-3,
   learningRateDecay: Double = 0.01,
   weightDecay: Double = 0.005,
   momentum: Double = 0.5)
  (implicit ev: TensorNumeric[T]): Map[String, OptimMethod[T]] = {
    createOptimSeqForModule(model, lrScheGenerator, trust, learningRate, learningRateDecay,
      weightDecay, momentum).toMap
  }

  /**
   * Create a Seq of (name,Lars) pair for the model
   *
   * @see createOptimLRSchedulerForModule
   */
  private def createOptimSeqForModule[T: ClassTag](model: Module[T],
                                                   lrScheGenerator: AbstractModule[Activity,
                                                      Activity, T] => (LearningRateSchedule,
                                                      Boolean),
                                                   trust: Double,
                                                   learningRate: Double,
                                                   learningRateDecay: Double,
                                                   weightDecay: Double,
                                                   momentum: Double)
                                                  (implicit ev: TensorNumeric[T]): Seq[(String,
     OptimMethod[T])] = {
    model match {
      case container: Container[_, _, T] =>
        container.modules.filter(mod => mod.parameters() != null).flatMap(mod => {
          // generate Seq for each sub-module
          createOptimSeqForModule(mod, lrScheGenerator, trust, learningRate, learningRateDecay,
            weightDecay, momentum)
        })
      case _ =>
        if (model.parameters() != null) {
          val (lrSche, isOwner) = lrScheGenerator(model)
          Seq((model.getName(), new LarsSGD[T](isOwner, trust, learningRate, learningRateDecay,
            weightDecay, momentum, lrSche)))
        }
        else {
          Seq()
        }
    }

  }
}
