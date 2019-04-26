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

class Lars[@specialized(Float, Double) T: ClassTag](
                                                     lrScheduleOwner: Boolean,
                                                     trust: Double = 1.0,
                                                     learningRate: Double = 1e-3,
                                                     learningRateDecay: Double = 0.01,
                                                     weightDecay: Double = 0.0005,
                                                     momentum: Double = 0.5,
                                                     learningRateSchedule: LearningRateSchedule = Default()
                                                   )(implicit ev: TensorNumeric[T])
  extends OptimMethod[T]
{
  @transient
  private var buffer: Tensor[T] = null
  private val sgd = new SGD[T](learningRate,learningRateDecay,weightDecay,momentum,learningRateSchedule=this.learningRateSchedule)

  /**
    * An implementation of LARS https://arxiv.org/abs/1708.03888
    *
    * @param feval     a function that takes a single input (X), the point of a evaluation, and
    *                  returns f(X) and df/dX
    * @param parameter the initial point
    * @return the new x vector and the function list {fx}, evaluated before the update
    */
  override def optimize(feval: Tensor[T] => (T, Tensor[T]),
                        parameter: Tensor[T]): (Tensor[T], Array[T]) = {

    val weightDecay = this.weightDecay
    val momentum = this.momentum
    val state = sgd.state
    val (fx, dfdx) = feval(parameter)
    if (buffer == null) buffer = Tensor[T]().resizeAs(dfdx)

    val _v =
      if (state.get[Tensor[T]]("v").isDefined) {
        state.get[Tensor[T]]("v").get
      } else {
        Tensor[T]().resizeAs(dfdx).zero()
      }
    if(lrScheduleOwner)
      learningRateSchedule.updateHyperParameter(sgd)
    val globalLr = -learningRateSchedule.currentRate * trust

    val normGradient = ev.sqrt(dfdx.sumSquare())
    val normParam = ev.sqrt(parameter.sumSquare())

    //scale = (normGradient + weightDecay * normParam) / normParam
    val scale = Tensor.scalar[T](normParam)

    scale.mul(ev.fromType[Double](weightDecay)).add(normGradient).div(normParam)

    val raw_scale_value=scale.value()
    val scale_value = if (ev.isFinite(raw_scale_value) && ev.isGreater(raw_scale_value,ev.fromType[Double](0.001))){
      raw_scale_value
    }else{
      ev.fromType[Double](1.0)
    }


    //rate = globalLr / scale
    val rate =  ev.divide(ev.fromType[Double](globalLr), scale_value)

    //_v = momentum * _v + rate * (dfdx + weightDecay * parameter)
    _v.mul(ev.fromType[Double](momentum))
    buffer.mul(parameter,ev.fromType[Double](weightDecay)).add(dfdx).mul(rate)
    _v.add(buffer)


    parameter.sub(_v)
    state("v") = _v

    (parameter, Array(fx))
  }

  override def loadFromTable(config: Table): this.type = {
    sgd.loadFromTable(config)
    this
  }

  override def clearHistory(): Unit = {
    sgd.clearHistory()
  }

  /**
    * return an string of current hyperParameter.
    */
  override def getHyperParameter(): String = {
    if(lrScheduleOwner){
      val clr = -this.learningRateSchedule.currentRate
      s"Current learning rate is $clr. "
    }
    else
      ""
  }

  /**
    * return an string of current hyperParameter.
    */
  override def getHyperParameter(config: Table): String = {
    if(lrScheduleOwner){
      val clr = -config[Double]("clr")
      s"Current learning rate is $clr. "
    }
    else
      ""
  }

  override def updateHyperParameter(): Unit = {
    this.learningRateSchedule.updateHyperParameter(sgd)
  }



  override def updateHyperParameter(config: Table, state: Table): Unit = {
    val lrSchedule = config.get[LearningRateSchedule]("learningRateSchedule").getOrElse(Default())
    lrSchedule.updateHyperParameter(config, state)
  }

  override def getLearningRate(): Double = this.learningRateSchedule.currentRate
}

object Lars {
  /**
    * Create a Map(String, OptimMethod) for a container. For each submodule in the container,
    * generate (module.getName(), new Lars[T]) pair in the returned map. The resulting map can be
    * used in setOptimMethods.
    * Note: each Lars optim uses the same LearningRateSchedule
    * @param model the container to build LARS optimizer for
    * @param trust the trust rate of the learning rate scale, should be in (0,1)

    * */
  def createOptimForModule[T: ClassTag](model: Module[T],
                                        trust: Double = 1,
                                        learningRate: Double = 1e-3,
                                        learningRateDecay: Double = 0.01,
                                        weightDecay: Double = 0.005,
                                        momentum: Double = 0.5,
                                        learningRateSchedule: LearningRateSchedule = Default())
                                       (implicit ev: TensorNumeric[T]): Map[String, OptimMethod[T]] = {

    createOptimSeqForModule(model, (_:AbstractModule[Activity,Activity,T])=>learningRateSchedule, true,
      trust,learningRate,learningRateDecay,weightDecay,momentum).toMap

  }


  /**
    * Create a Map(String, OptimMethod) for a container. For each submodule in the container,
    * generate (module.getName(), new Lars[T]) pair in the returned map. The resulting map can be
    * used in setOptimMethods.
    * This function sets different LearningRateSchedules for different submodules
    * @param model the container to build LARS optimizer for
    * @param lrScheGenerator the learning rate schedule generater for each sub-module. The input variable is the sub-module that the schedule is linked to
    * @param trust the trust rate of the learning rate scale, should be in (0,1)

    * */

  def createOptimLRSchedulerForModule[A<:Activity,B<:Activity,T : ClassTag](model : Container[A,B,T],
                                                                            lrScheGenerator: AbstractModule[Activity,Activity,T] => LearningRateSchedule,
                                                                            trust: Double = 1,
                                                                            learningRate: Double = 1e-3,
                                                                            learningRateDecay: Double = 0.01,
                                                                            weightDecay: Double = 0.005,
                                                                            momentum: Double = 0.5)
                                                                           (implicit ev: TensorNumeric[T]): Map[String,OptimMethod[T]] ={
    createOptimSeqForModule(model,lrScheGenerator,true, trust,learningRate,learningRateDecay,weightDecay,momentum).toMap
  }

  /**
    * Create a Seq of (name,Lars) pair for the model
    * @param lrScheduleOwner whether the first sub-module should return the hyper parameters in getHyperParameter(). Use this
    *                      parameter to avoid verbose output of hyper parameters
    */
  private def createOptimSeqForModule[T: ClassTag](model: Module[T],
                                                   lrScheGenerator: AbstractModule[Activity,Activity,T] => LearningRateSchedule,
                                                   lrScheduleOwner: Boolean,
                                                   trust: Double ,
                                                   learningRate: Double ,
                                                   learningRateDecay: Double ,
                                                   weightDecay: Double ,
                                                   momentum: Double )
                                                  (implicit ev: TensorNumeric[T]): Seq[(String, OptimMethod[T])] = {
    var _lrScheduleOwner = lrScheduleOwner
    model match {
      case container: Container[_,_,T]=>
        container.modules.filter(mod => mod.parameters() != null).flatMap(mod => {
          //generate Seq for each sub-module
          val ret = createOptimSeqForModule(mod,lrScheGenerator,_lrScheduleOwner,trust, learningRate,learningRateDecay,weightDecay,momentum)
          _lrScheduleOwner = false
          ret
        })
      case _=>
        if(model.parameters()!=null){
          Seq((model.getName(),new Lars[T](lrScheduleOwner, trust, learningRate, learningRateDecay, weightDecay, momentum, lrScheGenerator(model))))
        }
        else
          Seq()
    }

  }
}