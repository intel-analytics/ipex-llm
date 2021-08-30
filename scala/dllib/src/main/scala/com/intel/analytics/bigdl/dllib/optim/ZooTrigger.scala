/*
 * Copyright 2018 Analytics Zoo Authors.
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
package com.intel.analytics.zoo.common

import com.intel.analytics.bigdl.optim.Trigger
import com.intel.analytics.bigdl.utils.{T, Table}

/**
 * A trigger specifies a timespot or several timespots during training,
 * and a corresponding action will be taken when the timespot(s)
 * is reached.
 */
trait ZooTrigger extends Trigger {
  protected var zooState: Table = T()

  /**
   * We also hold some training metrics to control trigger.
   * @param zooState zoo state table
   */
  private[zoo] def setZooState(zooState: Table): Unit = {
    this.zooState = zooState
  }
}

/**
 * A trigger that triggers an action when each epoch finishs.
 * Could be used as trigger in setValidation and setCheckpoint
 * in Optimizer, and also in TrainSummary.setSummaryTrigger.
 */
case class EveryEpoch() extends ZooTrigger{
  private var lastEpoch = -1

  override def apply(state: Table): Boolean = {
    if (lastEpoch == -1) {
      lastEpoch = state[Int]("epoch")
      false
    } else {
      if (state[Int]("epoch") <= lastEpoch) {
        false
      } else {
        if (zooState.contains("numSlice") && zooState.contains("currentSlice")) {
          if (zooState[Int]("currentSlice") % zooState[Int]("numSlice") == 0) {
            lastEpoch = state[Int]("epoch")
            true
          } else {
            false
          }
        } else {
          lastEpoch = state[Int]("epoch")
          true
        }
      }
    }
  }
}
/**
 * A trigger that triggers an action every "n" iterations.
 * Could be used as trigger in setValidation and setCheckpoint
 * in Optimizer, and also in TrainSummary.setSummaryTrigger.
 *
 * @param interval - trigger interval "n"
 */
case class SeveralIteration(interval: Int) extends ZooTrigger{
  override def apply(state: Table): Boolean = {
    val curIteration = state[Int]("neval") - 1
    curIteration != 0 && curIteration % interval == 0
  }
}

/**
 * A trigger that triggers an action when training reaches
 * the number of epochs specified by "max".
 * Usually used in Optimizer.setEndWhen.
 *
 * @param max the epoch when the action takes place
 */
case class MaxEpoch(max: Int) extends ZooTrigger{
  override def apply(state: Table): Boolean = {
    state[Int]("epoch") > max
  }
}

/**
 * A trigger that triggers an action when training reaches
 * the number of iterations specified by "max".
 * Usually used in Optimizer.setEndWhen.
 *
 * @param max the iteration when the action takes place
 *
 */
case class MaxIteration(max: Int) extends ZooTrigger {
  override def apply(state: Table): Boolean = {
    state[Int]("neval") > max
  }
}

/**
 * A trigger that triggers an action when validation score larger than "max" score
 * @param max max score
 */
case class MaxScore(max: Float) extends ZooTrigger {
  override def apply(state: Table): Boolean = {
    state[Float]("score") > max
  }
}

/**
 * A trigger that triggers an action when training loss less than "min" loss
 * @param min min loss
 */
case class MinLoss(min: Float) extends ZooTrigger {
  override def apply(state: Table): Boolean = {
    state[Float]("Loss") < min
  }
}

/**
 * A trigger contains other triggers and triggers when all of them trigger (logical AND)
 * @param first first trigger
 * @param others others triggers
 */
case class And(first : ZooTrigger, others : ZooTrigger*) extends ZooTrigger {
  override def setZooState(zooState: Table): Unit = {
    super.setZooState(zooState)
    first.setZooState(zooState)
    others.foreach{zt =>
      zt.setZooState(zooState)
    }
  }

  override def apply(state: Table): Boolean = {
    first.apply(state) && others.forall(_.apply(state))
  }
}

/**
 * A trigger contains other triggers and triggers when any of them trigger (logical OR)
 * @param first first trigger
 * @param others others triggers
 */
case class Or(first : ZooTrigger, others : ZooTrigger*) extends ZooTrigger {
  override def setZooState(zooState: Table): Unit = {
    super.setZooState(zooState)
    first.setZooState(zooState)
    others.foreach{zt =>
      zt.setZooState(zooState)
    }
  }

  override def apply(state: Table): Boolean = {
    first.apply(state) || others.exists(_.apply(state))
  }
}
