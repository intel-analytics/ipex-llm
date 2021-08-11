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

import com.intel.analytics.bigdl.utils.Table

/**
 * A trigger specifies a timespot or several timespots during training,
 * and a corresponding action will be taken when the timespot(s)
 * is reached.
 */
trait Trigger extends Serializable {
  def apply(state: Table): Boolean
}

object Trigger {

  /**
   * A trigger that triggers an action when each epoch finishs.
   * Could be used as trigger in setValidation and setCheckpoint
   * in Optimizer, and also in TrainSummary.setSummaryTrigger.
   */
  def everyEpoch: Trigger = {
    new Trigger() {
      private var lastEpoch = -1

      override def apply(state: Table): Boolean = {
        if (lastEpoch == -1) {
          lastEpoch = state[Int]("epoch")
          false
        } else {
          if (state[Int]("epoch") == lastEpoch) {
            false
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
  def severalIteration(interval: Int): Trigger = {
    new Trigger() {
      override def apply(state: Table): Boolean = {
        val curIteration = state[Int]("neval")
        curIteration != 0 && curIteration % interval == 0
      }
    }
  }

  /**
   * A trigger that triggers an action when training reaches
   * the number of epochs specified by "max".
   * Usually used in Optimizer.setEndWhen.
   *
   * @param max the epoch when the action takes place
   */
  def maxEpoch(max: Int): Trigger = {
    new Trigger() {
      override def apply(state: Table): Boolean = {
        state[Int]("epoch") > max
      }
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
  def maxIteration(max: Int): Trigger = {
    new Trigger() {
      override def apply(state: Table): Boolean = {
        state[Int]("neval") > max
      }
    }
  }

  /**
   * A trigger that triggers an action when validation score larger than "max" score
   * @param max max score
   */
  def maxScore(max: Float): Trigger = {
    new Trigger() {
      override def apply(state: Table): Boolean = {
        state[Float]("score") > max
      }
    }
  }

  /**
   * A trigger that triggers an action when training loss less than "min" loss
   * @param min min loss
   */
  def minLoss(min: Float): Trigger = {
    new Trigger() {
      override def apply(state: Table): Boolean = {
        state[Float]("Loss") < min
      }
    }
  }

  /**
   * A trigger contains other triggers and triggers when all of them trigger (logical AND)
   * @param first first trigger
   * @param others others triggers
   */
  def and(first : Trigger, others : Trigger*): Trigger = {
    new Trigger() {
      override def apply(state: Table): Boolean = {
        first.apply(state) && others.forall(_.apply(state))
      }
    }
  }

  /**
   * A trigger contains other triggers and triggers when any of them trigger (logical OR)
   * @param first first trigger
   * @param others others triggers
   */
  def or(first : Trigger, others : Trigger*): Trigger = {
    new Trigger() {
      override def apply(state: Table): Boolean = {
        first.apply(state) || others.exists(_.apply(state))
      }
    }
  }


}

