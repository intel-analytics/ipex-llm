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

trait Trigger {
  def apply(state: Table): Boolean
}

object Trigger {
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

  def severalIteration(interval: Int): Trigger = {
    new Trigger() {
      override def apply(state: Table): Boolean = {
        val curIteration = state[Int]("neval")
        curIteration != 0 && curIteration % interval == 0
      }
    }
  }

  def maxEpoch(max: Int): Trigger = {
    new Trigger() {
      override def apply(state: Table): Boolean = {
        state[Int]("epoch") > max
      }
    }
  }

  def maxIteration(max: Int): Trigger = {
    new Trigger() {
      override def apply(state: Table): Boolean = {
        state[Int]("neval") > max
      }
    }
  }
}

