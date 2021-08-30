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

import com.intel.analytics.bigdl.optim.SGD
import com.intel.analytics.bigdl.optim.SGD.LearningRateSchedule
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.zoo.pipeline.api.keras.models.{InternalOptimizerUtil}

object Optim {

  /**
   * A fixed learning rate scheduler, always return the same learning rate
   * @param lr learning rate
   */
  case class Fixed(lr: Double) extends LearningRateSchedule {
    override def updateHyperParameter(config: Table, state: Table): Unit = {
      val nevals = state.get[Int]("evalCounter").getOrElse(0)
      state("evalCounter") = nevals + 1
      config("clr") = lr
    }

    override def updateHyperParameter[T](optimMethod: SGD[T]): Unit = {
      val state = InternalOptimizerUtil.getStateFromOptiMethod[T](optimMethod)
      val nevals = state.get[Int]("evalCounter").getOrElse(0)
      state("evalCounter") = nevals + 1
      currentRate = lr
    }
  }
}
