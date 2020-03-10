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

package com.intel.analytics.bigdl.utils.wrapper.mkldnn

import java.util.{List, Map}
import com.intel.analytics.bigdl.mkl.hardware.Affinity

object AffinityWrapper {

  def getAffinity(): Array[Int] = {
    Affinity.getAffinity
  }

  def setAffinity(coreId: Int): Unit = {
    Affinity.setAffinity(coreId)
  }

  def setAffinity(coreIds: Array[Int]): Unit = {
    Affinity.setAffinity(coreIds)
  }

  def stats(): Map[Integer, List[java.lang.Long]] = {
    Affinity.stats()
  }

  def getOmpAffinity(): Array[Int] = {
    Affinity.getOmpAffinity()
  }

  def setOmpAffinity(): Unit = {
    Affinity.setOmpAffinity()
  }



}
