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
import com.intel.analytics.bigdl.mkl.MklDnn

object CoreWrapper {

  def isLoaded(): Boolean = {
    MklDnn.isLoaded
  }

  def setNumThreads(n: Int): Unit = {
    MklDnn.setNumThreads(n)
  }

  def setFlushDenormalState(): Unit = {
    MklDnn.setFlushDenormalState()
  }
}
