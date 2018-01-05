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

package com.intel.analytics.bigdl.nn.mkldnn

object Utils {
  def time[R](block: => R): (Double, R) = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    val takes = (t1 - t0) / 1e9
//    println(s"time elpased: ${takes}")
    (takes, result)
  }

  def manyTimes[R](block: => R)(iters: Int): (Double, R) = {
    time[R] {
      var i = 0
      while (i < iters - 1) {
        block
        i += 1
      }
      block
    }
  }

  def speedup(base: Double, after: Double): String = {
    val result = (base - after) / base
    ((result * 1000).toInt / 10.0).toString + "%"
  }
}
