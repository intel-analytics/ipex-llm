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

package com.intel.analytics.bigdl.serving.http

import com.codahale.metrics.Timer
import org.slf4j.LoggerFactory

trait Supportive {
  val logger = LoggerFactory.getLogger(getClass)

  def timing[T](name: String)(timers: Timer*)(f: => T): T = {
    val begin = System.nanoTime()
    val contexts = timers.map(_.time())
    val result = f
    contexts.map(_.stop())
    val end = System.nanoTime()
    val cost = (end - begin)
    logger.info(s"$name time elapsed [${cost / 1e6} ms]")
    result
  }

  def average[T](name: String)(batch: Int)(timers: Timer*)(f: => T): T = {
    val begin = System.currentTimeMillis
    val contexts = timers.map(_.time())
    val result = f
    contexts.map(_.stop())
    val end = System.currentTimeMillis
    val cost = (end - begin)
    logger.info(s"$name time elapsed [${cost / 1000} s, ${cost % 1000} ms], " +
      s"average: [${cost / batch} ms], throughput: [${batch * 1000.0 / cost}]")
    result
  }

  def silent[T](name: String)(timers: Timer*)(f: => T): T = {
    val begin = System.currentTimeMillis
    val contexts = timers.map(_.time())
    val result = f
    contexts.map(_.stop())
    val end = System.currentTimeMillis
    val cost = (end - begin)
    result
  }

}
