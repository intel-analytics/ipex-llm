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

package com.intel.analytics.bigdl.utils

import java.util.concurrent.{ConcurrentHashMap, CyclicBarrier}

import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.spark.TaskContext

import scala.reflect._

/**
 * A parameter synchronizer among threads per task
 */
private[bigdl] object ParameterSynchronizer {

  private val events = new java.util.concurrent.ConcurrentHashMap[String, Event[_]]

  /**
   * Register event with key and total thread number
   * @param eventKey key to specify an event
   * @param threadNum total thread number synchronizing on this key
   * @tparam T
   */
  def register[T: ClassTag](eventKey: String, threadNum: Int): Unit = {
    var event = events.get(eventKey)
    if (event == null) {
      event = new Event[T](threadNum)
      events.putIfAbsent(eventKey, event)
    }
  }

  /**
   * Reset event with given key
   * @param eventKey Event key
   * @tparam T
   */
  def reset[T: ClassTag](eventKey: String): Unit = {
    events.get(eventKey).reset
  }

  /**
   * Sync data per thread
   * @param eventKey Event key
   * @param dt  data to be synchronized
   * @tparam T
   */
  def syncData[T: ClassTag](eventKey: String, dt: Tensor[T]): Unit = {
    events.get(eventKey).asInstanceOf[Event[T]].addData(dt)
  }

  /**
   * Collect all data synchronized
   * @param eventKey Event key
   * @tparam T
   * @return Data list from waiting threads
   */
  def collect[T: ClassTag](eventKey: String): java.util.Map[String, Tensor[T]] = {
    events.get(eventKey).data.asInstanceOf[java.util.Map[String, Tensor[T]]]
  }
}

private[bigdl] class Event[T: ClassTag](threadNum: Int) {
  val barrier = new CyclicBarrier(threadNum)
  val data = new ConcurrentHashMap[String, Tensor[T]]()

  /**
   * Add data to sync list for current thread
   * @param dt data to be added
   */
  def addData(dt: Tensor[T]): Unit = {
    barrier.await
    val currentId = Thread.currentThread().getId.toString
    data.put(currentId, dt)
    barrier.await
  }

  /**
   * Reset event, clear the data
   */
  def reset(): Unit = {
    barrier.await
    if (data.size != 0) {
      data.synchronized {
        if (data.size != 0) {
          data.clear
        }
      }
    }
    barrier.await
  }
}
