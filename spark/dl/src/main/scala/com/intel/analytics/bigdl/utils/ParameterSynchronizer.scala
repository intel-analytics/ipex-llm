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

  val fEvents = new java.util.concurrent.ConcurrentHashMap[String, Event[Float]]

  val dEvents = new java.util.concurrent.ConcurrentHashMap[String, Event[Double]]

  /**
   * Register event with key and total thread number
   * @param eventKey key to specify an event
   * @param threadNum total thread number synchronizing on this key
   * @tparam T
   */
  def register[T: ClassTag](eventKey: String, threadNum: Int): Unit = {
    if (classTag[T] ==  classTag[Float]) {
      var event = fEvents.get(eventKey)
      if (event == null) {
        event = new Event[Float](threadNum)
        fEvents.putIfAbsent(eventKey, event)
      }
    } else if (classTag[T] ==  classTag[Double]) {
      var event = dEvents.get(eventKey)
      if (event == null) {
        event = new Event[Double](threadNum)
        dEvents.putIfAbsent(eventKey, event)
      }
    }
  }

  /**
   * Reset event with given key
   * @param eventKey Event key
   * @tparam T
   */
  def reset[T: ClassTag](eventKey: String): Unit = {
    if (classTag[T] ==  classTag[Float]) {
      fEvents.get(eventKey).reset
    } else if (classTag[T] ==  classTag[Double]) {
      dEvents.get(eventKey).reset
    }
  }

  /**
   * Sync data per thread
   * @param eventKey Event key
   * @param dt  data to be synchronized
   * @tparam T
   */
  def syncData[T: ClassTag](eventKey: String, dt: Tensor[T]): Unit = {
    val partitionKey = TaskContext.getPartitionId.toString
    if (classTag[T] ==  classTag[Float]) {
      fEvents.get(eventKey).asInstanceOf[Event[T]].addData(dt)
    } else if (classTag[T] ==  classTag[Double]) {
      dEvents.get(eventKey).asInstanceOf[Event[T]].addData(dt)
    }
  }

  /**
   * Collect all data synchronized
   * @param eventKey Event key
   * @tparam T
   * @return Data list from waiting threads
   */
  def collect[T: ClassTag](eventKey: String): java.util.Map[String, Tensor[T]] = {
    val partitionKey = TaskContext.getPartitionId.toString
    if (classTag[T] ==  classTag[Float]) {
      fEvents.get(eventKey).data.asInstanceOf[java.util.Map[String, Tensor[T]]]
    } else if (classTag[T] ==  classTag[Double]) {
      dEvents.get(eventKey).data.asInstanceOf[java.util.Map[String, Tensor[T]]]
    } else {
      null
    }
  }
}

private class Event[T: ClassTag](threadNum: Int) {
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
