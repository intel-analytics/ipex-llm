/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.visualization.tensorboard

import java.io.File
import java.net.InetAddress
import java.util.concurrent.ConcurrentLinkedDeque

import org.tensorflow.util.Event

class EventWriter(logDir: String, maxQueue: Int = 10, flushSeconds: Int = 30) extends Runnable {
  val eventQueue = new ConcurrentLinkedDeque[Event]()
  // TODO: a better name
  val outputFile = new File(logDir + s"/bigdl.out.tfevents.${System.currentTimeMillis() / 1e3}" +
    s".${InetAddress.getLocalHost().getHostName()}")
  val recordWriter = new RecordWriter(outputFile)
  // Add an empty Event to the queue.
  eventQueue.add(Event.newBuilder().setWallTime(System.currentTimeMillis() / 1e3).build())
  var running: Boolean = true

  def addEvent(event: Event): this.type = {
    eventQueue.add(event)
    this
  }

  def flush(): this.type = {
    while (eventQueue.size() > 0) {
      recordWriter.write(eventQueue.pop())
    }
    this
  }

  def close(): this.type = {
    running = false
    flush()
    recordWriter.close()
    this
  }

  override def run(): Unit = {
    while (running) {
      println("flush!!!!")
      flush()
      Thread.sleep(flushSeconds * 1000)
    }
  }
}
