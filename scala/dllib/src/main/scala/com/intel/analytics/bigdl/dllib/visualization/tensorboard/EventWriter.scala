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

package com.intel.analytics.bigdl.visualization.tensorboard

import java.net.InetAddress
import java.util.concurrent.{LinkedBlockingDeque, TimeUnit}

import org.apache.hadoop.fs.{FileSystem, Path}
import org.tensorflow.util.Event

/**
 * Event writer, write event protocol buffers to file.
 *
 * @param logDir Support local directory and HDFS directory
 * @param flushMillis
 */
private[bigdl] class EventWriter(logDir: String,
                                 flushMillis: Int = 1000,
                                 fs: FileSystem) extends Runnable {
  private val eventQueue = new LinkedBlockingDeque[Event]()
  private val outputFile = new Path(logDir +
    s"/bigdl.tfevents.${(System.currentTimeMillis() / 1e3).toInt}" +
    s".${InetAddress.getLocalHost().getHostName()}")
  private val recordWriter = new RecordWriter(outputFile, fs)
  // Add an empty Event to the queue.
  eventQueue.add(Event.newBuilder().setWallTime(System.currentTimeMillis() / 1e3).build())
  @volatile private var running: Boolean = true

  def addEvent(event: Event): this.type = {
    eventQueue.add(event)
    this
  }

  private def flush(): this.type = {
    while (!eventQueue.isEmpty) {
      recordWriter.write(eventQueue.pop())
    }
    this
  }

  private def writeEvent(): this.type = {
    val e = eventQueue.poll(flushMillis, TimeUnit.MILLISECONDS)
    if (null != e) recordWriter.write(e)
    this
  }

  def close(): this.type = {
    running = false
    this
  }

  override def run(): Unit = {
    while (running) {
      writeEvent()
    }
    flush()
    recordWriter.close()
  }
}
