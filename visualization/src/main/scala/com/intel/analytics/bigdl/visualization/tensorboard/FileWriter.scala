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

import java.util.concurrent.Executors

import org.tensorflow.framework.Summary
import org.tensorflow.util.Event

class FileWriter(val logDirecotry : String, flushMilliSeconds: Int = 10000) {
  val logDir = new java.io.File(logDirecotry)
  require(!logDir.exists() || logDir.isDirectory, s"FileWriter: can not create $logDir")
  if (!logDir.exists()) logDir.mkdirs()
  val eventWriter = new EventWriter(logDirecotry, flushMilliSeconds)
  val pool = Executors.newFixedThreadPool(1)
  pool.submit(eventWriter)
  // adds a Summary protocol buffer to the event file.
  // summary - a Summary protobuf String generted using TBLogger.scalar, etc.
  // globalStep - a global count of the event, could be epoch, or iteration,
  // as long as consistent.
  // refer to https://github.com/dmlc/tensorboard/tree/master/python/tensorboard for simplified
  // implementation of writers
  def addSummary(summary: Summary, globalStep: Long): this.type = {
    val event = Event.newBuilder().setSummary(summary).build()
    addEvent(event, globalStep)
    this
  }

  def addEvent(event: Event, globalStep: Long): this.type = {
    eventWriter.addEvent(
      event.toBuilder.setWallTime(System.currentTimeMillis() / 1e3).setStep(globalStep).build())
    this
  }

  def close(): this.type = {
    eventWriter.close()
    this
  }
}
