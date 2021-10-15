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

package com.intel.analytics.bigdl.serving.engine

import java.util.PriorityQueue

import org.apache.log4j.Logger

/**
 * Timer class
 */
class Timer() {
  class TimerUnit(_countFlag: Boolean = true) {
    var total: Float = 0 // total cost up to now
    var record: Int = 0 // total record number up to now
    var batchNum: Int = 0 // total batch number up to now
    var average: Float = 0 // average cost up to now
    var averageBatch: Float = 0 // average cost per batch up to now
    var first: Double = 0 // first record to trigger warm up
    var max: Double = Double.MinValue // max cost up to now
    var min: Double = Double.MaxValue // min cost up to now
    val topQ = new PriorityQueue[Double]()
    val nQ = 10
    val countFlag = _countFlag
    topQ.add(Long.MinValue)
    def update(cost: Double, num: Int): Unit = {
      total += cost.toFloat
      record += num
      batchNum += 1
      average = total / record
      averageBatch = total / batchNum
      if (max == Long.MinValue) {
        first = cost
      }
      if (cost > max) {
        max = cost
      }
      if (cost < min) {
        min = cost
      }
      if (topQ.size() >= nQ && cost > topQ.peek()) {
        topQ.poll()
        topQ.add(cost)
      }
      if (topQ.size() < nQ) {
        topQ.add(cost)
      }
    }
  }


  var timerMap = Map[String, TimerUnit]()

  def timing[T](name: String, num: Int)(f: => T): T = {
    val begin = System.nanoTime()
    val result = f
    val end = System.nanoTime()
    val cost = (end - begin) / 1e6
    Logger.getLogger(getClass).info(s"$name time elapsed [${(cost / 1e3).toInt} s, $cost ms].")
    if (!timerMap.contains(name)) {
      timerMap += (name -> new TimerUnit())
    }
    timerMap(name).update(cost, num)
    result
  }
  def print(): Unit = {
    timerMap.foreach(t => {
      val name = t._1
      val timer = t._2
      println(s"$name: Total cost of ${timer.total}, record num ${timer.record}, " +
        s"average per input ${timer.average}, " +
        s"average per batch ${timer.averageBatch}, first ${timer.first}, " +
        s"max ${timer.max}, min ${timer.min} (ms/batch)")
      println(s"Top ${timer.nQ} of statistic:")
      var tmpArr = Array[Double]()
      (0 until timer.nQ + 1).foreach(i => {
        if (!timer.topQ.isEmpty) {
          tmpArr = tmpArr :+ timer.topQ.peek()
          println(s"Top ${timer.nQ - i}: ${timer.topQ.poll()} ms")
        }
      })
      if (timer.topQ.isEmpty) {
        tmpArr.foreach(ele => timer.topQ.add(ele))
      }
    })
  }

}
