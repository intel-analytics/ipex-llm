/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
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

package com.intel.analytics.bigdl.nn.dnn

import scala.io.Source
import java.io.{FileNotFoundException, IOException}

import com.intel.analytics.bigdl.mkl.MKL

object Affinity {
  var totalNumSockets = 0
  var totalNumPhyCores = 0
  var totalNumLogCores = 0

  val savedAffinity = getAffinity()
  val halfAffinity = setHalfCore()

  private[this] def parseCpuInfo(): Unit = {
    try {
      val cpuinfoFile = Source.fromFile("/proc/cpuinfo")
      val allLines = try cpuinfoFile.getLines().toList finally cpuinfoFile.close()

      totalNumLogCores = allLines.filter(x => x.startsWith("processor")).distinct.size
      totalNumSockets = allLines.filter(x => x.startsWith("physical id")).distinct.size
      totalNumPhyCores = allLines.filter(x => x.startsWith("core id"))
        .distinct.size * totalNumSockets
    } catch {
      case ex: FileNotFoundException => println("Couldn't find /proc/cpuinfo.")
      case ex: IOException => println("Had an IOException trying to read /proc/cpuinfo")
    }
  }

  private[this] def getAffinity(): Array[Byte] = {
    parseCpuInfo()
    println(totalNumPhyCores)
    MKL.setNumThreads(totalNumPhyCores)
    MKL.getAffinity
  }

  private[this] def setAffinity(affinity: Array[Byte]): Unit = {
    MKL.setNumThreads(totalNumPhyCores)
    MKL.setAffinity()
  }

  private[this] def setHalfCore(): Array[Byte] = {
    require(savedAffinity != null, "Get affinity error")
    require(savedAffinity.length > 0, "Bind thread to CPUs error")

    def setBit(index: Int, target: Byte): Byte = {
      (target | (1 << index)).toByte
    }

    val nBytes = savedAffinity.length
    val affinity = new Array[Byte](nBytes)
    val halfCores = totalNumLogCores / 2

    for (i <- 0 until halfCores) {
      val indexInByte = i % 8
      // cpu_set_t has 1024 bits, and the last nBytes are cores we want to set
      val indexOfBytes = nBytes - i/8 - 1

      affinity(indexOfBytes) = setBit(indexInByte, affinity(indexOfBytes))
    }

    affinity
  }

  def acquireCore(): Unit = {
    setAffinity(halfAffinity)
  }

  def release(): Unit = {
    MKL.setNumThreads(totalNumPhyCores)
    MKL.release()
  }
}
