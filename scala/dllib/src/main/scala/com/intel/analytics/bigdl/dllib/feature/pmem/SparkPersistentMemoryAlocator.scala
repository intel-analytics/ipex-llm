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

// package com.intel.analytics.bigdl.dllib.feature.pmem
//
// import com.intel.analytics.bigdl.dllib.feature.pmem._
// import org.apache.spark.SparkEnv
//
// object MemoryAllocator {
//
//  def getInstance(memoryType: MemoryType = DIRECT): BasicMemoryAllocator = {
//    memoryType match {
//      case PMEM =>
//        println("Using persistent memory")
//        SparkPersistentMemoryAlocator.nativeAllocator
//      case DIRECT =>
//        println("Using main memory")
//        DRAMBasicMemoryAllocator.instance
//      case _ =>
//        throw new IllegalArgumentException(s"Not supported memoryType: ${memoryType}")
//    }
//  }
// }
//
// object SparkPersistentMemoryAlocator {
//  private val sparkConf = SparkEnv.get.conf
//  private val memPaths = sparkConf.get(
//    "analytics.zoo.pmem.paths", "/mnt/pmem0:/mnt/pmem1").split(":")
//
//  private val memSizePerByte = sparkConf.getInt(
//    "analytics.zoo.pmem.bytesize.socket", 248) * 1024 * 1024 * 1024
//  val pathIndex = executorID % memPaths.length
//  println(s"Executor: ${executorID()} is using ${memPaths(pathIndex)}")
//
//  lazy val nativeAllocator = {
//    val instance = PersistentMemoryAllocator.getInstance()
//    instance.initialize(memPaths(pathIndex), memSizePerByte)
//    instance
//  }
//
//  private def executorID(): Int = {
//    if (SparkEnv.get.executorId.equals("driver")) {
//      1
//    } else {
//      SparkEnv.get.executorId.toInt
//    }
//  }
//
//  def allocate(size: Long): Long = {
//    nativeAllocator.allocate(size)
//  }
//
//  def free(address: Long): Unit = {
//    nativeAllocator.free(address)
//  }
//
//  def copy(destAddress: Long, srcAddress: Long, size: Long): Unit = {
//    nativeAllocator.copy(destAddress, srcAddress, size)
//  }
// }
