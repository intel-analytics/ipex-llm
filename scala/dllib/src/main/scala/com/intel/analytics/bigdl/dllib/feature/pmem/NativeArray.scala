/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.feature.pmem

import scala.collection.mutable.ArrayBuffer

sealed trait MemoryType extends Serializable

case object PMEM extends MemoryType

case object DRAM extends MemoryType

case object DIRECT extends MemoryType

case class DISK_AND_DRAM(numSlice: Int) extends MemoryType

sealed trait DataStrategy

case object PARTITIONED extends DataStrategy

case object REPLICATED extends DataStrategy

object MemoryType {
  def fromString(str: String): MemoryType = {
    str.toUpperCase() match {
      case "PMEM" => PMEM
      case "DRAM" => DRAM
      case "DIRECT" => DIRECT
      case default =>
        try {
          DISK_AND_DRAM(str.toInt)
        } catch {
          case nfe: NumberFormatException =>
            throw new IllegalArgumentException(s"Unknown memory type $default," +
              s"excepted PMEM, DRAM, DIRECT or a int number.")
        }

    }
  }
}

object NativeArray {
  private val natives = new ArrayBuffer[NativeArray[_]]()

  def free(): Unit = {
    NativeArray.natives.map{_.free()}
  }
}

/**
 *
 * @param totalBytes
 */
abstract class NativeArray[T](totalBytes: Long, memoryType: MemoryType) {

  assert(totalBytes > 0, s"The size of bytes should be larger than 0, but got: ${totalBytes}!")

  val memoryAllocator = MemoryAllocator.getInstance(memoryType)

  val startAddr: Long = NativeArray.synchronized {
    val addr = memoryAllocator.allocate(totalBytes)
    NativeArray.natives.append(this)
    addr
  }

  assert(startAddr > 0, s"Not enough memory to allocate: ${totalBytes} bytes!")

  val lastOffSet = startAddr + totalBytes

  var deleted: Boolean = false

  def get(i: Int): T

  def set(i: Int, value: T): Unit

  def free(): Unit = NativeArray.synchronized {
    if (!deleted) {
      memoryAllocator.free(startAddr)
      deleted = true
    }
  }

  protected def indexOf(i: Int): Long
}


