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

sealed trait MemoryType

case object PMEM extends MemoryType

case object DRAM extends MemoryType

case object DIRECT extends MemoryType

sealed trait DataStrategy

case object PARTITIONED extends DataStrategy

case object REPLICATED extends DataStrategy

object MemoryType {
  def fromString(str: String): MemoryType = {
    str.toUpperCase() match {
      case "PMEM" => PMEM
      case "DRAM" => DRAM
      case "DIRECT" => DIRECT
    }
  }
}


/**
 *
 * @param totalBytes
 */
abstract class NativeArray[T](totalBytes: Long, memoryType: MemoryType) {

  assert(totalBytes > 0, s"The size of bytes should be larger than 0, but got: ${totalBytes}!")

  val memoryAllocator = MemoryAllocator.getInstance(memoryType)

  val startAddr: Long = memoryAllocator.allocate(totalBytes)

  assert(startAddr > 0, s"Not enough memory to allocate: ${totalBytes} bytes!")

  val lastOffSet = startAddr + totalBytes

  var deleted: Boolean = false

  def get(i: Int): T

  def set(i: Int, value: T): Unit

  def free(): Unit = {
    if (!deleted) {
      memoryAllocator.free(startAddr)
      deleted = true
    }
  }

  protected def indexOf(i: Int): Long
}


