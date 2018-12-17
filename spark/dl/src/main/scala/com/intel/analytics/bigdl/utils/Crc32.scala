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

import netty.Crc32c


private[bigdl] object Crc32 {

  def maskedCRC32(crc32c: Crc32c, data: Array[Byte], offset: Int, length: Int): Long = {
    crc32c.reset()
    crc32c.update(data, offset, length)
    val x = u32(crc32c.getValue)
    u32(((x >> 15) | u32(x << 17)) + 0xa282ead8)
  }

  def maskedCRC32(crc32c: Crc32c, data: Array[Byte]): Long = {
    maskedCRC32(crc32c, data, 0, data.length)
  }

  def maskedCRC32(data: Array[Byte]): Long = {
    val crc32c = new Crc32c()
    maskedCRC32(crc32c, data)
  }

  def maskedCRC32(data: Array[Byte], offset: Int, length: Int): Long = {
    val crc32c = new Crc32c()
    maskedCRC32(crc32c, data, offset, length)
  }


  def u32(x: Long): Long = {
    x & 0xffffffff
  }

}
