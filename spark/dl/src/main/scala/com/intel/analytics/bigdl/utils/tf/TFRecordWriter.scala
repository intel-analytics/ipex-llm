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
package com.intel.analytics.bigdl.utils.tf

import java.io.OutputStream
import java.nio.{ByteBuffer, ByteOrder}

import com.intel.analytics.bigdl.utils.Crc32

class TFRecordWriter(out: OutputStream) {

  private def toByteArrayAsLong(data: Long): Array[Byte] = {
    val buff = new Array[Byte](8)
    val bb = ByteBuffer.wrap(buff)
    bb.order(ByteOrder.LITTLE_ENDIAN)
    bb.putLong(data)
    buff
  }

  private def toByteArrayAsInt(data: Int): Array[Byte] = {
    val buff = new Array[Byte](4)
    val bb = ByteBuffer.wrap(buff)
    bb.order(ByteOrder.LITTLE_ENDIAN)
    bb.putInt(data)
    buff
  }

  def write(record: Array[Byte], offset: Int, length: Int): Unit = {
    val len = toByteArrayAsLong(length)
    out.write(len)
    out.write(toByteArrayAsInt(Crc32.maskedCRC32(len).toInt))
    out.write(record, offset, length)
    out.write(toByteArrayAsInt(Crc32.maskedCRC32(record, offset, length).toInt))
  }

  def write(record: Array[Byte]): Unit = {
    write(record, 0, record.length)
  }
}
