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

import java.io.{BufferedInputStream, File, FileInputStream, InputStream}
import java.nio.{ByteBuffer, ByteOrder}

/**
 * Internal use only.
 *
 * TF record format:
 *  uint64 length
 *  uint32 masked_crc32_of_length
 *  byte   data[length]
 *  uint32 masked_crc32_of_data
 *
 */
class TFRecordIterator(inputStream: InputStream) extends Iterator[Array[Byte]] {

  private var dataBuffer: Array[Byte] = null

  private val lengthBuffer: Array[Byte] = new Array[Byte](8)



  override def hasNext: Boolean = {
    if (dataBuffer != null) {
      true
    } else {
      val numOfBytes = inputStream.read(lengthBuffer)
      if (numOfBytes == 8) {
        val lengthWrapper = ByteBuffer.wrap(lengthBuffer)
        lengthWrapper.order(ByteOrder.LITTLE_ENDIAN)
        val length = lengthWrapper.getLong().toInt
        // todo, do crc check, simply skip now
        inputStream.skip(4)

        dataBuffer = new Array[Byte](length)
        inputStream.read(dataBuffer)
        // todo, do crc check, simply skip now
        inputStream.skip(4)
        true
      } else {
        inputStream.close()
        false
      }
    }
  }

  override def next(): Array[Byte] = {
    if (hasNext) {
      val data = this.dataBuffer
      this.dataBuffer = null
      data
    } else {
      throw new NoSuchElementException("next on empty iterator")
    }
  }
}

object TFRecordIterator {
  def apply(file: File): TFRecordIterator = {
    val inputStream = new FileInputStream(file)
    new TFRecordIterator(inputStream)
  }
}
