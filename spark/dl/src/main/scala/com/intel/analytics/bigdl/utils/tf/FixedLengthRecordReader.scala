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

import java.io.{File, FileInputStream}

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
class FixedLengthRecordReader(fileName: File,
                              footerBytes: Int,
                              headerBytes: Int,
                              hopBytes: Int,
                              recordBytes: Int) extends Iterator[Array[Byte]] {

  private val inputStream = new FileInputStream(fileName)

  private var dataBuffer: Array[Byte] = null

  inputStream.skip(headerBytes)


  override def hasNext: Boolean = {
    if (dataBuffer != null) {
      true
    } else {
      dataBuffer = new Array[Byte](recordBytes)
      val numOfBytes = inputStream.read(dataBuffer)
      if (numOfBytes == recordBytes) {
        inputStream.skip(hopBytes)
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
