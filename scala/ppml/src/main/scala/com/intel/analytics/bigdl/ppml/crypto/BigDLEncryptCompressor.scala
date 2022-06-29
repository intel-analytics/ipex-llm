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

package com.intel.analytics.bigdl.ppml.crypto

import com.intel.analytics.bigdl.dllib.utils.Log4Error
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.compress.Compressor

class BigDLEncryptCompressor extends Compressor{
  val bigdlEncrypt = Crypto(AES_CBC_PKCS5PADDING)
  // TODO
  bigdlEncrypt.init(AES_CBC_PKCS5PADDING, ENCRYPT, "1234567890123456")
  var isFinished = false
  var b: Array[Byte] = null
  var off = 0
  var len = 0
  var hasHeader = false
  var tryFinished = false
  private var bytesRead = 0L
  private var bytesWritten = 0L

  override def setInput(b: Array[Byte], off: Int, len: Int): Unit = {
    this.b = b
    this.off = off
    this.len = len
  }

  override def needsInput(): Boolean = {
    len <= 0
  }

  override def setDictionary(b: Array[Byte], off: Int, len: Int): Unit = {
    Log4Error.invalidOperationError(false, "Unsupported setDictionary.")
  }

  override def getBytesRead: Long = ???

  override def getBytesWritten: Long = ???

  override def finish(): Unit = {
    tryFinished = true
  }

  override def finished(): Boolean = {
    isFinished
  }

  var lv2Buffer: Array[Byte] = null
  var lv2Off = 0
  var lv2Len = 0

  override def compress(b: Array[Byte], off: Int, len: Int): Int = {
    // lazy encrypt, in order to doFinal in the right time.
    if (tryFinished) {
      val o = bigdlEncrypt.doFinal(this.b, this.off, this.len)
      isFinished = true
      o._1 ++ o._2
      o._1.copyToArray(b, 0)
      o._2.copyToArray(b, o._1.length)
      o._1.length + o._2.length
    } else {
      val o = if (hasHeader) {
        val o = bigdlEncrypt.update(this.lv2Buffer, this.lv2Off, this.lv2Len)
        lv2Buffer = this.b
        lv2Off = this.off
        lv2Len = this.len
        this.len = 0
        o
      } else {
        hasHeader = true
        lv2Buffer = this.b
        lv2Off = this.off
        lv2Len = this.len
        this.len = 0
        bigdlEncrypt.genHeader()
      }
      o.copyToArray(b, 0)
      o.length
    }
  }

//  def output(): Array[Byte] = {
//    compressed
//  }

  override def reset(): Unit = {
    isFinished = false
    tryFinished = false
    hasHeader = false
    b = null
    len = 0
    off = 0
    bytesRead = 0L
    bytesWritten = 0L
  }

  override def end(): Unit = ???

  override def reinit(conf: Configuration): Unit = {
    reset()
    //TODO reinit bigdl encrypt
  }
}

object BigDLEncryptCompressor {

  def getCompressorType(conf: Configuration): Class[_ <: Compressor] = {
    classOf[BigDLEncryptCompressor]
  }

  def apply(conf: Configuration): BigDLEncryptCompressor = {
    // TODO read parameter
    new BigDLEncryptCompressor()
  }
}
