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

class BigDLEncryptCompressor(cryptoMode: CryptoMode,
  dataKeyPlainText: String,
  dataKeyCipherText: String = "",
  encrypterType: String = BigDLEncrypt.COMMON) extends Compressor {
  val bigdlEncrypt = BigDLEncrypt(encrypterType)
  var hasHeader = false
  encrypterType match {
    case BigDLEncrypt.COMMON =>
      bigdlEncrypt.asInstanceOf[BigDLEncrypt]
        .init(cryptoMode, ENCRYPT, dataKeyPlainText, dataKeyCipherText)
    case BigDLEncrypt.NATIVE_AES_CBC =>
      bigdlEncrypt.init(cryptoMode, ENCRYPT, dataKeyPlainText)
  }
  var isFinished = false
  var b: Array[Byte] = null
  var off = 0
  var len = 0
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

  override def getBytesRead: Long = {
    bytesRead
  }

  override def getBytesWritten: Long = {
    bytesWritten
  }

  override def finish(): Unit = {
    if (lv2Len == 0 && len == 0) {
      isFinished = true
    } else {
      tryFinished = true
    }
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
      val o = bigdlEncrypt.doFinal(this.lv2Buffer, this.lv2Off, this.lv2Len)
      bytesRead += this.lv2Len
      isFinished = true
      if (o._2 != null) { // has Mac
        o._1 ++ o._2
        o._1.copyToArray(b, 0)
        o._2.copyToArray(b, o._1.length)
        o._1.length + o._2.length
      } else {
        o._1.copyToArray(b, 0)
        o._1.length
      }
    } else {
      val o = if (hasHeader) {
        val o = bigdlEncrypt.update(this.lv2Buffer, this.lv2Off, this.lv2Len)
        bytesRead += this.lv2Len
        // create a buffer to cache undecrypted data, size of this.b is changing.
        if (lv2Buffer.size >= this.b.size) {
          this.b.copyToArray(this.lv2Buffer)
        } else {
          lv2Buffer = this.b.clone()
        }
        lv2Off = this.off
        lv2Len = this.len
        this.len = 0
        o
      } else {
        hasHeader = true
        lv2Buffer = this.b.clone()
        lv2Off = this.off
        lv2Len = this.len
        this.len = 0
        bigdlEncrypt.genHeader
      }
      o.copyToArray(b, 0)
      bytesWritten += o.length
      o.length
    }
  }

  override def reset(): Unit = {
    isFinished = false
    b = null
    off = 0
    len = 0
    hasHeader = false
    tryFinished = false
    bytesRead = 0L
    bytesWritten = 0L
  }

  override def end(): Unit = {
    Log4Error.invalidOperationError(false, "Unsupported operation end.")
  }

  override def reinit(conf: Configuration): Unit = {
    reset()
    // TODO reinit bigdl encrypt
  }
}

object BigDLEncryptCompressor {

  def getCompressorType(conf: Configuration): Class[_ <: Compressor] = {
    classOf[BigDLEncryptCompressor]
  }

  def apply(conf: Configuration): BigDLEncryptCompressor = {
    val (dataKeyPlainText, dataKeyCipherText, encrypterType) = (
      conf.get("bigdl.write.dataKey.plainText"),
      conf.get("bigdl.write.dataKey.cipherText"),
      conf.get("spark.bigdl.encryter.type", BigDLEncrypt.COMMON)
    )
    new BigDLEncryptCompressor(AES_CBC_PKCS5PADDING,
      dataKeyPlainText, dataKeyCipherText, encrypterType)
  }
}


