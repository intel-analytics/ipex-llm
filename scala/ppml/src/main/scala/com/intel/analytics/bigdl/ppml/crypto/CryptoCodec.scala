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
import com.intel.analytics.bigdl.ppml.crypto.CryptoCodec.CryptoDecompressStream
import org.apache.commons.logging.{Log, LogFactory}
import org.apache.hadoop.classification.InterfaceAudience
import org.apache.hadoop.classification.InterfaceStability
import org.apache.hadoop.conf.{Configurable, Configuration}
import org.apache.hadoop.io.compress.{DefaultCodec, _}
import org.apache.hadoop.io.compress.zlib.BuiltInGzipDecompressor
import org.apache.hadoop.io.compress.zlib.ZlibCompressor
import org.apache.hadoop.io.compress.zlib.ZlibCompressor.{CompressionHeader, CompressionLevel, CompressionStrategy}
import org.apache.hadoop.io.compress.zlib.ZlibDecompressor
import org.apache.hadoop.io.compress.zlib.ZlibDecompressor.{CompressionHeader, ZlibDirectDecompressor}
import org.apache.hadoop.io.compress.zlib.ZlibFactory

import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.util.zip.GZIPOutputStream
import org.apache.hadoop.util.PlatformName.IBM_JAVA

class CryptoCodec
  extends Configurable with CompressionCodec with DirectDecompressionCodec {
  protected var conf: Configuration = null

  override def setConf(conf: Configuration): Unit = {
    this.conf = conf
  }

  override def getConf: Configuration = conf

  @throws[IOException]
  override def createOutputStream(out: OutputStream): CompressionOutputStream = {
    // TODO: CompressionCodec.Util.createOutputStreamWithCodecPool(this, conf, out)
    val compressor = this.createCompressor()
    this.createOutputStream(out, compressor)
  }

  @throws[IOException]
  override def createOutputStream(out: OutputStream, compressor: Compressor) = {
    new CompressorStream(out, compressor, conf.getInt("io.file.buffer.size", 4 * 1024))
  }

  override def getCompressorType: Class[_ <: Compressor] = {
    BigDLEncryptCompressor.getCompressorType(conf)
//    ZlibFactory.getZlibCompressorType(conf)
  }

  override def createCompressor: Compressor = {
    BigDLEncryptCompressor(conf)
//    ZlibFactory.getZlibCompressor(conf)
  }

  @throws[IOException]
  override def createInputStream(in: InputStream): CompressionInputStream = {
    // TODO:  CompressionCodec.Util.createInputStreamWithCodecPool(this, conf, in)
//    val decompressor = this.createDecompressor()
//    this.createInputStream(in, decompressor)
    new CryptoDecompressStream(in, conf.getInt("io.file.buffer.size", 4 * 1024))
  }

  @throws[IOException]
  override def createInputStream(in: InputStream, decompressor: Decompressor) = {
//    new DecompressorStream(in, decompressor, conf.getInt("io.file.buffer.size", 4 * 1024))
    new CryptoDecompressStream(in, conf.getInt("io.file.buffer.size", 4 * 1024))
  }

  override def getDecompressorType: Class[_ <: Decompressor] = {
    ZlibFactory.getZlibDecompressorType(conf)
    null
  }

  override def createDecompressor: Decompressor = {
    ZlibFactory.getZlibDecompressor(conf)
    null
  }

  /**
   * {@inheritDoc }
   */
  override def createDirectDecompressor: DirectDecompressor = {
    ZlibFactory.getZlibDirectDecompressor(conf)
    null
  }

  override def getDefaultExtension: String = {
    ".cbc"
  }
}

object CryptoCodec {
  class CryptoDecompressStream(
      in: InputStream, bufferSize: Int) extends DecompressorStream(in) {
    buffer = new Array[Byte](bufferSize)
    val bigdlEncrypt = Crypto(AES_CBC_PKCS5PADDING)
    // TODO
    bigdlEncrypt.init(AES_CBC_PKCS5PADDING, DECRYPT, "1234567890123456")
    var headerVerified = false

    override def decompress(b: Array[Byte], off: Int, len: Int): Int = {
      if(!headerVerified) {
        bigdlEncrypt.verifyHeader(in)
        headerVerified = true
      }

      val decompressed = if (in.available() < -100 || in.available() > buffer.size) {
        val m = getCompressedData()
        bigdlEncrypt.update(buffer, 0, m)
      } else {
        val last = getCompressedData()
        if (last == -1) { // apparently the previous end-of-stream was also end-of-file:
          // return success, as if we had never called getCompressedData()
          eof = true
          return -1
        }
        val hmacSize = 32
        val inputHmac = buffer.slice(last - hmacSize, last)
        val (lastSlice, streamHmac) = bigdlEncrypt.doFinal(buffer, 0, last - hmacSize)
        Log4Error.invalidInputError(!inputHmac.sameElements(streamHmac), "hmac not match")
        lastSlice
      }

      decompressed.copyToArray(b, 0)
      decompressed.length
    }
  }

}
