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

import org.apache.hadoop.classification.InterfaceAudience
import org.apache.hadoop.classification.InterfaceStability
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.compress._
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


/**
 * This class creates gzip compressors/decompressors.
 */
@InterfaceAudience.Public
@InterfaceStability.Evolving object CryptoCodec {
  /**
   * A bridge that wraps around a DeflaterOutputStream to make it
   * a CompressionOutputStream.
   */
  @InterfaceStability.Evolving protected object GzipOutputStream {
    private object ResetableGZIPOutputStream {
      private val TRAILER_SIZE = 8
      val JVMVersion: String = System.getProperty("java.version")
      private val HAS_BROKEN_FINISH = IBM_JAVA && JVMVersion.contains("1.6.0")
    }

    private class ResetableGZIPOutputStream @throws[IOException]
    (out: OutputStream) extends GZIPOutputStream(out) {
      @throws[IOException]
      def resetState(): Unit = {
        `def`.reset()
      }
    }
  }

  @InterfaceStability.Evolving
  protected class GzipOutputStream(out: OutputStream) extends CompressorStream(out) {

    @throws[IOException]
    override def close(): Unit = {
      out.close()
    }

    @throws[IOException]
    override def flush(): Unit = {
      out.flush()
    }

    @throws[IOException]
    override def write(b: Int): Unit = {
      out.write(b)
    }

    @throws[IOException]
    override def write(data: Array[Byte], offset: Int, length: Int): Unit = {
      out.write(data, offset, length)
    }

    @throws[IOException]
    override def finish(): Unit = {
      out.asInstanceOf[GzipOutputStream.ResetableGZIPOutputStream].finish()
    }

    @throws[IOException]
    override def resetState(): Unit = {
      out.asInstanceOf[GzipOutputStream.ResetableGZIPOutputStream].resetState()
    }
  }

  final private[crypto] class GzipZlibCompressor(
        level: ZlibCompressor.CompressionLevel, strategy: ZlibCompressor.CompressionStrategy,
        header: ZlibCompressor.CompressionHeader, directBufferSize: Int) extends
        ZlibCompressor(CompressionLevel.DEFAULT_COMPRESSION,
    CompressionStrategy.DEFAULT_STRATEGY,
      ZlibCompressor.CompressionHeader.GZIP_FORMAT, 64 * 1024) {
    def this(conf: Configuration) {
      this (ZlibFactory.getCompressionLevel(conf),
        ZlibFactory.getCompressionStrategy(conf),
        ZlibCompressor.CompressionHeader.GZIP_FORMAT, 64 * 1024)
    }
  }

  final private[crypto] class BigDLDecompressor() extends ZlibDecompressor(
    ZlibDecompressor.CompressionHeader.AUTODETECT_GZIP_ZLIB, 64 * 1024) {
  }
}

@InterfaceAudience.Public
@InterfaceStability.Evolving class CryptoCodec extends DefaultCodec {
  @throws[IOException]
  override def createOutputStream(out: OutputStream): CompressionOutputStream = {
    if (!ZlibFactory.isNativeZlibLoaded(getConf)) return new CryptoCodec.GzipOutputStream(out)
    val compressor = CodecPool.getCompressor(this, getConf)
    this.createOutputStream(out, compressor)
  }

  @throws[IOException]
  override def createOutputStream(
        out: OutputStream,
        compressor: Compressor): CompressionOutputStream = {
    if (compressor != null) {
      new CompressorStream(out, compressor, getConf.getInt("io.file.buffer.size", 4 * 1024))
    } else {
      createOutputStream(out)
    }
  }

  override def createCompressor: Compressor = if (ZlibFactory.isNativeZlibLoaded(getConf)) {
    new CryptoCodec.GzipZlibCompressor(getConf)
  }
  else null

  override def getCompressorType: Class[_ <: Compressor] = {
    if (ZlibFactory.isNativeZlibLoaded(getConf)) {
      classOf[CryptoCodec.GzipZlibCompressor]
    } else {
      null
    }
  }

  @throws[IOException]
  override def createInputStream(in: InputStream): CompressionInputStream = {
    val decompressor = CodecPool.getDecompressor(this)
    this.createInputStream(in, decompressor)
  }

  @throws[IOException]
  override def createInputStream(
        in: InputStream,
        decompressor: Decompressor): CompressionInputStream = {
    val _decompressor = if (decompressor == null) {
      createDecompressor
    } else {
      decompressor
    }
    // always succeeds (or throws)
    new DecompressorStream(in, _decompressor, getConf.getInt("io.file.buffer.size", 4 * 1024))
  }

  override def createDecompressor: Decompressor = {
    if (ZlibFactory.isNativeZlibLoaded(getConf)) {
      new CryptoCodec.BigDLDecompressor
    } else {
      new BuiltInGzipDecompressor
    }
  }

  override def getDecompressorType: Class[_ <: Decompressor] = {
    if (ZlibFactory.isNativeZlibLoaded(getConf)) {
      classOf[CryptoCodec.BigDLDecompressor]
    } else {
      classOf[BuiltInGzipDecompressor]
    }
  }

  override def createDirectDecompressor: DirectDecompressor = {
    if (ZlibFactory.isNativeZlibLoaded(getConf)) {
      new ZlibDecompressor.ZlibDirectDecompressor(
        ZlibDecompressor.CompressionHeader.AUTODETECT_GZIP_ZLIB, 0)
    } else {
      null
    }
  }

  override def getDefaultExtension: String = {
    ".gz"
  }
}
