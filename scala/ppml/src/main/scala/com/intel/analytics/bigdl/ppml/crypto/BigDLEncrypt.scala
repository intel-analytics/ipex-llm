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

import com.intel.analytics.bigdl.dllib.utils.{File, Log4Error}
import com.intel.analytics.bigdl.ppml.crypto.CryptoMode
import org.apache.hadoop.fs.Path

import java.io._
import java.security.SecureRandom
import java.time.Instant
import java.util.Arrays
import javax.crypto.spec.{IvParameterSpec, SecretKeySpec}
import javax.crypto.{Cipher, Mac}
import org.apache.spark.input.PortableDataStream

import java.nio.ByteBuffer
import scala.util.Random

class BigDLEncrypt extends Crypto {
  protected var cipher: Cipher = null
  protected var mac: Mac = null
  protected var ivParameterSpec: IvParameterSpec = null
  protected var opMode: OperationMode = null
  protected var initializationVector: Array[Byte] = null
  override def init(cryptoMode: CryptoMode, mode: OperationMode, dataKeyPlaintext: String): Unit = {
    opMode = mode
    val secret = dataKeyPlaintext.getBytes()
    // key encrypt
    val signingKey = Arrays.copyOfRange(secret, 0, 16)
    val encryptKey = Arrays.copyOfRange(secret, 16, 32)
//    initializationVector = Arrays.copyOfRange(secret, 0, 16)
    val r = new Random(signingKey.sum)
    initializationVector = Array.tabulate(16)(_ => (r.nextInt(256) - 128).toByte)
    ivParameterSpec = new IvParameterSpec(initializationVector)
    val encryptionKeySpec = new SecretKeySpec(encryptKey, cryptoMode.secretKeyAlgorithm)
    cipher = Cipher.getInstance(cryptoMode.encryptionAlgorithm)
    cipher.init(mode.opmode, encryptionKeySpec, ivParameterSpec)
    mac = Mac.getInstance(cryptoMode.signingAlgorithm)
    val signingKeySpec = new SecretKeySpec(signingKey, cryptoMode.signingAlgorithm)
    mac.init(signingKeySpec)
  }

  protected var signingDataStream: DataOutputStream = null

  override def genFileHeader(): Array[Byte] = {
    Log4Error.invalidOperationError(cipher != null,
      s"you should init BigDLEncrypt first.")
    val timestamp: Instant = Instant.now()
    val signingByteBuffer = ByteBuffer.allocate(1 + 8 + ivParameterSpec.getIV.length)
    val version: Byte = (0x80).toByte
    signingByteBuffer.put(version)
    signingByteBuffer.putLong(timestamp.getEpochSecond())
    signingByteBuffer.put(ivParameterSpec.getIV())
    signingByteBuffer.array()
  }

  override def verifyFileHeader(header: Array[Byte]): Unit = {
    val headerBuffer = ByteBuffer.wrap(header)
    val version: Byte = headerBuffer.get()
    if (version.compare((0x80).toByte) != 0) {
      throw new EncryptRuntimeException("File header version error!")
    }
    val timestampSeconds: Long = headerBuffer.getLong
    val initializationVector: Array[Byte] = header.slice(1 + 8, header.length)
    if (!initializationVector.sameElements(this.initializationVector)) {
      throw new EncryptRuntimeException("File header not match!" +
        "expected: " + this.initializationVector.mkString(",") +
        ", but got: " + initializationVector.mkString(", "))
    }
  }

  override def update(content: Array[Byte]): Array[Byte] = {
    val cipherText: Array[Byte] = cipher.update(content)
    mac.update(cipherText)
    cipherText
  }

  override def update(content: Array[Byte], offset: Int, len: Int): Array[Byte] = {
    val cipherText: Array[Byte] = cipher.update(content, offset, len)
    mac.update(cipherText, offset, len)
    cipherText
  }

  override def doFinal(content: Array[Byte]): (Array[Byte], Array[Byte]) = {
    val cipherText: Array[Byte] = cipher.doFinal(content)
    val hmac: Array[Byte] = mac.doFinal(cipherText)
    (cipherText, hmac)
  }

  override def doFinal(content: Array[Byte], offset: Int, len: Int): (Array[Byte], Array[Byte]) = {
    val cipherText: Array[Byte] = cipher.doFinal(content, offset, len)
    val hmac: Array[Byte] = mac.doFinal(cipherText.slice(offset, offset + len))
    (cipherText, hmac)
  }

  val blockSize = 1024 * 1024 // 1m per update
  val byteBuffer = new Array[Byte](blockSize)
  override def encryptStream(inputStream: DataInputStream, outputStream: DataOutputStream): Unit = {
    val header = genFileHeader()
    outputStream.write(header)
    while (inputStream.available() > blockSize) {
      val readLen = inputStream.read(byteBuffer)
      outputStream.write(update(byteBuffer, 0, readLen))
    }
    val last = inputStream.read(byteBuffer)
    val (lastSlice, hmac) = doFinal(byteBuffer, 0, last)
    outputStream.write(lastSlice)
    outputStream.write(hmac)
    outputStream.flush()
  }

  val hmacSize = 32
  override def decryptStream(inputStream: DataInputStream, outputStream: DataOutputStream): Unit = {
    val header = read(inputStream, 25)
    verifyFileHeader(header)
    while (inputStream.available() > blockSize) {
      val readLen = inputStream.read(byteBuffer)
      outputStream.write(update(byteBuffer, 0, readLen))
    }
    val last = inputStream.read(byteBuffer)
    val inputHmac = byteBuffer.slice(last - hmacSize, last)
    val (lastSlice, streamHmac) = doFinal(byteBuffer, 0, last - hmacSize)
    if(inputHmac.sameElements(streamHmac)) {
      throw new EncryptRuntimeException("hmac not match")
    }
    outputStream.write(lastSlice)
    outputStream.flush()
  }

  override def decryptFile(binaryFilePath: String, savePath: String): Unit = {
    Log4Error.invalidInputError(savePath != null && savePath != "",
      "decrypted file save path should be specified")
    val fs = File.getFileSystem(binaryFilePath)
    val bis = fs.open(new Path(binaryFilePath))
    val outs = fs.create(new Path(savePath))
    encryptStream(bis, outs)
    bis.close()
    outs.close()
  }

  override def encryptFile(binaryFilePath: String, savePath: String): Unit = {
    Log4Error.invalidInputError(savePath != null && savePath != "",
      "decrypted file save path should be specified")
    val fs = File.getFileSystem(binaryFilePath)
    val bis = fs.open(new Path(binaryFilePath))
    val outs = fs.create(new Path(savePath))
    decryptStream(bis, outs)
    bis.close()
    outs.close()
  }

  private def read(stream: DataInputStream, numBytes: Int): Array[Byte] = {
    val retval = new Array[Byte](numBytes)
    val bytesRead: Int = stream.read(retval)
    if (bytesRead < numBytes) {
      throw new EncryptRuntimeException("Not enough bits to read!")
    }
    retval
  }

  override def decryptBigContent(
        ite: Iterator[(String, PortableDataStream)]): Iterator[String] = {
    var result: Iterator[String] = Iterator[String]()

    while (ite.hasNext == true) {
      val inputStream: DataInputStream = ite.next._2.open()
      verifyFileHeader(read(inputStream, 25))

      // do first
      var lastString = ""
      while (inputStream.available() > blockSize) {
        val readLen = inputStream.read(byteBuffer)
        Log4Error.unKnowExceptionError(readLen != blockSize)
        val currentSplitDecryptString = new String(byteBuffer, 0, readLen)
        val splitDecryptString = lastString + currentSplitDecryptString
        val splitDecryptStringArray = splitDecryptString.split("\r").flatMap(_.split("\n"))
        lastString = splitDecryptStringArray.last
        result = result ++ splitDecryptStringArray.dropRight(1)
      }
      // do last
      val last = inputStream.read(byteBuffer)
      val inputHmac = byteBuffer.slice(last - hmacSize, last)
      val (lastSlice, streamHmac) = doFinal(byteBuffer, 0, last - hmacSize)
      if (inputHmac.sameElements(streamHmac)) {
        throw new EncryptRuntimeException("hmac not match")
      }
      val lastDecryptString = lastString + new String(lastSlice)
      val splitDecryptStringArray = lastDecryptString.split("\r").flatMap(_.split("\n"))
      result = result ++ splitDecryptStringArray
    }
    result

  }

}
