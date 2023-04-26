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
import java.nio.charset.StandardCharsets
import org.apache.spark.input.PortableDataStream

import java.nio.ByteBuffer
import scala.util.Random

/**
 * BigDL general crypto for encrypt and decrypt data.
 */
class BigDLEncrypt extends Crypto {
  protected var cipher: Cipher = null
  protected var mac: Mac = null
  protected var ivParameterSpec: IvParameterSpec = null
  protected var encryptionKeySpec: SecretKeySpec = null
  protected var opMode: OperationMode = null
  protected var initializationVector: Array[Byte] = null
  protected var encryptedDataKey: String = ""
  // If inputStream.available() > Int.maxValue, the return value is
  // -2147483162 in FSDataInputStream.
  protected val outOfSize = -2e9.toInt
  // Init an encrypter
  def init(cryptoMode: CryptoMode, mode: OperationMode,
           dataKeyPlaintext: String, dataKeyCipherText: String): Unit = {
    encryptedDataKey = dataKeyCipherText
    init(cryptoMode, mode, dataKeyPlaintext)
  }

  // Init a decrypter
  def init(cryptoMode: CryptoMode, mode: OperationMode,
           dataKeyPlaintext: String, initializationVector: Array[Byte]): Unit = {
    init(cryptoMode, mode, dataKeyPlaintext)
    verifyHeader(initializationVector)
  }


  /**
   * Init this crypto with crypto mode, operation mode and keys.
   * @param cryptoMode cryptoMode to en/decrypt data, such as AES_CBC_PKCS5PADDING.
   * @param mode en/decrypt mode, one of ENCRYPT or DECRYPT.
   * @param dataKeyPlaintext signing key and data key.
   */
  override def init(cryptoMode: CryptoMode, mode: OperationMode, dataKeyPlaintext: String): Unit = {
    opMode = mode
    val secret = dataKeyPlaintext.getBytes()
    // key encrypt
    val signingKey = Arrays.copyOfRange(secret, 0, 16)
    val encryptKey = Arrays.copyOfRange(secret, 16, 32)
    val r = new SecureRandom()
    initializationVector = Array.tabulate(16)(_ => (r.nextInt(256) - 128).toByte)
    ivParameterSpec = new IvParameterSpec(initializationVector)
    encryptionKeySpec = new SecretKeySpec(encryptKey, cryptoMode.secretKeyAlgorithm)
    cipher = Cipher.getInstance(cryptoMode.encryptionAlgorithm)
    cipher.init(mode.opmode, encryptionKeySpec, ivParameterSpec)
    mac = Mac.getInstance(cryptoMode.signingAlgorithm)
    val signingKeySpec = new SecretKeySpec(signingKey, cryptoMode.signingAlgorithm)
    mac.init(signingKeySpec)
  }

  protected var signingDataStream: DataOutputStream = null

  /**
   * If encrypt data, should generate header and put return value to the head.
   * @return header bytes
   */
  override def genHeader(): Array[Byte] = {
    Log4Error.invalidOperationError(cipher != null,
      s"you should init BigDLEncrypt first.")
    val dataKeyCipherTextBytes = encryptedDataKey.getBytes(StandardCharsets.UTF_8)
    val signingByteBuffer = ByteBuffer.allocate(400)
    signingByteBuffer.putInt(dataKeyCipherTextBytes.length)
    signingByteBuffer.putInt(ivParameterSpec.getIV.length)
    signingByteBuffer.put(dataKeyCipherTextBytes)
    signingByteBuffer.put(ivParameterSpec.getIV())
    val suffixLength = (
      400 - 4 - dataKeyCipherTextBytes.length - 4 - ivParameterSpec.getIV.length).max(0)
    val suffix: Array[Byte] = Array.fill(suffixLength)((0x80).toByte)
    signingByteBuffer.put(suffix)
    signingByteBuffer.array()
  }

  /**
   * Verify the header bytes when decrypt.
   * @param header header bytes
   * @return encryptedDataKey String
   */
  override def verifyHeader(initializationVector: Array[Byte]): Unit = {
    if (!initializationVector.sameElements(this.initializationVector)) {
      ivParameterSpec = new IvParameterSpec(initializationVector)
      cipher.init(opMode.opmode, encryptionKeySpec, ivParameterSpec)
    }
  }

  /**
   * Verify the header bytes in the stream.
   * @param header header bytes
   */
  override def verifyHeader(in: InputStream): Unit = {
    val header = read(in, 400)
    verifyHeader(header)
  }

  /**
   * Continues a multiple-part encryption or decryption operation
   * (depending on how this crypto was initialized).
   * @param content byte to be encrypted or decrypted.
   * @return encrypted or decrypted bytes.
   */
  override def update(content: Array[Byte]): Array[Byte] = {
    val cipherText: Array[Byte] = cipher.update(content)
    mac.update(cipherText)
    cipherText
  }

  /**
   * Continues a multiple-part encryption or decryption operation
   * (depending on how this crypto was initialized).
   * @param content bytes to be encrypted or decrypted.
   * @param offset bytes offset of content.
   * @param len bytes len of content.
   * @return encrypted or decrypted bytes.
   */
  override def update(content: Array[Byte], offset: Int, len: Int): Array[Byte] = {
    val cipherText: Array[Byte] = cipher.update(content, offset, len)
    mac.update(cipherText)
    cipherText
  }

  /**
   * Encrypts or decrypts data in a single-part operation,
   * or finishes a multiple-part operation. The data is encrypted
   * or decrypted, depending on how this crypto was initialized.
   * @param content bytes to be encrypted or decrypted.
   * @return (encrypted or decrypted bytes, Message Authentication Code)
   */
  override def doFinal(content: Array[Byte]): (Array[Byte], Array[Byte]) = {
    val cipherText: Array[Byte] = cipher.doFinal(content)
    val hmac: Array[Byte] = mac.doFinal(cipherText)
    (cipherText, hmac)
  }

  /**
   * Encrypts or decrypts data in a single-part operation,
   * or finishes a multiple-part operation. The data is encrypted
   * or decrypted, depending on how this crypto was initialized.
   * @param content bytes to be encrypted or decrypted.
   * @param offset bytes offset of content.
   * @param len bytes len of content.
   * @return (encrypted or decrypted bytes, Message Authentication Code)
   */
  override def doFinal(content: Array[Byte], offset: Int, len: Int): (Array[Byte], Array[Byte]) = {
    val cipherText: Array[Byte] = cipher.doFinal(content, offset, len)
    val hmac: Array[Byte] = mac.doFinal(cipherText)
    (cipherText, hmac)
  }

  /**
   * Encrypts or decrypts a byte stream. The data is encrypted
   * or decrypted, depending on how this crypto was initialized.
   * @param inputStream input stream
   * @param outputStream output stream
   */
  def doFinal(inputStream: DataInputStream, outputStream: DataOutputStream): Unit = {
    if (opMode == ENCRYPT) {
      encryptStream(inputStream, outputStream)
    } else {
      decryptStream(inputStream, outputStream)
    }
  }

  /**
   * Encrypts or decrypts a file. The data is encrypted
   * or decrypted, depending on how this crypto was initialized.
   * @param binaryFilePath
   * @param savePath
   */
  def doFinal(binaryFilePath: String, savePath: String): Unit = {
    if (opMode == ENCRYPT) {
      encryptFile(binaryFilePath, savePath)
    } else {
      decryptFile(binaryFilePath, savePath)
    }
  }

  val blockSize = 1024 * 1024 // 1m per update
  lazy val byteBuffer = new Array[Byte](blockSize)
  protected def encryptStream(
        inputStream: DataInputStream,
        outputStream: DataOutputStream): Unit = {
    val header = genHeader()
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
  def decryptPart(in: InputStream, buffer: Array[Byte]): Array[Byte] = {
    if (in.available() == 0) {
      return new Array[Byte](0)
    }
    val readLen = in.read(buffer)
    if (in.available() <= hmacSize && in.available() >= 0) {
      val last = new Array[Byte](in.available())
      if (in.available() != 0) {
        in.read(last)
      }
      val inputHmac = buffer.slice(readLen - hmacSize + last.length, readLen) ++ last
      val (lastSlice, streamHmac) = doFinal(buffer, 0, readLen - hmacSize + last.length)
      Log4Error.invalidInputError(!inputHmac.sameElements(streamHmac),
        "hmac not match")
      lastSlice
    } else {
      update(buffer, 0, readLen)
    }
  }

  def getHeader(in: InputStream): (String, Array[Byte]) = {
    val header = read(in, 400)
    val headerBuffer = ByteBuffer.wrap(header)
    val dataKeyCipherTextBytesLength = headerBuffer.getInt
    val initializationVectorLength = headerBuffer.getInt
    val dataKeyCipherTextBytes: Array[Byte] = header.slice(
      4 + 4, 4 + 4 + dataKeyCipherTextBytesLength)
    val initializationVector: Array[Byte] = header.slice(
      4 + 4 + dataKeyCipherTextBytesLength,
      4 + 4 + dataKeyCipherTextBytesLength + initializationVectorLength)
    val encryptedDataKeyStr = new String(dataKeyCipherTextBytes, StandardCharsets.UTF_8)
    (encryptedDataKeyStr, initializationVector)
  }

  protected def decryptStream(
        inputStream: DataInputStream,
        outputStream: DataOutputStream): Unit = {
    val header = read(inputStream, 25)
    verifyHeader(header)
    while (inputStream.available() != 0) {
      val decrypted = decryptPart(inputStream, byteBuffer)
      outputStream.write(decrypted)
    }
    outputStream.flush()
  }

  protected def decryptFile(binaryFilePath: String, savePath: String): Unit = {
    Log4Error.invalidInputError(savePath != null && savePath != "",
      "decrypted file save path should be specified")
    val fs = File.getFileSystem(binaryFilePath)
    val bis = fs.open(new Path(binaryFilePath))
    val outs = fs.create(new Path(savePath))
    decryptStream(bis, outs)
    bis.close()
    outs.close()
  }

  protected def encryptFile(binaryFilePath: String, savePath: String): Unit = {
    Log4Error.invalidInputError(savePath != null && savePath != "",
      "decrypted file save path should be specified")
    val fs = File.getFileSystem(binaryFilePath)
    val bis = fs.open(new Path(binaryFilePath))
    val outs = fs.create(new Path(savePath))
    encryptStream(bis, outs)
    bis.close()
    outs.close()
  }

  protected def read(stream: InputStream, numBytes: Int): Array[Byte] = {
    val retval = new Array[Byte](numBytes)
    val bytesRead: Int = stream.read(retval)
    Log4Error.invalidOperationError(bytesRead == numBytes,
      s"Not enough bits to read!, excepted $numBytes, but got $bytesRead.")
    retval
  }

  /**
   * decrypt big data stream.
   * @param ite stream iterator.
   * @return iterator of String.
   */
  override def decryptBigContent(
    inputStream: InputStream): Iterator[String] = {
    // verifyHeader(read(inputStream, 25))
    new Iterator[String] {
      var cachedArray: Array[String] = null
      var pointer = Int.MaxValue
      var lastString = ""

      override def hasNext: Boolean = {
        inputStream.available() != 0 ||
          (cachedArray != null && pointer < cachedArray.length)
      }

      override def next: String = {
        // return empty string when next is not existed
        if (!hasNext) {
          return ""
        }

        if (cachedArray == null || pointer >= cachedArray.length) {
          Log4Error.invalidOperationError(inputStream.available() != 0,
            "next on empty iterator.")
          val decrypted = decryptPart(inputStream, byteBuffer)
          val currentSplitDecryptString = new String(decrypted)
          val splitDecryptString = lastString + currentSplitDecryptString
          val splitDecryptStringArray = splitDecryptString.split("\n")
          if (splitDecryptString.last == '\n') {
            lastString = ""
            cachedArray = splitDecryptStringArray
          } else {
            lastString = splitDecryptStringArray.last
            cachedArray = splitDecryptStringArray.dropRight(1)
          }
          pointer = 0
        }

        pointer += 1
        cachedArray(pointer - 1)
      }
    }
  }
}

object BigDLEncrypt {

  val COMMON = "common"
  val NATIVE_AES_CBC = "nativeaescbc"

  /**
   * Create encrypter by type string
   */
   def apply(s: String): BigDLEncrypt = {
     s.toLowerCase() match {
       case COMMON =>
         new BigDLEncrypt
       case NATIVE_AES_CBC =>
         new BigDLAESCBCEncrypt
       case _ =>
         Log4Error.invalidInputError(false,
          s"Excepted $COMMON or $NATIVE_AES_CBC " +
          s"in spark.bigdl.encryter.type but got $s")
         null
     }
   }
}
