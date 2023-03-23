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
import com.intel.analytics.bigdl.ppml.crypto.CryptoMode
import java.io._
import java.security.SecureRandom
import java.util.Base64
import javax.crypto.spec.{IvParameterSpec, SecretKeySpec}
import javax.crypto.Cipher

/**
 * BigDL AES CBC crypto for encrypt and decrypt data.
 */
class BigDLAESCBCEncrypt extends BigDLEncrypt {

  // Init a decrypter
  override def init(cryptoMode: CryptoMode, mode: OperationMode,
    dataKeyPlaintext: String, initializationVector: Array[Byte]): Unit = {
    this.initializationVector = initializationVector
    init(cryptoMode, mode, dataKeyPlaintext)
  }

  /**
   * Init this crypto with crypto mode, operation mode and keys.
   * @param cryptoMode cryptoMode to en/decrypt data, such as AES_CBC_PKCS5PADDING.
   * @param mode en/decrypt mode, one of ENCRYPT or DECRYPT.
   * @param dataKeyPlaintext signing key and data key.
   */
  override def init(cryptoMode: CryptoMode, mode: OperationMode, dataKeyPlaintext: String): Unit = {
    opMode = mode
    opMode match {
      case DECRYPT =>
        Log4Error.invalidInputError(initializationVector != null,
          "initializationVector got from input data file is empty")
      case ENCRYPT =>
        val r = new SecureRandom()
        initializationVector = Array.tabulate(16)(_ => (r.nextInt(256) - 128).toByte)
    }
    ivParameterSpec = new IvParameterSpec(initializationVector)
    val dataKeyPlainBytes = Base64.getDecoder().decode(dataKeyPlaintext)
    encryptionKeySpec = new SecretKeySpec(dataKeyPlainBytes,
      AES_CBC_PKCS5PADDING.secretKeyAlgorithm)
    cipher = Cipher.getInstance(AES_CBC_PKCS5PADDING.encryptionAlgorithm)
    cipher.init(opMode.opmode, encryptionKeySpec, ivParameterSpec)
  }

  /**
   * If encrypt data, the header is IV and put return value to the head.
   * @return header bytes
   */
  override def genHeader(): Array[Byte] = {
    Log4Error.invalidOperationError(initializationVector != null,
      s"AES IV does not exist! you should init BigDLEncrypt first.")
    initializationVector
  }

  /**
   * Continues a multiple-part encryption or decryption operation
   * No mac is updated in AES CBC Encrypt
   * (depending on how this crypto was initialized).
   * @param content byte to be encrypted or decrypted.
   * @return encrypted or decrypted bytes.
   */
  override def update(content: Array[Byte]): Array[Byte] = {
    val cipherText: Array[Byte] = cipher.update(content)
    cipherText
  }

  /**
   * Continues a multiple-part encryption or decryption operation
   * (depending on how this crypto was initialized).
   * No mac is updated in AES CBC Encrypt
   * @param content bytes to be encrypted or decrypted.
   * @param offset bytes offset of content.
   * @param len bytes len of content.
   * @return encrypted or decrypted bytes.
   */
  override def update(content: Array[Byte], offset: Int, len: Int): Array[Byte] = {
    val cipherText: Array[Byte] = cipher.update(content, offset, len)
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
    (cipherText, null)
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
    (cipherText, null)
  }

  override protected def encryptStream(
        inputStream: DataInputStream,
        outputStream: DataOutputStream): Unit = {
    val header = genHeader()
    outputStream.write(header)
    while (inputStream.available() > blockSize) {
      val readLen = inputStream.read(byteBuffer)
      outputStream.write(update(byteBuffer, 0, readLen))
    }
    val last = inputStream.read(byteBuffer)
    val (lastSlice, _) = doFinal(byteBuffer, 0, last)
    outputStream.write(lastSlice)
    outputStream.flush()
  }

  override def decryptPart(in: InputStream, buffer: Array[Byte]): Array[Byte] = {
    if (in.available() == 0) {
      return new Array[Byte](0)
    }
    val readLen = in.read(buffer)
    if (in.available() >= 0) {
      val last = new Array[Byte](in.available())
      if (in.available() != 0) {
        in.read(last)
      }
      val (lastSlice, _) = doFinal(buffer, 0, readLen + last.length)
      lastSlice
    } else {
      update(buffer, 0, readLen)
    }
  }

  override def getHeader(in: InputStream): (String, Array[Byte]) = {
    val initializationVector: Array[Byte] = super.read(in, 16)
    (null, initializationVector)
  }
}

