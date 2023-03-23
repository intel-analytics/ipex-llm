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

import com.intel.analytics.bigdl.dllib.nn.NormMode.Value
import com.intel.analytics.bigdl.dllib.utils.Log4Error
import com.intel.analytics.bigdl.ppml.utils.Supportive
import org.apache.spark.input.PortableDataStream

import java.io.{DataInputStream, DataOutputStream, InputStream}
import javax.crypto.Cipher

/**
 * BigDL crypto interface for encrypt and decrypt data.
 */
trait Crypto extends Supportive with Serializable {
  /**
   * Init this crypto with crypto mode, operation mode and keys.
   * @param cryptoMode cryptoMode to en/decrypt data, such as AES_CBC_PKCS5PADDING.
   * @param mode en/decrypt mode, one of ENCRYPT or DECRYPT.
   * @param dataKeyPlaintext signing key and data key.
   */
  def init(cryptoMode: CryptoMode, mode: OperationMode, dataKeyPlaintext: String): Unit

  /**
   * decrypt big data stream.
   * @param ite stream iterator.
   * @return iterator of String.
   */
  def decryptBigContent(in: InputStream): Iterator[String]

  /**
   * If encrypt data, should generate header and put return value to the head.
   * @return header bytes
   */
  def genHeader(): Array[Byte]

  /**
   * Verify the header bytes when decrypt.
   * @param header header bytes
   */
  def verifyHeader(header: Array[Byte]): Unit

  def getHeader(in: InputStream): (String, Array[Byte])

  /**
   * Verify the header bytes when decrypt.
   * @param header header bytes
   */
  def verifyHeader(in: InputStream): Unit

  /**
   * Continues a multiple-part encryption or decryption operation
   * (depending on how this crypto was initialized).
   * @param content byte to be encrypted or decrypted.
   * @return encrypted or decrypted bytes.
   */
  def update(content: Array[Byte]): Array[Byte]

  /**
   * Continues a multiple-part encryption or decryption operation
   * (depending on how this crypto was initialized).
   * @param content bytes to be encrypted or decrypted.
   * @param offset bytes offset of content.
   * @param len bytes len of content.
   * @return encrypted or decrypted bytes.
   */
  def update(content: Array[Byte], offset: Int, len: Int): Array[Byte]

  /**
   * Encrypts or decrypts data in a single-part operation,
   * or finishes a multiple-part operation. The data is encrypted
   * or decrypted, depending on how this crypto was initialized.
   * @param content bytes to be encrypted or decrypted.
   * @return (encrypted or decrypted bytes, Message Authentication Code)
   */
  def doFinal(content: Array[Byte]): (Array[Byte], Array[Byte])

  /**
   * Encrypts or decrypts data in a single-part operation,
   * or finishes a multiple-part operation. The data is encrypted
   * or decrypted, depending on how this crypto was initialized.
   * @param content bytes to be encrypted or decrypted.
   * @param offset bytes offset of content.
   * @param len bytes len of content.
   * @return (encrypted or decrypted bytes, Message Authentication Code)
   */
  def doFinal(content: Array[Byte], offset: Int, len: Int): (Array[Byte], Array[Byte])

  /**
   * Encrypts or decrypts a byte stream. The data is encrypted
   * or decrypted, depending on how this crypto was initialized.
   * @param inputStream input stream
   * @param outputStream output stream
   */
  def doFinal(inputStream: DataInputStream, outputStream: DataOutputStream): Unit

  /**
   * Encrypts or decrypts a file. The data is encrypted
   * or decrypted, depending on how this crypto was initialized.
   * @param binaryFilePath
   * @param savePath
   */
  def doFinal(binaryFilePath: String, savePath: String): Unit
}

object Crypto {
  /**
   * create a crypto with cryptoMode.
   * @param cryptoMode CryptoMode
   * @return crypto
   */
  def apply(cryptoMode: CryptoMode): Crypto = {
    val supportedCryptos = Array(AES_CBC_PKCS5PADDING)
    Log4Error.unKnowExceptionError(supportedCryptos.contains(cryptoMode),
      s"unsupported cryptoMode, $cryptoMode")
    cryptoMode match {
      case AES_CBC_PKCS5PADDING =>
        new BigDLEncrypt()
    }
  }
}

/**
 * Operation Mode for crypto
 */
sealed trait OperationMode extends Serializable {
  def opmode: Int
}

/**
 * Encrypt Operation Mode for crypto
 */
case object ENCRYPT extends OperationMode {
  override def opmode: Int = Cipher.ENCRYPT_MODE
}

/**
 * Decrypt Operation Mode for crypto
 */
case object DECRYPT extends OperationMode {
  override def opmode: Int = Cipher.DECRYPT_MODE
}

/**
 * CryptoMode
 */
trait CryptoMode extends Serializable {
  /**
   * encryption algorithm of crypto
   * @return encryption algorithm.
   */
  def encryptionAlgorithm: String

  /**
   * signing algorithm of crypto
   * @return signing algorithm.
   */
  def signingAlgorithm: String

  /**
   * secret key algorithm of crypto.
   * @return secret key algorithm.
   */
  def secretKeyAlgorithm: String
}

object CryptoMode {
  /**
   * Create cryptoMode by string.
   */
   def parse(s: String): CryptoMode = {
     s.toLowerCase() match {
       case "aes/cbc/pkcs5padding" =>
         AES_CBC_PKCS5PADDING
       case "plain_text" =>
         PLAIN_TEXT
       case "aes_gcm_v1" =>
         AES_GCM_V1
       case "aes_gcm_ctr_v1" =>
         AES_GCM_CTR_V1
     }
   }
}

/**
 * CryptoMode AES_CBC_PKCS5PADDING.
 */
case object AES_CBC_PKCS5PADDING extends CryptoMode {
  /**
   * encryption algorithm of crypto
   * @return AES/CBC/PKCS5Padding
   */
  override def encryptionAlgorithm: String = "AES/CBC/PKCS5Padding"

  /**
   * signing algorithm of crypto
   * @return HmacSHA256
   */
  override def signingAlgorithm: String = "HmacSHA256"

  /**
   * secret key algorithm of crypto.
   * @return AES
   */
  override def secretKeyAlgorithm: String = "AES"
}

/**
 * CryptoMode AES_GCM_V1 for parquet only
 */
case object AES_GCM_V1 extends CryptoMode {
  /**
   * encryption algorithm of crypto
   *
   * @return AES/CBC/PKCS5Padding
   */
  override def encryptionAlgorithm: String = "AES_GCM_V1"

  override def signingAlgorithm: String = ""

  override def secretKeyAlgorithm: String = "AES"
}

/**
 * CryptoMode AES_GCM_CTR_V1 for parquet only
 */
case object AES_GCM_CTR_V1 extends CryptoMode {
  /**
   * encryption algorithm of crypto
   *
   * @return AES/CBC/PKCS5Padding
   */
  override def encryptionAlgorithm: String = "AES_GCM_CTR_V1"

  override def signingAlgorithm: String = ""

  override def secretKeyAlgorithm: String = "AES"
}


/**
 * CryptoMode PLAIN_TEXT.
 */
case object PLAIN_TEXT extends CryptoMode {
  /**
   * encryption algorithm of crypto
   * @return plain_text
   */
  override def encryptionAlgorithm: String = "plain_text"

  /**
   * signing algorithm of crypto
   * @return
   */
  override def signingAlgorithm: String = ""

  /**
   * secret key algorithm of crypto.
   * @return
   */
  override def secretKeyAlgorithm: String = ""
}

