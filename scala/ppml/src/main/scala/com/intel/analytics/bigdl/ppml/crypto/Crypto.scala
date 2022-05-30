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
import com.intel.analytics.bigdl.ppml.utils.Supportive
import org.apache.spark.input.PortableDataStream

import java.io.{DataInputStream, DataOutputStream}
import javax.crypto.Cipher

trait Crypto extends Supportive with Serializable {
  def init(cryptoMode: CryptoMode, mode: OperationMode, dataKeyPlaintext: String): Unit

  def decryptBigContent(ite: Iterator[(String, PortableDataStream)]): Iterator[String]

  def genFileHeader(): Array[Byte]

  def verifyFileHeader(header: Array[Byte]): Unit

  def update(content: Array[Byte]): Array[Byte]

  def update(content: Array[Byte], offset: Int, len: Int): Array[Byte]

  def doFinal(content: Array[Byte]): (Array[Byte], Array[Byte])

  def doFinal(content: Array[Byte], offset: Int, len: Int): (Array[Byte], Array[Byte])

  def encryptStream(inputStream: DataInputStream, outputStream: DataOutputStream): Unit

  def decryptStream(inputStream: DataInputStream, outputStream: DataOutputStream): Unit

  def decryptFile(binaryFilePath: String, savePath: String): Unit

  def encryptFile(binaryFilePath: String, savePath: String): Unit
}

object Crypto {
  def apply(cryptoMode: CryptoMode): Crypto = {
    cryptoMode match {
      case AES_CBC_PKCS5PADDING =>
        new FernetEncrypt()
      case default =>
        throw new EncryptRuntimeException("No such crypto mode!")
    }
  }
}

// object OperationMode extends Enumeration {
//   type OperationMode = Value
//   val ENCRYPT, DECRYPT = Value
// }
sealed trait OperationMode extends Serializable {
  def opmode: Int
}

case object ENCRYPT extends OperationMode {
  override def opmode: Int = Cipher.ENCRYPT_MODE
}
case object DECRYPT extends OperationMode {
  override def opmode: Int = Cipher.DECRYPT_MODE
}

trait CryptoMode extends Serializable {
  def encryptionAlgorithm: String
  def signingAlgorithm: String
  def secretKeyAlgorithm: String
}

object CryptoMode {
   def parse(s: String): CryptoMode = {
     s.toLowerCase() match {
       case "aes/cbc/pkcs5padding" =>
         AES_CBC_PKCS5PADDING
       case "plain_text" =>
         PLAIN_TEXT
     }
   }
}

case object AES_CBC_PKCS5PADDING extends CryptoMode {
  override def encryptionAlgorithm: String = "AES/CBC/PKCS5Padding"

  override def signingAlgorithm: String = "HmacSHA256"

  override def secretKeyAlgorithm: String = "AES"
}

case object PLAIN_TEXT extends CryptoMode {
  override def encryptionAlgorithm: String = "plain_text"

  override def signingAlgorithm: String = ""

  override def secretKeyAlgorithm: String = ""
}

// object CryptoMode extends Enumeration {
//  type CryptoMode = Value
//  val PLAIN_TEXT = value("plain_text", "plain_text")
//  val AES_CBC_PKCS5PADDING = value("AES/CBC/PKCS5Padding", "AES/CBC/PKCS5Padding")
//  val UNKNOWN = value("UNKNOWN", "UNKNOWN")
//  class EncryptModeEnumVal(name: String, val value: String) extends Val(nextId, name)
//  protected final def value(name: String, value: String): EncryptModeEnumVal = {
//    new EncryptModeEnumVal(name, value)
//  }
//  def parse(s: String): Value = {
//    values.find(_.toString.toLowerCase() == s.toLowerCase).getOrElse(CryptoMode.UNKNOWN)
//  }
//}
