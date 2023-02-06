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
package com.intel.analytics.bigdl.ppml.attestation.utils

import java.math.BigInteger
import java.io.{File, InputStream, PrintWriter}
import java.io.{FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}
import java.io.{BufferedOutputStream, BufferedInputStream};
import javax.crypto.{Cipher, SecretKey, SecretKeyFactory}
import javax.crypto.spec.{PBEKeySpec, SecretKeySpec}
import org.json4s.jackson.Serialization
import org.json4s._

import scala.concurrent.{Await, Future, ExecutionContext}
import scala.concurrent.duration._
import scala.util.parsing.json._

object FileEncryptUtil {
  implicit val ec: ExecutionContext = ExecutionContext.global
  private val salt = Array[Byte](0, 1, 2, 3, 4, 5, 6, 7)
  private val iterations = 65536
  private val keySize = 256
  private val algorithm = "PBKDF2WithHmacSHA256"
  def encrypt(data: Array[Byte], secretKey: String): Array[Byte] = {
    val factory = SecretKeyFactory.getInstance(algorithm)
    val spec = new PBEKeySpec(secretKey.toCharArray, salt, iterations, keySize)
    val key = factory.generateSecret(spec).getEncoded()
    val keySpec = new SecretKeySpec(key, "AES")
    val cipher = Cipher.getInstance("AES/ECB/PKCS5Padding")
    cipher.init(Cipher.ENCRYPT_MODE, keySpec)
    cipher.doFinal(data)
  }

  def decrypt(encryptedData: Array[Byte], secretKey: String): Array[Byte] = {
    val factory = SecretKeyFactory.getInstance(algorithm)
    val spec = new PBEKeySpec(secretKey.toCharArray, salt, iterations, keySize)
    val key = factory.generateSecret(spec).getEncoded()
    val keySpec = new SecretKeySpec(key, "AES")
    val cipher = Cipher.getInstance("AES/ECB/PKCS5Padding")
    cipher.init(Cipher.DECRYPT_MODE, keySpec)
    cipher.doFinal(encryptedData)
  }

  def saveFile(filename: String, content: String, secretKey: String): Future[Unit] = Future {
    val file = new File(filename)
    if (!file.exists()) {
      file.createNewFile()
    }
    val encryptedContent = encrypt(content.getBytes("UTF-8"), secretKey)
    val out = new BufferedOutputStream(new FileOutputStream(file))
    out.write(encryptedContent)
    out.close()
  }

  def loadFile(filename: String, secretKey: String): Future[String] = Future {
    val file = new File(filename)
    if (!file.exists()) {
      ""
    } else {
      val in = new FileInputStream(file)
      val bufIn = new BufferedInputStream(in)
      val encryptedContent =
        Iterator.continually(bufIn.read()).takeWhile(_ != -1).map(_.toByte).toArray
      bufIn.close()
      in.close()
      val decryptedContent = new String(decrypt(encryptedContent, secretKey), "UTF-8")
      decryptedContent
    }
  }
}