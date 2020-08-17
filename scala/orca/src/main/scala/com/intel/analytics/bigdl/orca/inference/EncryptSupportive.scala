/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.inference

import java.io.PrintWriter
import java.util.Base64
import javax.crypto.{Cipher, SecretKeyFactory}
import javax.crypto.spec.{IvParameterSpec, PBEKeySpec, SecretKeySpec}

trait EncryptSupportive {

  def encryptWithAES256(content: String, secret: String, salt: String): String = {
    val iv = Array[Byte](0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    val ivParameterSpec = new IvParameterSpec(iv)
    val secretKeyFactory = SecretKeyFactory.getInstance("PBKDF2WithHmacSHA256")
    val spec = new PBEKeySpec(secret.toCharArray(), salt.getBytes(), 65536, 256)
    val tmp = secretKeyFactory.generateSecret(spec)
    val secretKeySpec = new SecretKeySpec(tmp.getEncoded(), "AES")

    val cipher = Cipher.getInstance("AES/CBC/PKCS5PADDING")
    cipher.init(Cipher.ENCRYPT_MODE, secretKeySpec, ivParameterSpec)
    Base64.getEncoder().encodeToString(cipher.doFinal(content.getBytes("UTF-8")))
  }

  def decryptWithAES256(content: String, secret: String, salt: String): String = {
    val iv = Array[Byte](0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    val ivParameterSpec = new IvParameterSpec(iv)
    val secretKeyFactory = SecretKeyFactory.getInstance("PBKDF2WithHmacSHA256")
    val spec = new PBEKeySpec(secret.toCharArray(), salt.getBytes(), 65536, 256)
    val tmp = secretKeyFactory.generateSecret(spec)
    val secretKeySpec = new SecretKeySpec(tmp.getEncoded(), "AES")

    val cipher = Cipher.getInstance("AES/CBC/PKCS5PADDING")
    cipher.init(Cipher.DECRYPT_MODE, secretKeySpec, ivParameterSpec)
    new String(cipher.doFinal(Base64.getDecoder().decode(content)))
  }

  def encryptFileWithAES256(filePath: String, secret: String, salt: String, outputFile: String,
  encoding: String = "UTF-8")
  : Unit = {
    val source = scala.io.Source.fromFile(filePath, encoding)
    val content = try source.mkString finally source.close()
    val encrypted = encryptWithAES256(content, secret, salt)
    new PrintWriter(outputFile) { write(encrypted); close }
  }

  def decryptFileWithAES256(filePath: String, secret: String, salt: String): String = {
    val source = scala.io.Source.fromFile(filePath)
    val content = try source.mkString finally source.close()
    decryptWithAES256(content, secret, salt)
  }

  def decryptFileWithAES256(filePath: String, secret: String, salt: String, outputFile: String)
  : Unit = {
    val source = scala.io.Source.fromFile(filePath)
    val content = try source.mkString finally source.close()
    val decrypted = decryptWithAES256(content, secret, salt)
    new PrintWriter(outputFile) { write(decrypted); close }
  }

}


