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

package com.intel.analytics.bigdl.ppml.kms.common

import java.nio.charset.StandardCharsets
import java.util.Base64

import org.slf4j.LoggerFactory
import com.intel.analytics.bigdl.dllib.utils.Log4Error

import java.sql.{Connection, DriverManager, ResultSet, SQLException, Statement}

import java.security.MessageDigest

import javax.crypto.spec.SecretKeySpec
import javax.crypto.Cipher
import javax.crypto.{KeyGenerator, SecretKey}

import java.io.File
import java.security.{KeyStore, SecureRandom}
import java.util.concurrent.TimeUnit
import javax.net.ssl.{KeyManagerFactory, SSLContext, TrustManagerFactory}
import akka.http.scaladsl.ConnectionContext
import com.codahale.shamir.Scheme
import com.intel.analytics.bigdl.ppml.utils.Supportive
import com.intel.analytics.bigdl.ppml.attestation.utils.JsonUtil

object BigDLKMServerUtil extends Supportive {
  val logger = LoggerFactory.getLogger(getClass)
  def md5(data: String): String = {
    val md = MessageDigest.getInstance("MD5")
    md.update(data.getBytes(StandardCharsets.UTF_8))
    val digest: Array[Byte] = md.digest
    val hash: String = Base64.getEncoder().encodeToString(digest)
    hash
  }

  def insertDB(statement: String, url: String): Unit = {
    logger.info(s"Insert statment: $statement")
    val connection = DriverManager.getConnection(url)
    connection.createStatement().executeUpdate(statement)
    if(connection != null) connection.close()
  }

  def queryDB(statement: String, url: String,
    columnName: String): Option[String] = {
      logger.info(s"Query statment: $statement")
      val connection = DriverManager.getConnection(url)
      val rs = connection.createStatement().executeQuery(statement)
      val value = if (rs.next) rs.getString(columnName) else null
      if(connection != null) connection.close()
      Some(value)
  }

  def login(user: String, token: String, url: String): Unit = {
    val userHash = md5(user)
    val statement = s"select * from user where name='$userHash'"
    val tokenHashFromDB = queryDB(statement, url, "token").get
    val userProvidedTokenHash = md5(token)
    Log4Error.invalidOperationError(tokenHashFromDB != null
      && tokenHashFromDB == userProvidedTokenHash, "wrong token or user")
  }

  def saveKey2DB(user: String, keyName: String,
    encryptedKey: String, url: String): Unit = {
      val (userHash, keyNameHash) = (md5(user), md5(keyName))
      val statement = s"insert into key select '$userHash', '$keyNameHash', '$encryptedKey' " +
        s"where not exists(select 1 from key where user='$userHash' and name='$keyNameHash')"
      insertDB(statement, url)
  }

  def saveUser2DB(user: String, token: String, url: String): Unit = {
      val (userHash, tokenHash) = (md5(user), md5(token))
      val statement = s"insert into user select '$userHash', '$tokenHash' " +
        s"where not exists(select 1 from user where name='$userHash')"
      insertDB(statement, url)
  }

  def queryKeyFromDB(user: String, keyName: String,
    url: String): Option[String] = {
      val (userHash, keyNameHash) = (md5(user), md5(keyName))
      val statement = s"select * from key where user='$userHash' and name='$keyNameHash'"
      val encryptedKey = queryDB(statement, url, "data")
      encryptedKey
  }

  def keyCryptoCodec(base64WrappingKeyPlainText: String,
    base64ChildKey: String, om: Int): String = {
      Log4Error.invalidOperationError(base64WrappingKeyPlainText != null &&
        base64WrappingKeyPlainText != "", "empty encryption key!")
      val wrappingKeyBytes = Base64.getDecoder().decode(base64WrappingKeyPlainText)
      val encryptionKeySpec = new SecretKeySpec(wrappingKeyBytes, "AES")
      val cipher = Cipher.getInstance("AES")
      cipher.init(om, encryptionKeySpec)
      val childKeyBytes = Base64.getDecoder().decode(base64ChildKey)
      val operatedChildKeyBytes = cipher.doFinal(childKeyBytes)
      val base64OperatedChildKey = Base64.getEncoder.encodeToString(operatedChildKeyBytes)
      base64OperatedChildKey
  }

  def encryptKey(wrappingKeyPlianText: String, childKeyPlainText: String): String = {
    keyCryptoCodec(wrappingKeyPlianText,
      childKeyPlainText, Cipher.ENCRYPT_MODE)
  }

  def decryptKey(wrappingKeyPlianText: String, childKeyCipherText: String): String = {
    keyCryptoCodec(wrappingKeyPlianText,
      childKeyCipherText, Cipher.DECRYPT_MODE)
  }

  def generateAESKey(keySize: Int): String = {
    val generator = KeyGenerator.getInstance("AES")
    generator.init(keySize, SecureRandom.getInstanceStrong())
    val key: SecretKey = generator.generateKey()
    val base64Key: String = Base64.getEncoder().encodeToString(key.getEncoded())
    base64Key
  }

  def defineServerContext(httpsKeyStoreToken: String,
                          httpsKeyStorePath: String): ConnectionContext = {
    val token = httpsKeyStoreToken.toCharArray
    val keyStore = KeyStore.getInstance("PKCS12")
    val keystoreInputStream = new File(httpsKeyStorePath).toURI().toURL().openStream()
    Log4Error.invalidOperationError(keystoreInputStream != null, "Keystore required!")
    keyStore.load(keystoreInputStream, token)
    val keyManagerFactory = KeyManagerFactory.getInstance("SunX509")
    keyManagerFactory.init(keyStore, token)
    val trustManagerFactory = TrustManagerFactory.getInstance("SunX509")
    trustManagerFactory.init(keyStore)
    val sslContext = SSLContext.getInstance("TLS")
    sslContext.init(keyManagerFactory.getKeyManagers,
      trustManagerFactory.getTrustManagers, new SecureRandom)
    ConnectionContext.https(sslContext)
  }

  def splitRootKey(rootKey: String,
    threshold: Int): String = {
      Log4Error.invalidOperationError(rootKey != null && rootKey != "",
        "try to split an empty root key string!")
      // need at least $threshold of 5 secrets to restore the root key
      val scheme = new Scheme(new SecureRandom(), 5, threshold)
      import scala.collection.JavaConverters._
      "{" + scheme.split(rootKey.getBytes(StandardCharsets.UTF_8))
        .asScala
        .map {
          case (k, v) => JsonUtil.toJson(SecretFormat(k,
            Base64.getEncoder.encodeToString(v)))
        }.mkString(",") + "}"
  }

  def recoverRootKey(secretStore: java.util.Map[Integer, Array[Byte]],
    threshold: Int): String = {
      // need at least $threshold of 5 secret to restore the root key
      val scheme = new Scheme(new SecureRandom(), 5, threshold)
      new String(scheme.join(secretStore), StandardCharsets.UTF_8)
  }

  def isBase64(s: String): Boolean = {
    org.apache.commons.codec.binary.Base64.isBase64(s)
  }
}

class SecretStore {
  val secretStoreMap = new java.util.HashMap[Integer, Array[Byte]]
  def addSecret(secretJsonStr: String): Unit = {
    val secret = JsonUtil.fromJson(classOf[SecretFormat], secretJsonStr)
    secretStoreMap.put(secret.index,
      Base64.getDecoder.decode(secret.content))
  }
  def getSecrets(): java.util.HashMap[Integer, Array[Byte]] = secretStoreMap
  def count(): Int = secretStoreMap.size
  def clear(): Unit = secretStoreMap.clear
}

case class SecretFormat(index: Integer, content: String)
