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

package com.intel.analytics.bigdl.ppml.kms

import java.io.File
import java.security.{KeyStore, SecureRandom}
import java.util.concurrent.TimeUnit
import java.sql.{Connection, DriverManager, ResultSet, SQLException, Statement}
import java.security.MessageDigest 
import javax.net.ssl.{KeyManagerFactory, SSLContext, TrustManagerFactory}

import java.security.SecureRandom
import javax.crypto.{KeyGenerator, SecretKey}

import akka.actor.{ActorRef, ActorSystem, Props}
import akka.http.scaladsl.{ConnectionContext, Http}
import akka.http.scaladsl.server.Directives.{complete, path, _}
import akka.pattern.ask
import akka.stream.ActorMaterializer
import akka.util.Timeout
import org.slf4j.LoggerFactory

import scala.collection.mutable
import scala.concurrent.Await
import com.intel.analytics.bigdl.dllib.utils.Log4Error
import sys.process._
import java.util.Base64
import java.nio.charset.StandardCharsets
import javax.crypto.spec.SecretKeySpec
import javax.crypto.Cipher

import com.intel.analytics.bigdl.ppml.utils.Supportive

object EasyKeyManagementServer extends Supportive {
  val logger = LoggerFactory.getLogger(getClass)
  Class.forName("org.sqlite.JDBC")
  val name = "easy-key-management-server"
  implicit val system = ActorSystem(name)
  implicit val materializer = ActorMaterializer()
  implicit val executionContext = system.dispatcher
  val rootKey = sys.env("ROOT_KEY")
  Log4Error.invalidOperationError(rootKey != "",
    "Excepted ROOTKEY but found it empty, please upload it as k8s secret")

  val md = MessageDigest.getInstance("MD5")

  def main(args: Array[String]): Unit = {
    val arguments = timing("parse arguments") {
      argumentsParser.parse(args, EasyKeyManagementArguments()) match {
        case Some(arguments) => logger.info(s"starting with $arguments"); arguments
        case None => argumentsParser.failure("miss args, please see the usage info"); null
      }
    }
    val url = "jdbc:sqlite:" + arguments.dbFilePath
    val route = timing("initialize https route") {
      path("") {
        timing("welcome") {
          val response = s"welcome to $name \n \n" +
          "create a user like: " +
          "POST /user/{userName}?token=a_token_string_for_the_user \n" +
          "create a primary key like: " +
          "POST /primaryKey/{primaryKeyName}?user=your_username&&token=your_token \n" +
          "create a data key like: " +
          "POST /dataKey/{dataKeyName}?" +
          "primaryKeyName=the_primary_key_name&&user=your_username&&token=your_token \n" +
          "get the data key like: " +
          "GET /dataKey/{dataKeyName}?" +
          "primaryKeyName=the_primary_key_name&&user=your_username&&token=your_token"
          complete(response)
        }
      } ~ path("primaryKey" / Segment) { primaryKeyName =>
        post {
          parameters("user", "token") {
            (user, token) => { timing("generate primary key") {
              try {
                login(user, token, url)
                val encryptedPrimaryKey = {
                  val base64AES256Key: String = generateAESKey(256)
                  encryptKey(rootKey, base64AES256Key)
                }
                saveKey2DB(user, primaryKeyName, encryptedPrimaryKey, url)
                complete(s"generate primaryKey [$primaryKeyName] successfully!")
              } catch {
                case e: Exception =>
                  e.printStackTrace()
                  complete(500, e.getMessage + "\n please get a primary key like: " +
                    "POST /primaryKey/{primaryKeyName}?user=your_username&&token=your_token")
              }
             }
            }
          }
        }
      } ~ path("dataKey" / Segment) { dataKeyName =>
        post {
          parameters("primaryKeyName", "user", "token") {
            (primaryKeyName, user, token) => { timing("generate data key") {
              try {
                login(user, token, url)
                val encryptedDataKey: String = {
                  val encryptedPrimaryKey: String = queryKeyFromDB(user,
                    primaryKeyName, url).get
                  Log4Error.invalidOperationError(encryptedPrimaryKey != null,
                    "wrong primary key")
                  val primaryKeyPlainText = decryptKey(rootKey, encryptedPrimaryKey)
                  val base64AES128Key: String = generateAESKey(128)
                  encryptKey(primaryKeyPlainText, base64AES128Key)
                }
                saveKey2DB(user, dataKeyName, encryptedDataKey, url)
                complete(s"dataKey [$dataKeyName] is generated successfully!")
              } catch {
                case e: Exception =>
                  e.printStackTrace()
                  complete(500, e.getMessage + "\n please get a data key like: " +
                    "POST /dataKey/{dataKeyName}?primaryKeyName=the_primary_key_name" +
                    "&&user=your_username&&token=your_token")
              }
             }
            }
          }
        }
      } ~ path("user" / Segment) { userName =>
        post {
          parameters("token") {
            (token) => { timing("enroll") {
              try {
                saveUser2DB(userName, token, url)
                complete(s"user [$userName] is created successfully!")
              } catch {
                case e: Exception =>
                  e.printStackTrace()
                  complete(500, e.getMessage + "\n please create a user like: " +
                    "POST /user/{userName}?token=a_token_for_the_user")
              }
             }
            }
          }
        }
      } ~ path("dataKey" / Segment) { dataKeyName =>
        get {
          parameters("primaryKeyName", "user", "token") {
            (primaryKeyName, user, token) => { timing("get data key") {
              try {
                login(user, token, url)
                val base64DataKeyPlainText: String = {
                  val encryptedPrimaryKey = queryKeyFromDB(user,
                    primaryKeyName, url).get
                  Log4Error.invalidOperationError(encryptedPrimaryKey != null,
                    "wrong primary key")
                  val encryptedDataKey = queryKeyFromDB(user,
                    dataKeyName, url).get
                  val primaryKeyPlainText = decryptKey(rootKey, encryptedPrimaryKey)
                  decryptKey(primaryKeyPlainText, encryptedDataKey)
                }
                complete(base64DataKeyPlainText)
              } catch {
                case e: Exception =>
                  e.printStackTrace()
                  complete(500, e.getMessage + "\n please get the data key like: " +
                    "GET /dataKey/{dataKeyName}?primaryKeyName=the_primary_key_name" +
                    "&&user=your_username&&token=your_token")
              }
             }
            }
          }
        }
      }
    }

      val serverContext = defineServerContext(arguments.httpsKeyStoreToken,
        arguments.httpsKeyStorePath)
      Http().bindAndHandle(route, arguments.ip, port = arguments.port,
        connectionContext = serverContext)
      logger.info(s"$name started at https://${arguments.ip}:${arguments.port}")
  }

  def md5(data: String): String = {
    md.update(data.getBytes(StandardCharsets.UTF_8))
    val digest: Array[Byte] = md.digest
    val hash: String = Base64.getEncoder().encodeToString(digest)
    hash
  }

  def insertDB(statement: String, url: String): Unit = {
    logger.info(s"[INFO] insert statment: $statement")
    val connection = DriverManager.getConnection(url)
    connection.createStatement().executeUpdate(statement)
    if(connection != null) connection.close()
  }

  def queryDB(statement: String, url: String,
    columnName: String): Option[String] = {
      logger.info(s"[INFO] query statment: $statement")
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
    println(s"[INFO] tokenHashFromDB:$tokenHashFromDB, userProvidedTokenHash:$userProvidedTokenHash")
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
    val generator = KeyGenerator.getInstance("AES");
    generator.init(keySize, SecureRandom.getInstanceStrong());
    val key: SecretKey = generator.generateKey();
    val base64Key: String = Base64.getEncoder().encodeToString(key.getEncoded());
    return base64Key;
  }

  val argumentsParser =
   new scopt.OptionParser[EasyKeyManagementArguments](name) {
    head(name)
    opt[String]('i', "ip")
      .action((x, c) => c.copy(ip = x))
      .text(s"ip of $name")
    opt[Int]('p', "port")
      .action((x, c) => c.copy(port = x))
      .text(s"port of $name")
    opt[String]('p', "dbFilePath")
      .action((x, c) => c.copy(dbFilePath = x))
      .text("database file path of KMS storage")
    opt[String]('p', "httpsKeyStorePath")
      .action((x, c) => c.copy(httpsKeyStorePath = x))
      .text("https keyStore path")
    opt[String]('w', "httpsKeyStoreToken")
      .action((x, c) => c.copy(httpsKeyStoreToken = x))
      .text("https keyStore token")
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
}

case class EasyKeyManagementArguments(
  ip: String = "0.0.0.0",
  port: Int = 9875,
  dbFilePath: String = "/ppml/data/kms.db",
  httpsKeyStorePath: String = null,
  httpsKeyStoreToken: String = null
)

