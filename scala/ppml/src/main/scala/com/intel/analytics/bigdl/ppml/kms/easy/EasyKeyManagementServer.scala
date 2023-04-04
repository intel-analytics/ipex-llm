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

package com.intel.analytics.bigdl.ppml.kms.frontend

import java.io.File
import java.security.{KeyStore, SecureRandom}
import java.util.concurrent.TimeUnit
import javax.net.ssl.{KeyManagerFactory, SSLContext, TrustManagerFactory}

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
import javax.crypto.spec.SecretKeySpec
import javax.crypto.Cipher

import com.intel.analytics.bigdl.ppml.utils.Supportive

object EasyKeyManagementServer extends Supportive {
  val logger = LoggerFactory.getLogger(getClass)
  val name = "Easy Key Management Server"
  implicit val system = ActorSystem(name)
  implicit val materializer = ActorMaterializer()
  implicit val executionContext = system.dispatcher
  implicit val (encrypter, decrypter) = {
    val base64RootK = sys.env("ROOTKEY")
    Log4Error.invalidOperationError(base64RootK != "",
      "Excepted ROOTKEY but found it empty, please upload it as k8s secret")
    val kSpec = new SecretKeySpec(byteBase64.getDecoder().decode(base64RootK), "AES")
    (createCipher(kSpec, Cipher.ENCRYPT_MODE), createCipher(kSpec, Cipher.DECRYPT_MODE))
  }
  implicit val jdbc = {

  }

  def main(args: Array[String]): Unit = {
      val arguments = timing("parse arguments") {
        argumentsParser.parse(args, EasyKeyManagementArguments()) match {
          case Some(arguments) => logger.info(s"starting with $arguments"); arguments
          case None => argumentsParser.failure("miss args, please see the usage info"); null
        }
      }
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
                  login(user, token)
                  val base64AES256Key: String = generateAESKey(256)
                  saveK2DB(user, primaryKeyName, base64AES256Key)
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
                  login(user, token)
                  val encryptedDataKey: String = {
                    val primaryKey: String = queryKFromDB(user, primaryKeyName)
                    val base64AES128Key: String = generateAESKey(128)
                    encryptDataKey(primaryKey, base64AES128Key)
                  }
                  saveK2DB(user, dataKeyName, encryptedDataKey)
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
                    saveUser2DB(userName, token)
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
                      login(user, token)
                      val base64DataKeyPlainText: String = {
                        val primaryKey = getKeyFromDB(user, primaryKeyName)
                        val base64DataKeyCipherText = getKeyFromDB(user, dataKeyName)
                        decryptDataKey(primaryKey, base64DataKeyCipherText)
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

  def login(user: String, token: String): Unit = {

  }

  def saveK2DB(user, primaryKeyName, base64AES256Key): Unit = {

  }

  def saveUser2DB(userName, token): Unit = {

  }

  def queryKFromDB(user, primaryKeyName): Unit = {

  }

  def createCipher(kSpec: SecretKeySpec, om: Int): Cipher = {
    val cipher = Cipher.getInstance("AES")
    cipher.init(om, kSpec)
    cipher
  }

  def dataKeyCryptoCodec(base64PrimaryKeyPlainText: String,
    base64DataKey: String, om: Int): String = {
    val bytePrimaryKeyPlainText = Base64.getDecoder().decode(base64PrimaryKeyPlainText)
    val encryptionKeySpec = new SecretKeySpec(bytePrimaryKeyPlainText, "AES")
    val cipher = createCipher(encryptionKeySpec, om)
    val byteDataKey = Base64.getDecoder().decode(base64DataKey)
    val byteDataKeyOperated = cipher.doFinal(byteDataKey)
    val base64DataKeyOperated = Base64.getEncoder.encodeToString(byteDataKeyOperated)
    base64DataKeyOperated
  }

  def encryptDataKey(base64PrimaryKeyPlainText: String,
    base64DataKeyPlainText: String): String = {
    dataKeyCryptoCodec(base64PrimaryKeyPlainText,
      base64DataKeyPlainText, Cipher.ENCRYPT_MODE)
  }

  def decryptDataKey(base64PrimaryKeyPlainText: String,
    base64DataKeyCipherText: String): String = {
    dataKeyCryptoCodec(base64PrimaryKeyPlainText,
      base64DataKeyCipherText, Cipher.DECRYPT_MODE)
  }

  val argumentsParser =
   new scopt.OptionParser[BigDLKMSFrontendArguments](name) {
    head(name)
    opt[String]('i', "ip")
      .action((x, c) => c.copy(ip = x))
      .text(s"ip of $name")
    opt[Int]('p', "port")
      .action((x, c) => c.copy(port = x))
      .text(s"port of $name")
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

case class BigDLKMSFrontendArguments(
    ip: String = "0.0.0.0",
    port: Int = 9875,
    httpsKeyStorePath: String = null,
    httpsKeyStoreToken: String = null
)

