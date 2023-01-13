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
import java.lang.Object
import java.net.{URL, URLClassLoader}
import java.lang.reflect.Method
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

object BigDLKMSFrontend extends Supportive {
  val logger = LoggerFactory.getLogger(getClass)

  val name = "BigDL KMS Frontend"

  implicit val system = ActorSystem("bigdl-kms-frontend-system")
  implicit val materializer = ActorMaterializer()
  implicit val executionContext = system.dispatcher
  implicit val timeout: Timeout = Timeout(100, TimeUnit.SECONDS)

  def main(args: Array[String]): Unit = {
      val arguments = timing("parse arguments") {
        argumentsParser.parse(args, BigDLKMSFrontendArguments()) match {
          case Some(arguments) => logger.info(s"starting with $arguments"); arguments
          case None => argumentsParser.failure("miss args, please see the usage info"); null
        }
      }
      val (keywhizCliMethod, keywhizCliInstance) = externalClassMethodLoader(
        arguments.keywhizCliJarPath,
        "keywhiz.cli.CliMain",
        "main")
      val (keyProviderMethod, keyProviderInstance) = externalClassMethodLoader(
        arguments.keyProviderJarPath,
        "keywhiz.KeywhizService",
        "main")

      val route = timing("initialize http route") {
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
            (user, token) => {
            timing("generate primary key") {
            try {
              val base64AES256Key: String = generateAESKey(256,
                                                           keyProviderMethod,
                                                           keyProviderInstance)
              loginKeywhiz(user, token, keywhizCliMethod, keywhizCliInstance)
              addKeyToKeywhiz(user,
                              primaryKeyName,
                              base64AES256Key,
                              keywhizCliMethod,
                              keywhizCliInstance)
              complete(s"primaryKey [$primaryKeyName] is generated successfully!")
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
            (primaryKeyName, user, token) => {
            timing("generate data key") {
            try {
                loginKeywhiz(user, token, keywhizCliMethod, keywhizCliInstance)
                val primaryKey: String = getKeyFromKeywhiz(user,
                                                           primaryKeyName,
                                                           keywhizCliMethod,
                                                           keywhizCliInstance)
                val base64AES128Key: String = generateAESKey(128,
                                                             keyProviderMethod,
                                                             keyProviderInstance)
                val encryptedDataKey: String = encryptDataKey(primaryKey, base64AES128Key)
                addKeyToKeywhiz(user,
                                dataKeyName,
                                encryptedDataKey,
                                keywhizCliMethod,
                                keywhizCliInstance)
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
                (token) => {
                timing("enroll") {
                try {
                   createUserToKeywhiz(userName,
                                       token,
                                       arguments.frontendKeywhizConf,
                                       keywhizCliMethod,
                                       keywhizCliInstance)
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
            (primaryKeyName, user, token) => {
            timing("get data key") {
            try {
                loginKeywhiz(user, token, keywhizCliMethod, keywhizCliInstance)
                val primaryKey = getKeyFromKeywhiz(user,
                                                   primaryKeyName,
                                                   keywhizCliMethod,
                                                   keywhizCliInstance)
                val base64DataKeyCiphertext = getKeyFromKeywhiz(user,
                                                                dataKeyName,
                                                                keywhizCliMethod,
                                                                keywhizCliInstance)
                val base64DataKeyPlaintext = decryptDataKey(primaryKey, base64DataKeyCiphertext)
                complete(base64DataKeyPlaintext)
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
      Http().bindAndHandle(route, arguments.interface, port = arguments.port,
          connectionContext = serverContext)
      logger.info(s"https started at https://${arguments.interface}:${arguments.port}")
  }

  def createUserToKeywhiz(user: String,
                          token: String,
                          frontendKeywhizConf: String,
                          keywhizCliMethod: Method,
                          keywhizCliInstance: URLClassLoader): Unit = {
    val args: String = constructKeywhizArgs(
        s"add-user $frontendKeywhizConf --user $user --token $token")
    invokeExternalMethod(keywhizCliMethod, keywhizCliInstance, args)
  }

  def loginKeywhiz(user: String,
                   token: String,
                   keywhizCliMethod: Method,
                   keywhizCliInstance: URLClassLoader): Unit = {
    val args: String = constructKeywhizArgs(
        s"--user $user --token $token login")
    invokeExternalMethod(keywhizCliMethod, keywhizCliInstance, args)
  }

  def generateAESKey(keysize: Int,
                     keyProviderMethod: Method,
                     keyProviderInstance: URLClassLoader): String = {
    val key = invokeExternalMethod(
        keyProviderMethod,
        keyProviderInstance,
        s"gen-aes --keysize $keysize").asInstanceOf[String].dropRight(1)
    key
  }

  def addKeyToKeywhiz(user: String,
                      keyName: String,
                      keyContent: String,
                      keywhizCliMethod: Method,
                      keywhizCliInstance: URLClassLoader): Unit = {
    val args: String = constructKeywhizArgs(
        s"--user $user " +
        s"add secret --name $keyName --datakey $keyContent"
    )
    invokeExternalMethod(keywhizCliMethod, keywhizCliInstance, args)
  }

  def getKeyFromKeywhiz(user: String,
                        keyName: String,
                        keywhizCliMethod: Method,
                        keywhizCliInstance: URLClassLoader): String = {
    val args: String = constructKeywhizArgs(
        s"--user $user get --name $keyName"
    )
    val key: String = invokeExternalMethod(
        keywhizCliMethod,
        keywhizCliInstance,
        args).asInstanceOf[String].dropRight(1)
    key
  }

  def dataKeyCryptoCodec(base64PrimaryKeyPlaintext: String,
                     base64DataKey: String,
                     om: Int): String = {
      val bytePrimaryKeyPlaintext = Base64.getDecoder().decode(base64PrimaryKeyPlaintext)
      val encryptionKeySpec = new SecretKeySpec(bytePrimaryKeyPlaintext, "AES")
      val cipher = Cipher.getInstance("AES")
      cipher.init(om, encryptionKeySpec)
      val byteDataKey = Base64.getDecoder().decode(base64DataKey)
      val byteDataKeyOperated = cipher.doFinal(byteDataKey)
      val base64DataKeyOperated = Base64.getEncoder.encodeToString(byteDataKeyOperated)
      base64DataKeyOperated
  }

  def encryptDataKey(base64PrimaryKeyPlaintext: String,
                     base64DataKeyPlaintext: String): String = {
      dataKeyCryptoCodec(base64PrimaryKeyPlaintext,
                         base64DataKeyPlaintext,
                         Cipher.ENCRYPT_MODE)
  }

  def decryptDataKey(base64PrimaryKeyPlaintext: String,
                     base64DataKeyCiphertext: String): String = {
      dataKeyCryptoCodec(base64PrimaryKeyPlaintext,
                         base64DataKeyCiphertext,
                         Cipher.DECRYPT_MODE)
  }

  def externalClassMethodLoader(jarPath: String,
                                className: String,
                                methodName: String): (Method, URLClassLoader) = {
    val classLoader = new URLClassLoader(Array[URL](new File(jarPath).toURI.toURL))
    val classToLoad = classLoader.loadClass(className)
    val method = classToLoad.getDeclaredMethod(methodName)
    val instance = classToLoad.newInstance()
    (method, instance.asInstanceOf[URLClassLoader])
  }

  def constructKeywhizArgs(args: String): String = {
    "--devTrustStore --url https://keywhiz-service:4444 " + args
  }

  def invokeExternalMethod(method: Method,
                           instance: URLClassLoader,
                           args: String = null): Object = {
    method.invoke(instance)
  }

  val argumentsParser =
    new scopt.OptionParser[BigDLKMSFrontendArguments]("BigDL Keywhiz KMS Frontend") {
    head("BigDL Keywhiz KMS Frontend")
    opt[String]('i', "interface")
      .action((x, c) => c.copy(interface = x))
      .text("network interface of frontend")
    opt[Int]('p', "port")
      .action((x, c) => c.copy(port = x))
      .text("https port of frontend")
    opt[String]('h', "keywhizHost")
      .action((x, c) => c.copy(keywhizHost = x))
      .text("host of keywhiz")
    opt[Int]('r', "keywhizPort")
      .action((x, c) => c.copy(keywhizPort = x))
      .text("port of keywhiz")
    opt[String]('p', "httpsKeyStorePath")
      .action((x, c) => c.copy(httpsKeyStorePath = x))
      .text("https keyStore path")
    opt[String]('w', "httpsKeyStoreToken")
      .action((x, c) => c.copy(httpsKeyStoreToken = x))
      .text("https keyStore token")
    opt[String]('x', "keywhizTrustStorePath")
      .action((x, c) => c.copy(keywhizTrustStorePath = x))
      .text("keywhiz trustStore path")
    opt[String]('m', "keywhizTrustStoreToken")
      .action((x, c) => c.copy(keywhizTrustStoreToken = x))
      .text("keywhiz trustStore password")
    opt[String]('f', "frontendKeywhizConf")
      .action((x, c) => c.copy(frontendKeywhizConf = x))
      .text("keywhiz configuration file path used by frontend")
    opt[String]('j', "keywhizCliJarPath")
      .action((x, c) => c.copy(keywhizCliJarPath = x))
      .text("path of keywhiz cli jar")
    opt[String]('n', "keyProviderJarPath")
      .action((x, c) => c.copy(keyProviderJarPath = x))
      .text("path of key provider jar")
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
    interface: String = "0.0.0.0",
    port: Int = 9876,
    keywhizHost: String = "keywhiz-service",
    keywhizPort: Int = 4444,
    httpsKeyStorePath: String = null,
    httpsKeyStoreToken: String = null,
    keywhizTrustStorePath: String = null,
    keywhizTrustStoreToken: String = null,
    frontendKeywhizConf: String = "/usr/src/app/frontend-keywhiz-conf.yaml",
    keywhizCliJarPath: String =
    "/usr/src/app/cli/target/keywhiz-cli-0.10.2-SNAPSHOT-shaded.jar",
    keyProviderJarPath: String =
    "/usr/src/app/server/target/keywhiz-server-0.10.2-SNAPSHOT-shaded.jar"
)

