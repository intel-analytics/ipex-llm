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

import akka.actor.{ActorRef, ActorSystem, Props}
import akka.http.scaladsl.Http
import akka.http.scaladsl.server.Directives.{complete, path, _}
import akka.pattern.ask
import akka.stream.ActorMaterializer
import akka.util.Timeout
import org.slf4j.LoggerFactory

import scala.collection.mutable
import scala.concurrent.Await
import com.intel.analytics.bigdl.dllib.utils.Log4Error
import sys.process._

import com.intel.analytics.bigdl.ppml.utils.Supportive

import com.intel.analytics.bigdl.ppml.kms.common.{BigDLKMServerUtil, SecretStore}

import java.nio.charset.StandardCharsets

object BigDLKeyManagementServer extends Supportive {
  val logger = LoggerFactory.getLogger(getClass)
  val name = "bigdl-key-management-server"
  Class.forName("org.sqlite.JDBC")
  implicit val system = ActorSystem(name)
  implicit val materializer = ActorMaterializer()
  implicit val executionContext = system.dispatcher
  implicit val secretStore = new SecretStore
  implicit val secretThreshold = 3
  // if obtained is not "", use k8s secret from env as root key
  // else use secret sharing schema
  implicit var rootKey: String = sys.env.getOrElse("ROOT_KEY", "")

  def main(args: Array[String]): Unit = {
    val arguments = timing("parse arguments") {
      argumentsParser.parse(args, BigDLKeyManagementArguments()) match {
        case Some(arguments) => logger.info(s"starting with $arguments"); arguments
        case None => argumentsParser.failure("miss args, please see the usage info"); null
      }
    }
    val url = "jdbc:sqlite:" + arguments.dbFilePath
    val route = timing("initialize https route") {
      path("") {
        timing("welcome") {
          val response = s"welcome to $name \n \n" +
          "please request/recover root key before using other APIs: \n" +
          "rotate (or generate) a root key: GET /rootKey/rotate \n" +
          "recover root key: PUT /rootKey?secret=a_secret_string_of_the_splited_root_key \n" +
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
                BigDLKMServerUtil.login(user, token, url)
                val encryptedPrimaryKey = {
                  val base64AES256Key: String = BigDLKMServerUtil.generateAESKey(256)
                  BigDLKMServerUtil.encryptKey(rootKey, base64AES256Key)
                }
                BigDLKMServerUtil.saveKey2DB(user, primaryKeyName, encryptedPrimaryKey, url)
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
                BigDLKMServerUtil.login(user, token, url)
                val encryptedDataKey: String = {
                  val encryptedPrimaryKey = BigDLKMServerUtil.queryKeyFromDB(user,
                    primaryKeyName, url).get
                  Log4Error.invalidOperationError(encryptedPrimaryKey != null,
                    "wrong primary key")
                  val primaryKeyPlainText = BigDLKMServerUtil.decryptKey(rootKey,
                    encryptedPrimaryKey)
                  val base64AES128Key = BigDLKMServerUtil.generateAESKey(128)
                  BigDLKMServerUtil.encryptKey(primaryKeyPlainText, base64AES128Key)
                }
                BigDLKMServerUtil.saveKey2DB(user, dataKeyName, encryptedDataKey, url)
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
                Log4Error.invalidOperationError(rootKey != "",
                  "empty root key! Need initilization first")
                BigDLKMServerUtil.saveUser2DB(userName, token, url)
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
                BigDLKMServerUtil.login(user, token, url)
                val base64DataKeyPlainText: String = {
                  val encryptedPrimaryKey = BigDLKMServerUtil.queryKeyFromDB(user,
                    primaryKeyName, url).get
                  Log4Error.invalidOperationError(encryptedPrimaryKey != null,
                    "wrong primary key")
                  val encryptedDataKey = BigDLKMServerUtil.queryKeyFromDB(user,
                    dataKeyName, url).get
                  val primaryKeyPlainText = BigDLKMServerUtil.decryptKey(rootKey,
                    encryptedPrimaryKey)
                  BigDLKMServerUtil.decryptKey(primaryKeyPlainText, encryptedDataKey)
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
      } ~ path("rootKey" / "rotate") {
        get {
          timing("rotate root key") {
            try {
              rootKey = BigDLKMServerUtil.generateAESKey(256)
              val secrets: String = BigDLKMServerUtil
                .splitRootKey(rootKey, secretThreshold)
              complete(s"root key is rotated! \n" +
                "please save secrets: " + secrets +
                s"need $secretThreshold of 5 secrets to recover root key.")
            } catch {
              case e: Exception =>
                e.printStackTrace()
                complete(500, e.getMessage + "\n please rotate a root key like: " +
                  "GET /rootKey/rotate")
            }
          }
        }
      } ~ path("rootKey" / "recover") {
        put {
          entity(as[String]) { secret =>
              // need at least $secretThreshold of 5 secrets to restore the root key
              timing("recover root key from secrets: " +
                s"${secretStore.count + 1}/$secretThreshold") {
                  try {
                    Log4Error.invalidOperationError(rootKey == "",
                      "Root Key exists and cannot repeat initialization")
                    secretStore.addSecret(secret)
                    if (secretStore.count >= secretThreshold) {
                      try {
                        rootKey = BigDLKMServerUtil
                          .recoverRootKey(secretStore.getSecrets, secretThreshold)
                        Log4Error.invalidOperationError(BigDLKMServerUtil
                          .isBase64(rootKey),
                          "recover failure! wrong secret or try other ones")
                        complete(s"recover root key and initialize server successfully!")
                      } catch {
                        case e: Exception =>
                          e.printStackTrace()
                          complete(500, e.getMessage + "\n join splits failure!" +
                            s"secrets may be wrong! please resend them 0/$secretThreshold")
                      } finally {
                        secretStore.clear
                      }
                    } else {
                      complete(s"received ${secretStore.count}/$secretThreshold shares")
                    }
                  } catch {
                    case e: Exception =>
                      rootKey = ""
                      e.printStackTrace()
                      complete(500, e.getMessage + "\n please recover a root key like: " +
                        "PUT /rootKey?secret=a_secret_string_of_the_splited_root_key")
                  }
              }
            }
          }
        }
    }

    val serverContext = BigDLKMServerUtil.defineServerContext(
      arguments.httpsKeyStoreToken, arguments.httpsKeyStorePath)
    Http().bindAndHandle(route, arguments.ip, port = arguments.port,
      connectionContext = serverContext)
    logger.info(s"$name started at https://${arguments.ip}:${arguments.port}")
  }

  val argumentsParser =
   new scopt.OptionParser[BigDLKeyManagementArguments](name) {
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
}

case class BigDLKeyManagementArguments(
  ip: String = "0.0.0.0",
  port: Int = 9875,
  dbFilePath: String = "/ppml/data/kms.db",
  httpsKeyStorePath: String = null,
  httpsKeyStoreToken: String = null
)

