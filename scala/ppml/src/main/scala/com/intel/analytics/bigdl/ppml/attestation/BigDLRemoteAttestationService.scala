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
package com.intel.analytics.bigdl.ppml.attestation

import java.util.concurrent.{LinkedBlockingQueue, TimeUnit}
import java.io.{File, InputStream, PrintWriter}
import java.io.{FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}
import java.io.{BufferedOutputStream, BufferedInputStream};
import java.security.{KeyStore, SecureRandom}
import javax.crypto.{Cipher, SecretKey, SecretKeyFactory}
import javax.crypto.spec.{PBEKeySpec, SecretKeySpec}
import javax.net.ssl.{KeyManagerFactory, SSLContext, TrustManagerFactory}
import java.util.Base64

import akka.actor.ActorSystem
import akka.http.scaladsl.{ConnectionContext, Http, HttpsConnectionContext}
import akka.Done
import akka.http.scaladsl.server.Route
import akka.http.scaladsl.server.Directives._
import akka.http.scaladsl.model.StatusCodes
import akka.stream.ActorMaterializer
import akka.util.Timeout

import akka.http.scaladsl.marshallers.sprayjson.SprayJsonSupport._
import spray.json.DefaultJsonProtocol._

import scala.io.{Source, StdIn}
import scala.util.Random
import scala.util.parsing.json._

import scala.concurrent.{Await, Future}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._

import org.apache.logging.log4j.LogManager
import scopt.OptionParser

import com.intel.analytics.bigdl.ppml.attestation.verifier.SGXDCAPQuoteVerifierImpl

object BigDLRemoteAttestationService {

  implicit val system = ActorSystem("BigDLRemoteAttestationService")
  implicit val materializer = ActorMaterializer()
  implicit val executionContext = system.dispatcher
  implicit val timeout: Timeout = Timeout(100, TimeUnit.SECONDS)

  final case class Quote(quote: String)
  implicit val quoteFormat = jsonFormat1(Quote)

  final case class Result(result: Int)
  implicit val resultFormat = jsonFormat1(Result)

  val quoteVerifier = new SGXDCAPQuoteVerifierImpl()

  private val salt = Array[Byte](0,1,2,3,4,5,6,7)
  private val iterations = 65536
  private val keySize = 256
  private var secretKey = "password"
  private val algorithm = "PBKDF2WithHmacSHA256"

  def encrypt(data: Array[Byte]): Array[Byte] = {
    val factory = SecretKeyFactory.getInstance(algorithm)
    val spec = new PBEKeySpec(secretKey.toCharArray, salt, iterations, keySize)
    val key = factory.generateSecret(spec).getEncoded()
    val keySpec = new SecretKeySpec(key, "AES")

    val cipher = Cipher.getInstance("AES/ECB/PKCS5Padding")
    cipher.init(Cipher.ENCRYPT_MODE, keySpec)
    cipher.doFinal(data)
  }

  def decrypt(encryptedData: Array[Byte]): Array[Byte] = {
    val factory = SecretKeyFactory.getInstance(algorithm)
    val spec = new PBEKeySpec(secretKey.toCharArray, salt, iterations, keySize)
    val key = factory.generateSecret(spec).getEncoded()
    val keySpec = new SecretKeySpec(key, "AES")

    val cipher = Cipher.getInstance("AES/ECB/PKCS5Padding")
    cipher.init(Cipher.DECRYPT_MODE, keySpec)
    cipher.doFinal(encryptedData)
  }

  def saveFile(filename: String, content: String): Future[Unit] = Future {
    val file = new File(filename)
    if (!file.exists()){
      file.createNewFile()
    }
    val encryptedContent = encrypt(content.getBytes("UTF-8"))
    val out = new BufferedOutputStream(new FileOutputStream(file))
    out.write(encryptedContent)
    out.close()
  }

  def loadFile(filename: String): Future[String] = Future {
    val file = new File(filename)
    if (!file.exists()){
      ""
    } else {
      val in = new FileInputStream(file)
      val bufIn = new BufferedInputStream(in)
      val encryptedContent = Iterator.continually(bufIn.read()).takeWhile(_ != -1).map(_.toByte).toArray
      bufIn.close()
      in.close()
      val decryptedContent = new String(decrypt(encryptedContent), "UTF-8")
      decryptedContent
    }
  }

  def mapToString(map: Map[String, Any]): String = {
    JSONObject(map).toString()
  }

  def stringToMap(str: String): Map[String, Any] = {
    JSON.parseFull(str) match {
      case Some(map: Map[String, Any]) => map
      case None => Map.empty
    }
  }

  def main(args: Array[String]): Unit = {

    val logger = LogManager.getLogger(getClass)
    case class CmdParams(serviceHost: String = "0.0.0.0",
                          servicePort: String = "9875",
                          httpsKeyStoreToken: String = "token",
                          httpsKeyStorePath: String = "./key",
                          httpsEnabled: Boolean = false,
                          basePath: String = "./BigDLRemoteAttestationService.dat",
                          policyPath: String = "./BigDLRemoteAttestationServicePolicy.dat",
                          secretKey: String = "password"
                          )

    val cmdParser : OptionParser[CmdParams] =
      new OptionParser[CmdParams]("BigDL Remote Attestation Service") {
        opt[String]('h', "serviceHost")
          .text("Attestation Service Host, default is 0.0.0.0")
          .action((x, c) => c.copy(serviceHost = x))
        opt[String]('p', "servicePort")
          .text("Attestation Service Port, default is 9875")
          .action((x, c) => c.copy(servicePort = x))
        opt[Boolean]('s', "httpsEnabled")
          .text("Whether enable https, default is false")
          .action((x, c) => c.copy(httpsEnabled = x))
        opt[String]('t', "httpsKeyStoreToken")
          .text("KeyStoreToken of https, default is token")
          .action((x, c) => c.copy(httpsKeyStoreToken = x))
        opt[String]('h', "httpsKeyStorePath")
          .text("KeyStorePath of https, default is ./key")
          .action((x, c) => c.copy(httpsKeyStorePath = x))
        opt[String]('k', "secretKey")
          .text("Secret Key to encrypt and decrypt BigDLRemoteAttestation data file")
          .action((x, c) => c.copy(secretKey = x))  
        opt[String]('b', "basePath")
          .text("Path of base data file to save user information, "
            + "default is ./BigDLRemoteAttestationService.dat")
          .action((x, c) => c.copy(basePath = x))
        
    }
    val params = cmdParser.parse(args, CmdParams()).get

    val route: Route =
        get {
          path("") {
            val res = s"Welcome to BigDL Remote Attestation Service \n \n" +
            "enroll an account like: " +
            "GET <bigdl_remote_attestation_address>/enroll \n" +
            "verify your quote like: " +
            "POST <bigdl_remote_attestation_address>/verifyQuote \n"
            complete(res)
          } ~
          path("enroll") {
            var app_id = Random.alphanumeric.take(12).mkString
            var api_key = Random.alphanumeric.take(16).mkString
            val basePath = params.basePath
            val userContent = Await.result(loadFile(basePath), 5.seconds)
            var userMap = stringToMap(userContent)
            while (userMap.contains(app_id)) {
              app_id = Random.alphanumeric.take(32).mkString
              api_key = Random.alphanumeric.take(32).mkString
            }
            userMap += (app_id -> api_key)
            saveFile(basePath, mapToString(userMap))
            val res = "{\"app_id\":\"" + app_id + ",\"api_key\":\"" + api_key + "\"}"
            complete(res)
          }
        } ~
        post {
          path("verifyQuote") {
            entity(as[String]) { jsonMsg =>
              logger.info(jsonMsg)
              val msg = stringToMap(jsonMsg)
              if (!msg.contains("app_id") || !msg.contains("api_key") || !msg.contains("quote")) {
                complete(400, "Required parameters are not provided.")
              } else {
                val appID = msg.get("app_id").mkString
                val apiKey = msg.get("api_key").mkString
                val quote = msg.get("quote").mkString
                val basePath = params.basePath
                val userContent = Await.result(loadFile(basePath), 5.seconds)
                val userMap = stringToMap(userContent)
                val userMapRes = userMap.get(appID).mkString
                if (userMapRes != "" && apiKey == userMapRes) {
                  val verifyQuoteResult = quoteVerifier.verifyQuote(
                    Base64.getDecoder().decode(quote.getBytes))
                  val res = new Result(verifyQuoteResult)
                  if (verifyQuoteResult >= 0) {
                    complete(200, res)
                  } else {
                    complete(400, res)
                  }
                } else {
                  print(userMap.get(appID).toString())
                  complete(400, "Invalid app_id and api_key.")
                }
              }
            }
          }
        }

    val serviceHost = params.serviceHost
    val servicePort = params.servicePort
    val servicePortInt = servicePort.toInt
    if (params.httpsEnabled) {
      val serverContext = defineServerContext(params.httpsKeyStoreToken,
        params.httpsKeyStorePath)
      val bindingFuture = Http().bindAndHandle(route,
       serviceHost, servicePortInt, connectionContext = serverContext)
      println("Server online at https://%s:%s/\n".format(serviceHost, servicePort) +
        "Press Ctrl + C to stop...")
    } else {
      val bindingFuture = Http().bindAndHandle(route, serviceHost, servicePortInt)
      println("Server online at http://%s:%s/\n".format(serviceHost, servicePort) +
        "Press Ctrl + C to stop...")
    }
  }

  def defineServerContext(httpsKeyStoreToken: String,
                          httpsKeyStorePath: String): ConnectionContext = {
    val token = httpsKeyStoreToken.toCharArray

    val keyStore = KeyStore.getInstance("PKCS12")
    val keystoreInputStream = new File(httpsKeyStorePath).toURI().toURL().openStream()

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
