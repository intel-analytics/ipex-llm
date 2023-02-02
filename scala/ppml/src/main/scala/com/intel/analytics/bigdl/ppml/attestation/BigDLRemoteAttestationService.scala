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
import java.math.BigInteger

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

import org.json4s.jackson.Serialization
import org.json4s._

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

  implicit val formats = DefaultFormats

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
    // JSONObject(map).toString()
    Serialization.write(map)
  }

  def stringToMap(str: String): Map[String, Any] = {
    JSON.parseFull(str) match {
      case Some(map: Map[String, Any]) => map
      case None => Map.empty
    }
  }

  def checkAppIDAndApiKey(filename: String, map: Map[String, Any]): Boolean = {
    if (!map.contains("app_id") || !map.contains("api_key")) {
      false
    } else {
      val appID = map.get("app_id").mkString
      val apiKey = map.get("api_key").mkString
      val fileContent = Await.result(loadFile(filename), 5.seconds)
      val userMap = stringToMap(fileContent)
      val userMapRes = userMap.get(appID).mkString
      if ((userMapRes != "") && apiKey == userMapRes) {
        true
      } else {
        false
      }
    }
  }

  def getMREnclaveFromQuote(quote: Array[Byte]): String = {
    new BigInteger(1, quote.slice(112, 144)).toString(16)
  }

  def getMRSignerFromQuote(quote: Array[Byte]): String = {
    new BigInteger(1, quote.slice(176, 208)).toString(16)
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
    secretKey = params.secretKey

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
            var appID = Random.alphanumeric.take(32).mkString
            var apiKey = Random.alphanumeric.take(32).mkString
            val basePath = params.basePath
            val userContent = Await.result(loadFile(basePath), 5.seconds)
            var userMap = stringToMap(userContent)
            while (userMap.contains(appID)) {
              appID = Random.alphanumeric.take(32).mkString
              apiKey = Random.alphanumeric.take(32).mkString
            }
            userMap += (appID -> apiKey)
            saveFile(basePath, mapToString(userMap))
            val res = "{\"app_id\":\"" + appID + ",\"api_key\":\"" + apiKey + "\"}"
            complete(res)
          }
        } ~
        post {
          path("registePolicy") {
            entity(as[String]) { jsonMsg =>
              logger.info(jsonMsg)
              val msg = stringToMap(jsonMsg)
              if (!checkAppIDAndApiKey(params.basePath, msg)) {
                complete(400, "Invalid app_id and api_key.")
              } else {
                if (!msg.contains("TDX") || msg.get("TDX").mkString == "false") {
                  if (!msg.contains("mr_enclave") || !msg.contains("mr_signer")) {
                    complete(400, "Required parameters are not provided.")
                  }
                  val appID = msg.get("app_id").mkString
                  val mrEnclave = msg.get("mr_enclave").mkString
                  val mrSigner = msg.get("mr_signer").mkString
                  val policyContent = Await.result(loadFile(params.policyPath), 5.seconds)
                  var policyMap = stringToMap(policyContent)
                  var policyID = Random.alphanumeric.take(48).mkString
                  while (policyMap.contains(policyID)) {
                    policyID = Random.alphanumeric.take(48).mkString
                  }
                  val curContent = Map[String, Any] (
                    "app_id" -> appID,
                    "mr_enclave" -> mrEnclave,
                    "mr_signer" -> mrSigner
                  )
                  policyMap += (policyID -> curContent)
                  saveFile(params.policyPath, mapToString(policyMap))
                  val res = mapToString(Map("policy_id" -> policyID))
                  complete(200, res)
                } else {
                  //TODO: TDX policy
                  complete(400, "Not Implemented.")
                }
              }
            }
          } ~
          path("verifyQuote") {
            entity(as[String]) { jsonMsg =>
              logger.info(jsonMsg)
              val msg = stringToMap(jsonMsg)
              if (!checkAppIDAndApiKey(params.basePath, msg)) {
                complete(400, "Invalid app_id and api_key.")
              } else {
                if (!msg.contains("quote")) {
                  complete(400, "Required parameters are not provided.")
                } else {
                  val quoteBase64 = msg.get("quote").mkString
                  val quote = Base64.getDecoder().decode(quoteBase64.getBytes)
                  
                  val verifyQuoteResult = quoteVerifier.verifyQuote(quote)
                  
                  if (verifyQuoteResult >= 0) {
                    if (msg.contains("policy_id")) {
                      val appID = msg.get("app_id").mkString
                      val mrEnclave = getMREnclaveFromQuote(quote)
                      val mrSigner = getMRSignerFromQuote(quote)

                      val policyID = msg.get("policy_id").mkString
                      val fileContent = Await.result(loadFile(params.policyPath), 5.seconds)
                      val policyMap = stringToMap(fileContent)
                      val policyContent: Map[String, Any] = policyMap.get(policyID) match {
                        case Some(map: Map[String, Any]) => map
                        case None => Map.empty
                      }
                      if (appID == policyContent.get("app_id").mkString &&
                        mrEnclave == policyContent.get("mr_enclave").mkString &&
                        mrSigner == policyContent.get("mr_signer").mkString) {
                        val res = "{\"result\":\"" + verifyQuoteResult.toString() + "\"}"
                        complete(200, res)
                      } else {
                        val res = "{\"result\": -1}"
                        complete(400, res)
                      }
                    } else {
                      val res = "{\"result\":" + verifyQuoteResult.toString() + "}"
                      complete(200, res)
                    }
                  } else {
                    val res = "{\"result\":\"" + verifyQuoteResult.toString() + "\"}"
                    complete(400, res)
                  }
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
