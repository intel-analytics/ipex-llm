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
import java.io.{File, InputStream}
import java.io.{BufferedOutputStream, BufferedInputStream};
import java.security.{KeyStore, SecureRandom}
import javax.net.ssl.{KeyManagerFactory, SSLContext, TrustManagerFactory}
import java.util.{Base64, UUID}

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

import scala.util.parsing.json._

import scala.concurrent.{Await, Future}
import scala.concurrent.duration._

import org.apache.logging.log4j.LogManager
import scopt.OptionParser

import com.intel.analytics.bigdl.ppml.attestation.verifier.SGXDCAPQuoteVerifierImpl
import com.intel.analytics.bigdl.ppml.attestation.utils.{AttestationUtil, FileEncryptUtil}
import com.intel.analytics.bigdl.ppml.attestation.utils.JsonUtil

case class Enroll(appID: String, apiKey: String)

case class PolicyBase(policyID: String, policyType: String)
case class SGXMREnclavePolicy(appID: String, mrEnclave: String) extends Policy
case class SGXMRSignerPolicy(appID: String, mrSigner: String, isvProdID: String) extends Policy

case class Quote(quote: String)

object BigDLRemoteAttestationService {
  implicit val system = ActorSystem("BigDLRemoteAttestationService")
  implicit val materializer = ActorMaterializer()
  implicit val executionContext = system.dispatcher
  implicit val timeout: Timeout = Timeout(100, TimeUnit.SECONDS)

  val quoteVerifier = new SGXDCAPQuoteVerifierImpl()

  var userMap : Map[String, String] = Map.empty
  var policyMap : Map[String, Policy] = Map.empty

  def checkAppIDAndApiKey(enroll: Enroll): Boolean = {
    val userMapRes = userMap.get(enroll.appID).mkString
    ((userMapRes != "") && enroll.apiKey == userMapRes)
  }

  def enroll(): Route = {
    val appID = UUID.randomUUID.toString
    val apiKey = AttestationUtil.generateToken(32)
    userMap += (appID -> apiKey)
    val enroll = new Enroll(appID, apiKey)
    val res = JsonUtil.toJson(enroll)
    complete(res)
  }

  def registerPolicy(msg: String): Route = {
    val policyType = JsonUtil.fromJson(classOf[PolicyBase], msg).policyType
    val curPolicy = policyType match {
      case "SGXMREnclavePolicy" =>
        JsonUtil.fromJson(classOf[SGXMREnclavePolicy], msg)
      case "SGXMRSignerPolicy" =>
        JsonUtil.fromJson(classOf[SGXMRSignerPolicy], msg)
      case _ =>
        null
    }
    if (curPolicy == null) {
      complete(400, "Unsupported policy type.")
    } else {
      val policyID = UUID.randomUUID.toString
      policyMap += (policyID -> curPolicy)
      val res = AttestationUtil.mapToString(Map("policyID" -> policyID))
      complete(200, res)
    }
  }

  def verifyQuote(msg: String): Route = {
    val quoteBase64 = JsonUtil.fromJson(classOf[Quote], msg).quote
    val quote = Base64.getDecoder().decode(quoteBase64.getBytes)
    val verifyQuoteResult = quoteVerifier.verifyQuote(quote)
    if (verifyQuoteResult < 0) {
      val res = "{\"result\":\"" + verifyQuoteResult.toString() + "\"}"
      complete(400, res)
    } else {
      val curPolicyID = JsonUtil.fromJson(classOf[PolicyBase], msg).policyID
      if (curPolicyID == null) {
        val res = "{\"result\":" + verifyQuoteResult.toString() + "}"
        complete(200, res)
      } else {
        val curPolicy = policyMap.get(curPolicyID)
        val appID = JsonUtil.fromJson(classOf[Enroll], msg).appID
        curPolicy match {
          case Some(SGXMREnclavePolicy(policyAppID, policyMREnclave)) =>
            val mrEnclave = AttestationUtil.getMREnclaveFromQuote(quote)
            if (appID == policyAppID && mrEnclave == policyMREnclave) {
              val res = "{\"result\":\"" + verifyQuoteResult.toString() + "\"}"
              complete(200, res)
            } else {
              val res = "{\"result\": -1}"
              complete(400, res)
            }
          case Some(SGXMRSignerPolicy(policyAppID, policyMRSigner, policyISVProdID)) =>
            val mrSigner = AttestationUtil.getMRSignerFromQuote(quote)
            val isvProdID = AttestationUtil.getISVProdIDFromQuote(quote)
            if (appID == policyAppID && mrSigner == policyMRSigner &&
              isvProdID == policyISVProdID) {
              val res = "{\"result\":\"" + verifyQuoteResult.toString() + "\"}"
              complete(200, res)
            } else {
              val res = "{\"result\": -1}"
              complete(400, res)
            }
          case _ =>
            complete(400, "Unsupported policy type.")
        }
      }
    }
  }

  def main(args: Array[String]): Unit = {
    val logger = LogManager.getLogger(getClass)
    case class CmdParams(serviceHost: String = "0.0.0.0",
                          servicePort: String = "9875",
                          httpsKeyStoreToken: String = "token",
                          httpsKeyStorePath: String = "keys/server.p12",
                          httpsEnabled: Boolean = true,
                          basePath: String = "/opt/bigdl-as/",
                          enrollFilePath: String = "data/enrolls.dat",
                          policyFilePath: String = "data/policies.dat",
                          secretKey: String = "bigdl"
                          )

    val cmdParser : OptionParser[CmdParams] =
      new OptionParser[CmdParams]("BigDL Remote Attestation Service") {
        opt[String]('h', "serviceHost")
          .text("Attestation Service Host, default is 0.0.0.0")
          .action((x, c) => c.copy(serviceHost = x))
        opt[String]('p', "servicePort")
          .text("Attestation Service Port, default is 9875")
          .action((x, c) => c.copy(servicePort = x))
        opt[String]('t', "httpsKeyStoreToken")
          .text("KeyStoreToken of https, default is token")
          .action((x, c) => c.copy(httpsKeyStoreToken = x))
        opt[String]("httpsKeyStorePath")
          .text("KeyStorePath of https, default is ./keys/server.p12")
          .action((x, c) => c.copy(httpsKeyStorePath = x))
        opt[String]('k', "secretKey")
          .text("Secret Key to encrypt and decrypt BigDLRemoteAttestation data file")
          .action((x, c) => c.copy(secretKey = x))
        opt[String]('b', "basePath")
          .text("Diretory for data files of BigDL Remote Attestation Service"
            + "default is ./data")
          .action((x, c) => c.copy(basePath = x))
        opt[String]('e', "enrollFilePath")
          .text("Path of base data file to save account information, "
            + "default is BigDLRemoteAttestationService.dat")
          .action((x, c) => c.copy(enrollFilePath = x))
        opt[String]('o', "policyFilePath")
          .text("Path of policy data file, default is BigDLRemoteAttestationServicePolicy.dat")
          .action((x, c) => c.copy(policyFilePath = x))
    }
    val params = cmdParser.parse(args, CmdParams()).get
    val secretKey = params.secretKey
    val enrollFilePath = params.basePath + "/" + params.enrollFilePath
    val policyFilePath = params.basePath + "/" + params.policyFilePath
    val userContent = Await.result(FileEncryptUtil.loadFile(enrollFilePath, secretKey), 5.seconds)
    userMap = AttestationUtil.stringToStrMap(userContent)
    val policyContent = Await.result(FileEncryptUtil.loadFile(policyFilePath, secretKey), 5.seconds)
    policyMap = AttestationUtil.stringToPolicyMap(policyContent)

    val t = new Thread {
      override def run(): Unit = {
        while (true) {
          Thread.sleep(30 * 1000)
          FileEncryptUtil.saveFile(enrollFilePath,
            AttestationUtil.mapToString(userMap), secretKey)
          FileEncryptUtil.saveFile(policyFilePath,
            AttestationUtil.mapToString(policyMap), secretKey)
        }
      }
    }
    t.start()

    val route: Route =
        get {
          path("") {
            val res = s"Welcome to BigDL Remote Attestation Service \n \n" +
            "enroll an account like: " +
            "GET <bigdl_remote_attestation_address>/enroll \n" +
            "register a policy like: " +
            "POST <bigdl_remote_attestation_address>/registerPolicy \n \n" +
            "Your post data should be json format, which contains appID, apiKey and the \n" +
            "MREnclave and MRSigner to be regisrtered, e.g.: \n" +
            "{\"appID\": \"your_app_id\",\"apiKey\": \"your_api_key\"," +
            "\"mrEnclave\": \"2b0376b7...\",\"mrSigner\": \"3ab0ac54...\"} \n \n" +
            "verify your quote like: " +
            "POST <bigdl_remote_attestation_address>/verifyQuote \n" +
            "Your post data should be json format, which contains appID, apiKey and your \n" +
            "BASE64 formatted quote to be verified, and optionally a policy to check, e.g.: \n" +
            "{\"appID\": \"your_app_id\",\"apiKey\": \"your_api_key\"," +
            "\"quote\": \"AwACAAAAAAAJ...\",\"policyID\":\"a_policy_id\"} \n \n"
            complete(res)
          } ~
          path("enroll") {
            logger.info("enroll\n")
            enroll()
          }
        } ~
      post {
        path("registerPolicy") {
          entity(as[String]) { jsonMsg =>
            logger.info("registerPolicy\n")
            val enroll = JsonUtil.fromJson(classOf[Enroll], jsonMsg)
            print(enroll)
            if (checkAppIDAndApiKey(enroll)) {
              registerPolicy(jsonMsg)
            } else {
              complete(400, "Invalid app_id and api_key.")
            }
          }
        } ~
        path("verifyQuote") {
          entity(as[String]) { jsonMsg =>
            logger.info("verifyQuote:\n")
            val enroll = JsonUtil.fromJson(classOf[Enroll], jsonMsg)
            if (checkAppIDAndApiKey(enroll)) {
              verifyQuote(jsonMsg)
            } else {
              complete(400, "Invalid app_id and api_key.")
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
