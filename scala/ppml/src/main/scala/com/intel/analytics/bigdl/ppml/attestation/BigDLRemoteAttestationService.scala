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
import java.security.{KeyStore, SecureRandom}
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

import scala.io.StdIn

import scala.concurrent.Future

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

  def main(args: Array[String]): Unit = {

    val logger = LogManager.getLogger(getClass)
    case class CmdParams(serviceHost: String = "0.0.0.0",
                          servicePort: String = "9875",
                          httpsKeyStoreToken: String = "token",
                          httpsKeyStorePath: String = "./key",
                          httpsEnabled: Boolean = false
                          )

    val cmdParser : OptionParser[CmdParams] =
      new OptionParser[CmdParams]("BigDL Remote Attestation Service") {
        opt[String]('h', "serviceHost")
          .text("Attestation Service Host")
          .action((x, c) => c.copy(serviceHost = x))
        opt[String]('p', "servicePort")
          .text("Attestation Service Port")
          .action((x, c) => c.copy(servicePort = x))
        opt[Boolean]('s', "httpsEnabled")
          .text("httpsEnabled")
          .action((x, c) => c.copy(httpsEnabled = x))
        opt[String]('t', "httpsKeyStoreToken")
          .text("httpsKeyStoreToken")
          .action((x, c) => c.copy(httpsKeyStoreToken = x))
        opt[String]('h', "httpsKeyStorePath")
          .text("httpsKeyStorePath")
          .action((x, c) => c.copy(httpsKeyStorePath = x))
    }
    val params = cmdParser.parse(args, CmdParams()).get

    val route: Route =
        get {
          path("") {
            val res = s"Welcome to BigDL Remote Attestation Service \n \n" +
            "verify your quote like: " +
            "POST <bigdl_remote_attestation_address>/verifyQuote \n"
            complete(res)
          }
        } ~
        post {
          path("verifyQuote") {
            entity(as[Quote]) { quoteMsg =>
              logger.info(quoteMsg)
              val verifyQuoteResult = quoteVerifier.verifyQuote(
                Base64.getDecoder().decode(quoteMsg.quote.getBytes))
              val res = new Result(verifyQuoteResult)
              if (verifyQuoteResult >= 0) {
                complete(200, res)
              } else {
                complete(400, res)
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
      // StdIn.readLine()
      // bindingFuture
      //   .flatMap(_.unbind())
      //   .onComplete(_ => system.terminate())
    } else {
      val bindingFuture = Http().bindAndHandle(route, serviceHost, servicePortInt)
      println("Server online at http://%s:%s/\n".format(serviceHost, servicePort) +
        "Press Ctrl + C to stop...")
      // StdIn.readLine()
      // bindingFuture
      //   .flatMap(_.unbind())
      //   .onComplete(_ => system.terminate())
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
