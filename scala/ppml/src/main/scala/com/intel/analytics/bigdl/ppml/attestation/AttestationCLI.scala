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

import org.apache.logging.log4j.LogManager
import scopt.OptionParser

import java.io.{BufferedOutputStream, BufferedInputStream};
import java.io.File;
import java.io.{FileInputStream, FileOutputStream};
import java.util.Base64
import java.security.MessageDigest
import javax.xml.bind.DatatypeConverter

import com.intel.analytics.bigdl.ppml.attestation.generator._
import com.intel.analytics.bigdl.ppml.attestation.service._
import com.intel.analytics.bigdl.ppml.attestation.verifier._

/**
 * Simple Attestation Command Line tool for attestation service
 */
object AttestationCLI {
    def hex(str: String): Array[Byte] = {
      str.sliding(2, 2).toArray.map(Integer.parseInt(_, 16).toByte)
    }

    def sha256(hex: Array[Byte]): Array[Byte] = {
      val md = MessageDigest.getInstance("SHA-256")
      md.update(hex)
      md.digest()
      // val bytes = sha256.grouped(4).flatMap(_.reverse).toArray
      // sha256.map("%02x".format(_))
    }

    def sha512(nonce: String, userReport: String): Array[Byte] = {
      val md = MessageDigest.getInstance("SHA-512")
      md.update(nonce.getBytes())
      md.update(userReport.getBytes())
      md.digest()
    }

    def main(args: Array[String]): Unit = {
        var quote = Array[Byte]()
        val logger = LogManager.getLogger(getClass)
        case class CmdParams(appID: String = "test",
                             apiKey: String = "test",
                             asType: String = ATTESTATION_CONVENTION.MODE_EHSM_KMS,
                             asURL: String = "127.0.0.1:9000",
                             challenge: String = "",
                             policyID: String = "",
                             quoteType: String = QUOTE_CONVENTION.MODE_GRAMINE,
                             apiVersion: String = "2020-10-01",
                             quotePath:String = "",
                             nonce: String = "",
                             proxyHost: String = "",
                             proxyPort: Int = 0,
                             userReport: String = "010203040506")

        val cmdParser: OptionParser[CmdParams] = new OptionParser[CmdParams](
          "PPML Attestation Quote Generation Cmd tool") {
            opt[String]('i', "appID")
              .text("app id for this app")
              .action((x, c) => c.copy(appID = x))
            opt[String]('k', "apiKey")
              .text("app key for this app")
              .action((x, c) => c.copy(apiKey = x))
            opt[String]('u', "asURL")
              .text("attestation service url, default is 127.0.0.1:9000")
              .action((x, c) => c.copy(asURL = x))
            opt[String]('t', "asType")
              .text("attestation service type, default is EHSMKeyManagementService")
              .action((x, c) => c.copy(asType = x))
            opt[String]('c', "challenge")
              .text("challenge to attestation service, defaultly skip bi-attestation")
              .action((x, c) => c.copy(challenge = x))
            opt[String]('o', "policyID")
              .text("policyID of registered MREnclave and MRSigner, defaultly empty")
              .action((x, c) => c.copy(policyID = x))
            opt[String]('p', "userReport")
              .text("userReportData, default is test")
              .action((x, c) => c.copy(userReport = x))
            opt[String]('O', "quoteType")
              .text("quoteType, default is gramine, occlum can be chose")
              .action((x, c) => c.copy(quoteType = x))
            opt[String]('v', "APIVersion")
              .text("APIType, default is 2020-10-01")
              .action((x, c) => c.copy(apiVersion = x))
            opt[String]('n', "nonce")
              .text("nonce, default is ''")
              .action((x, c) => c.copy(userReport = x))
            opt[String]('q', "quotePath")
              .text("quotePath, default is ''")
              .action((x, c) => c.copy(quotePath = x))
            opt[String]("proxyHost")
              .text("proxyHost, default is ''")
              .action((x, c) => c.copy(proxyHost = x))
            opt[Int]("proxyPort")
              .text("proxyPort, default is ''")
              .action((x, c) => c.copy(proxyPort = x.toInt))
        }
        val params = cmdParser.parse(args, CmdParams()).get

        // Attestation Client
        val as = params.asType match {
            case ATTESTATION_CONVENTION.MODE_EHSM_KMS =>
                new EHSMAttestationService(params.asURL.split(":")(0),
                    params.asURL.split(":")(1), params.appID, params.apiKey)
            case ATTESTATION_CONVENTION.MODE_BIGDL =>
                new BigDLAttestationService(params.asURL.split(":")(0),
                    params.asURL.split(":")(1), params.appID, params.apiKey)
            case ATTESTATION_CONVENTION.MODE_DUMMY =>
                new DummyAttestationService()
            case ATTESTATION_CONVENTION.MODE_AZURE =>
                new AzureAttestationService(params.asURL, params.apiVersion,
                 Base64.getUrlEncoder.encodeToString(hex(params.userReport)))
            case ATTESTATION_CONVENTION.MODE_AMBER =>
                new AmberAttestationService(params.asURL, params.apiKey, params.userReport, params.proxyHost, params.proxyPort)
            case _ => throw new AttestationRuntimeException("Wrong Attestation Service type")
        }

        // Generate quote
        val quotePath = params.quotePath
        if (quotePath.length() == 0) {
          val userReportData = params.asType match {
            case ATTESTATION_CONVENTION.MODE_AZURE =>
              sha256(hex(params.userReport))
            case ATTESTATION_CONVENTION.MODE_AMBER =>
              if (params.nonce.length() >0) {
                val nonceResp = as.getNonce()
                sha512(as.nonce, params.userReport)
              } else {
                params.userReport.getBytes
              }
            case _ =>
              params.userReport.getBytes
          }

          val quoteGenerator = params.quoteType match {
            case QUOTE_CONVENTION.MODE_GRAMINE =>
              new GramineQuoteGeneratorImpl()
            case QUOTE_CONVENTION.MODE_OCCLUM =>
              new OcclumQuoteGeneratorImpl()
            case QUOTE_CONVENTION.MODE_TDX =>
              new TDXQuoteGeneratorImpl()
            case _ => throw new AttestationRuntimeException("Wrong quote type")
          }
          quote = quoteGenerator.getQuote(userReportData)
        } else {
          println("[INFO] Using generated quote, only for debug!")
          val quoteFile = new File(quotePath)
          val in = new FileInputStream(quoteFile)
          val bufIn = new BufferedInputStream(in)
          quote = Iterator.continually(bufIn.read()).takeWhile(_ != -1).map(_.toByte).toArray
          bufIn.close()
          in.close()
        }

        val challengeString = params.challenge
        val debug = System.getenv("ATTESTATION_DEBUG")

        if (debug == "true") {
            val quote_base64 = Base64.getEncoder.encodeToString(quote)
            println(s"quote: ${quote_base64}")
        }

        if (challengeString.length() > 0 && params.asType != ATTESTATION_CONVENTION.MODE_DUMMY) {
            val asQuote = params.asType match {
              case ATTESTATION_CONVENTION.MODE_EHSM_KMS =>
                Base64.getDecoder().decode(as.getQuoteFromServer(challengeString))
              case _ => throw new AttestationRuntimeException("Wrong Attestation Service type")
            }
            val quoteVerifier = new SGXDCAPQuoteVerifierImpl()
            quoteVerifier.verifyQuote(asQuote)
        }

        val attResult = params.policyID match {
          case "" => as.attestWithServer(Base64.getEncoder.encodeToString(quote))
          case _ => as.attestWithServer(Base64.getEncoder.encodeToString(quote), params.policyID)
        }
        if (attResult._1) {
            System.out.println("Attestation Success!")
            // Bash success
            System.exit(0)
        } else if (debug == "true") {
          System.out.println("ERROR:Attestation Fail! In debug mode, continue.")
        }
        else {
            System.out.println("Attestation Fail! Application killed!")
            // bash fail
            System.exit(1)
        }
    }
}
