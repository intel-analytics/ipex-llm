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

package com.intel.analytics.bigdl.ppml.attestation.admin

import com.intel.analytics.bigdl.ppml.examples.Decrypt.timing
import com.intel.analytics.bigdl.ppml.utils.EHSMParams
import com.intel.analytics.bigdl.ppml.utils.HTTPSUtil
import com.intel.analytics.bigdl.ppml.utils.HTTPSUtil.postRequest
import org.apache.http.conn.ssl.{AllowAllHostnameVerifier, SSLConnectionSocketFactory}
import org.json.JSONObject
import scopt.OptionParser

import java.security.SecureRandom
import java.security.cert.X509Certificate
import javax.net.ssl.{SSLContext, TrustManager, X509TrustManager}

import com.intel.analytics.bigdl.ppml.attestation._
import com.intel.analytics.bigdl.ppml.attestation.service.ATTESTATION_CONVENTION
import com.intel.analytics.bigdl.ppml.attestation.utils.AttestationUtil

object RegisterMrEnclave {
  val sslConSocFactory = {
    val sslContext: SSLContext = SSLContext.getInstance("SSL")
    val trustManager: TrustManager = new X509TrustManager() {
      override def checkClientTrusted(chain: Array[X509Certificate], authType: String): Unit = {}

      override def checkServerTrusted(chain: Array[X509Certificate], authType: String): Unit = {}

      override def getAcceptedIssuers(): Array[X509Certificate] = Array.empty
    }
    sslContext.init(null, Array(trustManager), new SecureRandom())
    new SSLConnectionSocketFactory(sslContext, new AllowAllHostnameVerifier())
  }

  def main(args: Array[String]): Unit = {
    case class CmdParams(appID: String = "test",
                         apiKey: String = "test",
                         URL: String = "127.0.0.1:9000",
                         asType: String = ATTESTATION_CONVENTION.MODE_EHSM_KMS,
                         mrEnclave: String = "test",
                         mrSigner: String = "test")

    val cmdParser: OptionParser[CmdParams] = new OptionParser[CmdParams](
      "PPML Attestation Policy Register Cmd tool") {
      opt[String]('i', "appID")
        .text("app id for this app")
        .action((x, c) => c.copy(appID = x))
        .required()
      opt[String]('k', "apiKey")
        .text("app key for this app")
        .action((x, c) => c.copy(apiKey = x))
        .required()
      opt[String]('u', "URL")
        .text("attestation service url, default is 127.0.0.1:9000")
        .action((x, c) => c.copy(URL = x))
        .required()
      opt[String]('t', "asType")
        .text("attestation service type, default is ehsm")
        .action((x, c) => c.copy(asType = x))
      opt[String]('e', "mrEnclave")
        .text("mrEnclave")
        .action((x, c) => c.copy(mrEnclave = x))
        .required()
      opt[String]('s', "mrSigner")
        .text("mrSigner")
        .action((x, c) => c.copy(mrSigner = x))
        .required()
    }
    val params = cmdParser.parse(args, CmdParams()).get
    val appID = params.appID
    val apiKey = params.apiKey
    val mrEnclave = params.mrEnclave
    val mrSigner = params.mrSigner
    val URL = params.URL

    params.asType match {
      case ATTESTATION_CONVENTION.MODE_EHSM_KMS =>
        val action: String = "UploadQuotePolicy"

        val currentTime = System.currentTimeMillis() // ms
        val timestamp = s"$currentTime"
        val ehsmParams = new EHSMParams(appID, apiKey, timestamp)
        ehsmParams.addPayloadElement("mr_enclave", mrEnclave)
        ehsmParams.addPayloadElement("mr_signer", mrSigner)

        val postResult: JSONObject = timing("Request for Register MrEnclave") {
          val postString: String = ehsmParams.getPostJSONString()
          postRequest(constructUrl(action, URL), sslConSocFactory, postString)
        }
        // print policy_Id
        if (postResult == null || !postResult.has("policyId") ||
          postResult.get("policyId") == null || postResult.get("policyId") == "") {
          println("register error")
          return
        }

        println("policy_Id " + postResult.getString("policyId"))
      case ATTESTATION_CONVENTION.MODE_BIGDL =>
        val postResult: JSONObject = timing("BigDLAttestationService request for VerifyQuote") {
          val postContent = Map[String, Any](
            "appID" -> appID,
            "apiKey" -> apiKey,
            "mrEnclave" -> mrEnclave,
            "policyType" -> "SGXMREnclavePolicy"
          )
          val postString = AttestationUtil.mapToString(postContent)
          val postUrl = s"https://$URL/registerPolicy"
          var response: String = HTTPSUtil.retrieveResponse(postUrl, sslConSocFactory, postString)

          if (response != null && response.startsWith("\ufeff")) {
            response = response.substring(1)
          }
          // System.out.println(response)
          new JSONObject(response)
        }
        if (postResult == null || !postResult.has("policyID") ||
          postResult.get("policyID") == null || postResult.get("policyID") == "") {
          println("register error")
          return
        }

        println("policyID " + postResult.getString("policyID"))
    }
  }

  private def constructUrl(action: String, URL: String): String = {
    s"https://$URL/ehsm?Action=$action"
  }
}
