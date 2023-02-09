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


package com.intel.analytics.bigdl.ppml.attestation.service

import com.intel.analytics.bigdl.dllib.utils.Log4Error
import com.intel.analytics.bigdl.ppml.utils.HTTPSUtil.retrieveResponse
import org.apache.logging.log4j.LogManager
import org.json.JSONObject
import javax.net.ssl.SSLContext
import org.apache.http.conn.ssl.AllowAllHostnameVerifier
import org.apache.http.conn.ssl.SSLConnectionSocketFactory
import org.apache.http.ssl.SSLContextBuilder
import org.apache.http.ssl.SSLContexts
import javax.net.ssl.X509TrustManager
import java.security.cert.X509Certificate
import javax.net.ssl.TrustManager
import org.apache.http.util.EntityUtils
import java.security.SecureRandom
import java.util.Base64
import java.util.Arrays;

import com.azure.core.util.BinaryData
import com.azure.security.attestation.AttestationClientBuilder
import com.azure.security.attestation.models.AttestationOptions
import com.azure.security.attestation.models.AttestationResult
import com.azure.security.attestation.models.AttestationData
import com.azure.security.attestation.models.AttestationDataInterpretation

import com.intel.analytics.bigdl.ppml.attestation._

/**
 * Microsoft Azure Attestation Service
 * @param maaProviderURL
 * @param apiVersion
 */
class AzureAttestationService(maaProviderURL: String, apiVersion: String, userReportData: String)
  extends AttestationService {

  val logger = LogManager.getLogger(getClass)

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

  // Quote
  val PAYLOAD_QUOTE = "quote"

  // Respone keys
  val RES_TOKEN = "token"

  override def register(appID: String): String = "true"

  override def getPolicy(appID: String): String = "true"

  override def setPolicy(policy: JSONObject): String = "true"

  def getQuoteFromServer(challenge: String): String = "true"

  override def attestWithServer(quote: String): (Boolean, String) = {
    if (quote == null) {
      Log4Error.invalidInputError(false,
        "Quote should be specified")
    }
    val quote_plain=Base64.getDecoder().decode(quote.getBytes)
    val quote_url=Base64.getUrlEncoder.encodeToString(quote_plain)
    val postResult: JSONObject = timing("AzureAttestationService request for VerifyQuote") {
      val postString: String = "{\"quote\": \"" + quote_url + "\", \"runtimeData\": " +
      "{\"data\": \"" + userReportData + "\",\"dataType\": \"Binary\"}}"
      System.out.println(postString)
      System.out.println(constructUrl())
      val response: String = retrieveResponse(constructUrl(), sslConSocFactory, postString)
      new JSONObject(response)
    }

    System.out.println(postResult.toString)
    val token = postResult.getString(RES_TOKEN)
    val verifyQuoteResult = token.length() > 0
    (verifyQuoteResult, postResult.toString)
  }

  override def attestWithServer(quote: String, policyID: String): (Boolean, String) = {
    attestWithServer(quote)
  }

  private def constructUrl(): String = {
    s"$maaProviderURL/attest/SgxEnclave?api-version=$apiVersion"
  }
}