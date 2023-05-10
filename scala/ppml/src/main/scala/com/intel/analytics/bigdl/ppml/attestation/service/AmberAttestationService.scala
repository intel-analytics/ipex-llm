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

import javax.net.ssl.SSLContext
import org.apache.http.HttpEntity
import org.apache.http.HttpResponse
import org.apache.http.conn.ssl.NoopHostnameVerifier
import org.apache.http.conn.ssl.SSLConnectionSocketFactory
import org.apache.http.impl.client.CloseableHttpClient
import org.apache.http.impl.client.HttpClientBuilder
import org.apache.http.impl.client.HttpClients
import org.apache.http.ssl.SSLContextBuilder
import org.apache.http.ssl.SSLContexts
import org.apache.http.client.methods.{HttpGet, HttpPost}
import org.apache.http.entity.StringEntity
import org.apache.http.impl.client.HttpClients
import org.apache.http.message.BasicHeader

import scala.util.Random
import scala.util.parsing.json._

import com.intel.analytics.bigdl.ppml.attestation._
import com.intel.analytics.bigdl.ppml.attestation.utils.{AttestationUtil, JsonUtil}

/**
 * Attestation Service provided by Amber
 * @param attestationServerIP Attestation Service IP
 * @param attestationServerPort Attestation Service port
 * @param appID application ID
 * @param apiKey application Key
 */
class AmberAttestationService(attestationServerURL: String, apiKey: String, userReport: String) extends AttestationService {
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

  val ACTION_GET_NONCE = "/appraisal/v1/nonce"
  val ACTION_VERIFY_QUOTE = "/appraisal/v1/attest"
  // Respone keys
  val RES_TOKEN = "token"

  override def register(appID: String): String = "true"

  override def getPolicy(appID: String): String = "true"

  override def setPolicy(policy: JSONObject): String = "true"

  var nonceJson = new JSONObject()
  var nonce = ""
  
  def getNonce(): String = {
    val action: String = ACTION_GET_NONCE
    val getUrl = constructUrl(action)
    var response: String = null
    response = getRequest(getUrl, sslConSocFactory, apiKey)
    nonceJson = new JSONObject(response)
    nonce = nonceJson.getString("val")
    nonce
  }

  def getQuoteFromServer(challenge: String): String = {
    val userReportData = new Array[Byte](16)
    Random.nextBytes(userReportData)
    new String(userReportData)
  }

  override def attestWithServer(quote: String): (Boolean, String) = {
    if (quote == null) {
      Log4Error.invalidInputError(false,
        "Quote should be specified")
    }
    val action: String = ACTION_VERIFY_QUOTE

    val postResult: JSONObject = timing("AmberAttestationService request for VerifyQuote") {
      val postContent = Map[String, Any](
        "quote" -> quote,
        "nonce" -> nonce,
        "userdata" -> userReport
      )
      val postString = JsonUtil.toJson(postContent)
      val postUrl = constructUrl(action)
      var response: String = null
      response = postRequest(postUrl, sslConSocFactory, postString)

      if (response != null && response.startsWith("\ufeff")) {
        response = response.substring(1)
      }
      new JSONObject(response)
    }
    
    val token = postResult.getString(RES_TOKEN)
    val verifyQuoteResult = token.length() > 0
    (verifyQuoteResult, postResult.toString)
  }

  override def attestWithServer(quote: String, policyID: String): (Boolean, String) = {
    if (quote == null) {
      Log4Error.invalidInputError(false,
        "Quote should be specified")
    }
    val action: String = ACTION_VERIFY_QUOTE

    val postResult: JSONObject = timing("AmberAttestationService request for VerifyQuote") {
      val postContent = Map[String, Any](
        "quote" -> quote,
        "nonce" -> nonce,
        "policy" -> policyID,
        "userdata" -> userReport
      )
      val postString = JsonUtil.toJson(postContent)
      val postUrl = constructUrl(action)
      var response: String = null
      response = postRequest(postUrl, sslConSocFactory, postString)

      if (response != null && response.startsWith("\ufeff")) {
        response = response.substring(1)
      }
      new JSONObject(response)
    }
    
    val token = postResult.getString(RES_TOKEN)
    val verifyQuoteResult = token.length() > 0
    (verifyQuoteResult, postResult.toString)
  }

  private def constructUrl(action: String): String = {
    s"$attestationServerURL/$action"
  }

  
  def getRequest(url: String, sslConSocFactory: SSLConnectionSocketFactory, apiKey: String): String = {
    val clientbuilder = HttpClients.custom().setSSLSocketFactory(sslConSocFactory)
    val httpsClient: CloseableHttpClient = clientbuilder.build()
    

    val httpGet = new HttpGet(url)
    httpGet.setHeader(new BasicHeader("Content-Type", "application/json"));
    httpGet.setHeader(new BasicHeader("x-api-key", apiKey));
    val response = httpsClient.execute(httpGet)
    EntityUtils.toString(response.getEntity, "UTF-8")
  }

  def postRequest(url: String, sslConSocFactory: SSLConnectionSocketFactory, apiKey: String): String = {
    val clientbuilder = HttpClients.custom().setSSLSocketFactory(sslConSocFactory)
    val httpsClient: CloseableHttpClient = clientbuilder.build()
    val httpPost = new HttpPost(url)
    httpPost.setHeader(new BasicHeader("Content-Type", "application/json"));
    httpPost.setHeader(new BasicHeader("x-api-key", apiKey));
    val response = httpsClient.execute(httpPost)
    EntityUtils.toString(response.getEntity, "UTF-8")
  }

}
